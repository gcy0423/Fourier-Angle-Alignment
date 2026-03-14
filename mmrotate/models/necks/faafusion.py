# mmrotate/models/necks/faafusion.py
from ..builder import ROTATED_NECKS
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.necks.fpn import FPN
from mmcv.runner import auto_fp16
from typing import List


class FAAFusion(nn.Module):
    """
    Lightweight FAMFusion with channel reduction and fold normalization.

    Key improvements:
      - Project full C-channel input to c_mid channels once (not per-channel!)
      - Normalize fold output by overlap count
      - Keep spatial rotation alignment and LayerScale

    Args:
        m (int): Local window size (must be odd). Default: 7.
        c_mid (int): Intermediate channel dimension after 1x1 projection. Default: 16.
        eps (float): Small value for numerical stability. Default: 1e-8.
        layer_scale_init_value (float): Init value for LayerScale. Default: 1e-5.

    Inputs:
        x_high (Tensor): [B, C, H_h, W_h]
        x_low (Tensor):  [B, C, H_l, W_l]

    Output:
        fused (Tensor): [B, C, H_l, W_l]
    """

    def __init__(
            self,
            m: int = 7,
            c_mid: int = 16,
            eps: float = 1e-8,
            layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()
        self.m = m
        self.c_mid = c_mid
        self.eps = eps

        # Learnable LayerScale: per-channel scalar, initialized small
        self.layer_scale = nn.Parameter(
            torch.full((1, 1, 1, 1), layer_scale_init_value),
            requires_grad=True
        )

        # CORRECT: Project full C channels → c_mid (once per feature map)
        self.proj_low = nn.Conv2d(in_channels=256, out_channels=c_mid, kernel_size=1, bias=False)  # assuming C=256
        self.proj_high = nn.Conv2d(in_channels=256, out_channels=c_mid, kernel_size=1, bias=False)
        self.recon = nn.Conv2d(in_channels=c_mid, out_channels=256, kernel_size=1, bias=False)  # reconstruct to C

        self._init_freq_grids(m)

    def _init_freq_grids(self, m: int):
        h_freq = torch.fft.fftfreq(m, d=1.0) * m
        w_freq = torch.fft.fftfreq(m, d=1.0) * m
        h_grid, w_grid = torch.meshgrid(h_freq, w_freq)  # [m, m]

        rho = torch.sqrt(h_grid ** 2 + w_grid ** 2)
        theta = torch.atan2(h_grid, w_grid)
        theta = (theta + 2 * math.pi) % (2 * math.pi)

        mask = rho > self.eps
        self.register_buffer('valid_thetas', theta[mask])
        self.register_buffer('valid_rhos', rho[mask])
        self.register_buffer('mask_flat', mask.view(-1))

    def _estimate_main_direction(self, x_local: torch.Tensor) -> torch.Tensor:
        """Estimate dominant orientation from magnitude spectrum.
        x_local: [Bn, 1, m, m]
        Returns: [Bn]
        """
        Bn, _, m, _ = x_local.shape
        device = x_local.device

        x_fft = torch.fft.fft2(x_local.squeeze(1), norm='ortho')
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shifted.abs() + self.eps

        mag_flat = mag.view(Bn, -1)
        mag_valid = mag_flat[:, self.mask_flat]
        rho_valid = self.valid_rhos.to(device)

        weighted_energy = mag_valid * rho_valid.unsqueeze(0)
        max_idx = torch.argmax(weighted_energy, dim=1)
        theta_e = self.valid_thetas.to(device)[max_idx]
        return theta_e

    def _rotate_spatial_patch(self, patch: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        K, _, m, _ = patch.shape
        device = patch.device

        cos_t = torch.cos(theta).view(K, 1, 1)
        sin_t = torch.sin(theta).view(K, 1, 1)

        center = (m - 1) / 2.0
        rot_mat = torch.zeros(K, 2, 3, device=device)
        rot_mat[:, 0, 0] = cos_t.squeeze()
        rot_mat[:, 0, 1] = -sin_t.squeeze()
        rot_mat[:, 1, 0] = sin_t.squeeze()
        rot_mat[:, 1, 1] = cos_t.squeeze()
        rot_mat[:, 0, 2] = center - cos_t.squeeze() * center + sin_t.squeeze() * center
        rot_mat[:, 1, 2] = center - sin_t.squeeze() * center - cos_t.squeeze() * center

        grid = F.affine_grid(rot_mat, patch.size(), align_corners=False)
        rotated = F.grid_sample(patch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return rotated

    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        B, C, H_l, W_l = x_low.shape
        # assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        _, _, H_h, W_h = x_high.shape
        device = x_low.device

        # Step 1: Upsample x_high to low resolution
        if (H_h, W_h) != (H_l, W_l):
            x_high_up = F.interpolate(x_high, size=(H_l, W_l), mode='bilinear', align_corners=False)
        else:
            x_high_up = x_high

        # Step 2: Project to c_mid
        xl_proj = self.proj_low(x_low)  # [B, c_mid, H_l, W_l]
        xh_proj = self.proj_high(x_high_up)  # [B, c_mid, H_l, W_l]

        #pad = self.m // 2
        pad = 0
        #N = H_l * W_l  # number of spatial positions
        N = (H_l - self.m + 1) * (W_l - self.m + 1)  # number of spatial positions

        # We'll accumulate rotated high features in c_mid space
        xh_aligned_cmid = torch.zeros_like(xh_proj)  # [B, c_mid, H_l, W_l]

        # Step 3: Process each of the c_mid channels
        for c in range(self.c_mid):
            # Extract single channel from projected features
            xl_c = xl_proj[:, c:c + 1]  # [B, 1, H_l, W_l]
            xh_c = xh_proj[:, c:c + 1]  # [B, 1, H_l, W_l]

            # Unfold into patches
            xl_unfold = F.unfold(xl_c, kernel_size=self.m, stride=1, padding=pad)  # [B, m*m, N]
            xh_unfold = F.unfold(xh_c, kernel_size=self.m, stride=1, padding=pad)  # [B, m*m, N]

            # Reshape to [B*N, 1, m, m]
            xl_patches = xl_unfold.transpose(1, 2).reshape(B * N, 1, self.m, self.m)
            xh_patches = xh_unfold.transpose(1, 2).reshape(B * N, 1, self.m, self.m)

            # Estimate directions from projected features (now in c_mid space)
            theta_low = self._estimate_main_direction(xl_patches)  # [B*N]
            theta_high = self._estimate_main_direction(xh_patches)  # [B*N]

            theta_low_norm = torch.remainder(theta_low, math.pi)
            theta_high_norm = torch.remainder(theta_high, math.pi)
            theta_ = theta_low_norm - theta_high_norm  # [B*N]

            # Rotate high patch to align with low
            xh_rotated = self._rotate_spatial_patch(xh_patches, theta_)  # [B*N, 1, m, m]

            # Fold back to feature map
            xh_rotated_flat = xh_rotated.reshape(B, N, -1).transpose(1, 2)  # [B, m*m, N]
            xh_aligned_map = F.fold(
                xh_rotated_flat,
                output_size=(H_l, W_l),
                kernel_size=self.m,
                stride=1,
                padding=pad
            )  # [B, 1, H_l, W_l]

            # Normalize (optional but recommended)
            ones = torch.ones(1, 1, H_l, W_l, device=device)
            ones_unfold = F.unfold(ones, kernel_size=self.m, stride=1, padding=pad)
            ones_fold = F.fold(ones_unfold, output_size=(H_l, W_l), kernel_size=self.m, stride=1, padding=pad)
            xh_aligned_map = xh_aligned_map / (ones_fold + self.eps)
            # Store in c_mid-aligned tensor
            xh_aligned_cmid[:, c:c + 1] = xh_aligned_map

        # Step 4: Reconstruct from c_mid to original C channels
        xh_recon = self.recon(xh_aligned_cmid)  # [B, C, H_l, W_l]

        # Step 5: Apply LayerScale and fuse
        x_high_modulated = self.layer_scale * xh_recon + x_high_up
        fused = x_low + x_high_modulated
        return fused


@ROTATED_NECKS.register_module()
class FAAFusionFPN(FPN):
    """Flexible FPN with per-level fusion strategy.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        fusion_modes (list[str]): Fusion mode for each top-down fusion step.
            Length must be `num_outs - 1` (or `backbone_levels - 1` if no extra levels).
            Each element is either 'add' (original FPN) or 'fam' (use FAMFusionK2K).
        start_level (int): Index of the start input backbone level.
        end_level (int): Index of the end input backbone level.
        add_extra_convs (bool | str): Same as FPN.
        ... (other args same as FPN)
        fam_cfg (dict): Config for FAMFusionK2K. Default: dict(m=7, layer_scale_init_value=1e-5)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                 fam_cfg=dict(m=7, c_mid=64)):
        # Call parent FPN __init__
        super(FAAFusionFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg)

        # Validate fusion_modes
        backbone_levels = self.backbone_end_level - self.start_level
        expected_fusion_steps = backbone_levels - 1  # e.g., 4 levels → 3 fusion steps
        assert len(fusion_modes) == expected_fusion_steps, \
            f"fusion_modes length ({len(fusion_modes)}) must be {expected_fusion_steps} (backbone_levels - 1)"

        for mode in fusion_modes:
            assert mode in ['add', 'faa'], f"Invalid fusion mode: {mode}"

        self.fusion_modes = fusion_modes

        # Build FAM modules only where needed
        self.fam_modules = nn.ModuleList()
        for mode in fusion_modes:
            if mode == 'faa':
                self.fam_modules.append(FAAFusion(**fam_cfg))
            else:
                self.fam_modules.append(None)  # placeholder

    @auto_fp16()
    def forward(self, inputs):
        """Forward function with flexible fusion."""
        assert len(inputs) == len(self.in_channels)

        # Step 1: Build laterals (same as FPN)
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Step 2: Top-down fusion with flexible strategy
        used_backbone_levels = len(laterals)
        # We have (used_backbone_levels - 1) fusion steps: from top to bottom
        for i in range(used_backbone_levels - 1, 0, -1):
            fusion_idx = used_backbone_levels - 1 - i  # maps i=3→0, i=2→1, i=1→2
            mode = self.fusion_modes[fusion_idx]

            if mode == 'add':
                # Original FPN: interpolate + add
                if 'scale_factor' in self.upsample_cfg:
                    upsampled = F.interpolate(laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
                laterals[i - 1] = laterals[i - 1] + upsampled
            elif mode == 'faa':
                fused_low = self.fam_modules[fusion_idx](laterals[i], laterals[i - 1])
                laterals[i - 1] = fused_low
            else:
                raise ValueError(f"Unknown fusion mode: {mode}")

        # Step 3: Build outputs (same as FPN)
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # Step 4: Add extra levels (same as FPN)
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                # Use max pooling
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                # Use extra convs
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)

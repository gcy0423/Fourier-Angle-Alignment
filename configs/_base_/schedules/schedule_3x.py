# evaluation
evaluation = dict(
    interval=1,
    metric='mAP',
    save_best='mAP'  # 自动保存 best_mAP_epoch_xx.pth
)
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(
    interval=1,          # 每隔 1 个 epoch 保存一次权重
    max_keep_ckpts=7     # 只保留最新的 7 个 checkpoint
)

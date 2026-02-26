# Fourier Angle Alignment
**[CVPR 2026]** Official implementation of *Fourier Angle Alignment for Oriented Object Detection in Remote Sensing*

## 📰 Abstract
In remote sensing rotated object detection, mainstream methods suffer from two bottlenecks, directional incoherence at detector neck and task conflict at detecting head. Ulitising fourier rotation equivariance, we introduce **Fourier Angle Alignment**, which analyses angle information through frequency spectrum and aligns the main direction to a certain orientation. Then we propose two plug and play modules : **FAAFusion** and **FAA Head**. FAAFusion works at the detector neck, aligning the main direction of higher-level features to the lower-level features and then fusing them. FAA Head serves as a new detection head, which pre-aligns RoI features to a canonical angle and adds them to the original features before classification and regression. Experiments on DOTA-v1.0, DOTA-v1.5 and HRSC2016 show that our method can greatly improve previous work. Particularly, our method achieves new state-of-the-art results of 78.72% mAP on DOTA-v1.0 and 72.28% mAP on DOTA-v1.5 datasets with single scale training and testing, validating the efficacy of our approach in remote sensing object detection.



## 📖 Citation

If you find this work useful for your research, please consider citing our paper:

```
@InProceedings{gcy2026faa,
  title={Fourier Angle Alignment for Oriented Object Detection in Remote Sensing},
  author={Gu, Changyu and Chen, Linwei and Gu, Lin and Fu, Ying},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2026}
}
```

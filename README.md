# Group-CAM
By Zhang, Qinglong and Rao, Lu and Yang, Yubin

[State Key Laboratory for Novel Software Technology at Nanjing University]

This repo is the official implementation of ["Group-CAM: Group Score-Weighted Visual Explanations for Deep Convolutional Networks"](https://arxiv.org/pdf/2103.13859v4.pdf).

## Approach
<div align="center">
  <img src="https://github.com/wofmanaf/Group-CAM/blob/master/figure/fig_1.png">
</div>
<p align="center">
  Figure 1: Pipeline of Group-CAM.
</p>

## Target layer
ResNet: 'layer4.2',  Vgg19: 'features.36'

## Demo
To visualize a heatmap, run:
```bash
python demo.py --arch vgg19 --target_layer features.36 --input images/ILSVRC2012_val_00043392.JPEG --output base.png
```

## Citing Group-CAM

```
@article{zhql2021gc,
  title={Group-CAM: Group Score-Weighted Visual Explanations for Deep Convolutional Networks},
  author={Zhang, Qinglong and Rao, Lu and Yang, Yubin},
  journal={arXiv preprint arXiv:2103.13859v4},
  year={2021}
}
```
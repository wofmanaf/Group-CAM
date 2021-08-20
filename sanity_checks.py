import torch
import torch.nn as nn
import torchvision.models as models
from utils import *
from cam import GroupCAM

# check_type = 'cascade_random'
check_type = 'ind_random'

import warnings
warnings.filterwarnings("ignore")


def cascade_randomization(arch, num_layers_from_last=None):
    model = models.__dict__[arch](pretrained=True)

    num = -1 * num_layers_from_last
    conv2d_keys = []

    for key in model.features._modules.keys():
        if isinstance(model.features._modules[key], nn.Conv2d):
            conv2d_keys.append(key)

    for key in conv2d_keys[num:]:
        layer = model.features._modules[key]
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        model.features._modules[key] = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    return model


def independent_randomization(arch, layer=None):
    model = models.__dict__[arch](pretrained=True)
    conv2d_keys = []

    for key in model.features._modules.keys():
        if isinstance(model.features._modules[key], nn.Conv2d):
            conv2d_keys.append(key)

    conv2d_key = conv2d_keys[-1 * layer]
    layer = model.features._modules[conv2d_key]

    in_channels = layer.in_channels
    out_channels = layer.out_channels
    kernel_size = layer.kernel_size
    stride = layer.stride
    padding = layer.padding

    model.features._modules[conv2d_key] = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    return model


raw_img = cv2.imread('Images/' + 'ILSVRC2012_val_00000073.JPEG', 1)
# raw_img = cv2.imread('Images/' + 'ILSVRC2012_val_00043392.JPEG', 1)
raw_img = cv2.resize(raw_img, (224, 224), interpolation=cv2.INTER_LINEAR)

raw_img = np.float32(raw_img) / 255
image, norm_image = preprocess_img(raw_img)
vgg19 = models.vgg19(pretrained=True).eval()
vgg19 = vgg19.cuda()
norm_image = norm_image.cuda()

logit = vgg19(norm_image)
cls_idx = logit.max(1)[-1].item()

heatmap = GroupCAM(vgg19, target_layer='features.35', groups=32)(norm_image, class_idx=cls_idx).cpu().data

cam = show_cam(image, heatmap, 'base.png')
for i in range(1, 17):
    if check_type == 'cascade_random':
        model = cascade_randomization('vgg19', num_layers_from_last=i)
    else:
        model = independent_randomization('vgg19', layer=i)
    model = model.cuda()
    gc = GroupCAM(model, target_layer='features.35', groups=32)
    try:
        heatmap = gc(norm_image, class_idx=cls_idx).cpu().data
        torch.cuda.empty_cache()
        model = model.cpu()
        heatmap = show_heatmap(heatmap, title="Checks/" + check_type + str(i) + '.png')
    except Exception as e:
        continue
    del model

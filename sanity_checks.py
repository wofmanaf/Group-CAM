import torch
import torch.nn as nn
import torchvision.models as models
from utils import *
from cam import GroupCAM

check_type = 'cascade_random'


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


def preprocess_img(cv_img):
    """Turn a opencv image into tensor and normalize it"""
    # revert the channels from BGR to RGB
    img = cv_img.copy()[:, :, ::-1]
    # convert tor tensor
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1))))
    # Normalize
    transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    norm_img = transform_norm(img).unsqueeze(0)

    return img, norm_img


raw_img = cv2.imread('images/' + 'ILSVRC2012_val_00043392.JPEG', 1)
raw_img = cv2.resize(raw_img, (224, 224), interpolation=cv2.INTER_LINEAR)

raw_img = np.float32(raw_img) / 255
image, norm_image = preprocess_img(raw_img)
vgg19 = models.vgg19(pretrained=True).eval()
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
    heatmap = gc(norm_image, class_idx=cls_idx).cpu().data
    torch.cuda.empty_cache()
    model = model.cpu()
    heatmap = show_heatmap(heatmap, title=check_type + str(i) + '.png')
    del model

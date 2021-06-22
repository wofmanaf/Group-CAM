import argparse
import cv2
import numpy as np
import torchvision.models as models
from utils import preprocess_img, show_cam
from cam import GroupCAM

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser("group-cam demo")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19')
    parser.add_argument('--cls_idx', default=None, type=str)
    parser.add_argument('--target_layer', default='features.36', type=str)
    parser.add_argument('--input', default='', type=str, help='Input images')
    parser.add_argument('--output', default='', type=str, help='output path')
    return parser.parse_args()


def main():
    args = parse_args()
    raw_img = cv2.imread(args.input, 1)
    raw_img = cv2.resize(raw_img, (224, 224), interpolation=cv2.INTER_LINEAR)

    raw_img = np.float32(raw_img) / 255
    image, norm_image = preprocess_img(raw_img)
    model = models.__dict__[args.arch](pretrained=True).eval()

    heatmap = GroupCAM(model, target_layer=args.target_layer, groups=32)\
        (norm_image, class_idx=args.cls_idx).cpu().data
    cam = show_cam(image, heatmap, args.output)


if __name__ == '__main__':
    main()
import argparse
import os

import cv2
import numpy as np
import torchvision.models as models
from utils import preprocess_img, show_cam, write_video
from cam import *
import warnings
warnings.filterwarnings("ignore")

import torch
from kornia.filters.gaussian import gaussian_blur2d
from ins_del_gc import CausalMetric, auc

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def check_path_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def parse_args():
    parser = argparse.ArgumentParser("group-cam demo")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19')
    parser.add_argument('--cls_idx', default=None, type=int)
    parser.add_argument('--target_layer', default='features.35', type=str)
    parser.add_argument('--input', default='', type=str, help='Input images')
    parser.add_argument('--output', default='', type=str, help='output path')
    parser.add_argument('--ins_del', action='store_true', default=False,
                        help='whether to record the insertion and deletion results.')
    return parser.parse_args()


def main():
    args = parse_args()
    raw_img = cv2.imread(args.input, 1)
    raw_img = cv2.resize(raw_img, (224, 224), interpolation=cv2.INTER_LINEAR)

    raw_img = np.float32(raw_img) / 255
    image, norm_image = preprocess_img(raw_img)
    model = models.__dict__[args.arch](pretrained=True).eval()
    model = model.cuda()

    heatmap = GroupCAM(model, target_layer=args.target_layer)\
        (norm_image.cuda(), class_idx=args.cls_idx).cpu().data
    cam = show_cam(image, heatmap, args.output)

    if args.ins_del:
        blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))
        insertion = CausalMetric(model, 'ins', 224 * 8, substrate_fn=blur)
        deletion = CausalMetric(model, 'del', 224 * 8, substrate_fn=torch.zeros_like)
        out_video_path = './VIDEO'
        check_path_exist(out_video_path)

        ins_path = os.path.join(os.path.join(out_video_path, "ins"))
        del_path = os.path.join(os.path.join(out_video_path, "del"))
        check_path_exist(ins_path)
        check_path_exist(del_path)

        norm_image = norm_image.cpu()
        heatmap = heatmap.cpu().numpy()

        ins_score = insertion.evaluate(norm_image, mask=heatmap, cls_idx=None, save_to=ins_path)
        del_score = deletion.evaluate(norm_image, mask=heatmap, cls_idx=None, save_to=del_path)
        print("\nDeletion - {:.5f}\nInsertion - {:.5f}".format(auc(del_score), auc(ins_score)))

        # generate video
        video_ins = os.path.join(ins_path, args.input.split('/')[-1].split('.')[0] + '.avi')
        video_del = os.path.join(del_path, args.input.split('/')[-1].split('.')[0] + '.avi')
        cmd_str_ins = 'ffmpeg -f image2 -i {}/%06d.jpg -b 5000k -r 30 -c:v mpeg4 {} -y'.format(ins_path, video_ins)
        cmd_str_del = 'ffmpeg -f image2 -i {}/%06d.jpg -b 5000k -r 30 -c:v mpeg4 {} -y'.format(del_path, video_del)
        os.system(cmd_str_ins)
        os.system(cmd_str_del)


if __name__ == '__main__':
    main()
"""
run: python plt_curve.py --input Images/ILSVRC2012_val_00000073.JPEG --ins_del
"""
import argparse
import os

import cv2
import numpy as np
import torchvision.models as models
from utils import preprocess_img, show_cam, write_video
from cam import *
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import torch
from kornia.filters.gaussian import gaussian_blur2d
from ins_del_gc import CausalMetric, auc

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser("group-cam demo")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19')
    parser.add_argument('--cls_idx', default=None, type=int)
    parser.add_argument('--target_layer', default='features.35', type=str)
    parser.add_argument('--input', default='', type=str, help='Input Images')
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

    rise = RISE(model, input_size=(224, 224), batch_size=40)
    rise.generate_masks()
    gd = GradCAM(model, target_layer=args.target_layer)
    gc = GroupCAM(model, target_layer=args.target_layer)

    rise_heatmap = rise(norm_image.cuda(), class_idx=args.cls_idx).cpu().data
    gd_heatmap = gd(norm_image.cuda(), class_idx=args.cls_idx).cpu().data
    gc_heatmap = gc(norm_image.cuda(), class_idx=args.cls_idx).cpu().data

    if args.output is not None:
        rise_cam = show_cam(image, rise_heatmap, "rise_base.png")
        gd_cam = show_cam(image, gd_heatmap, "gd_base.png")
        gc_cam = show_cam(image, gc_heatmap, "gc_base.png")

    if args.ins_del:
        blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))
        insertion = CausalMetric(model, 'ins', 224 * 2, substrate_fn=blur)
        deletion = CausalMetric(model, 'del', 224 * 2, substrate_fn=torch.zeros_like)

        norm_image = norm_image.cpu()
        gd_heatmap = gd_heatmap.cpu().numpy()
        gc_heatmap = gc_heatmap.cpu().numpy()
        rise_heatmap = rise_heatmap.cpu().numpy()

        gc_ins_score = insertion.evaluate(norm_image, mask=gc_heatmap, cls_idx=None)
        gd_ins_score = insertion.evaluate(norm_image, mask=gd_heatmap, cls_idx=None)
        rise_ins_score = insertion.evaluate(norm_image, mask=rise_heatmap, cls_idx=None)

        gc_del_score = deletion.evaluate(norm_image, mask=gc_heatmap, cls_idx=None)
        gd_del_score = deletion.evaluate(norm_image, mask=gd_heatmap, cls_idx=None)
        rise_del_score = deletion.evaluate(norm_image, mask=rise_heatmap, cls_idx=None)

        legend = ["RISE", "Grad-CAM", "Group-CAM"]
        ins_scores = [auc(rise_ins_score), auc(gd_ins_score), auc(gc_ins_score)]
        del_scores = [auc(rise_del_score), auc(gd_del_score), auc(gc_del_score)]
        ins_scores = [round(i*100, 2) for i in ins_scores]
        del_scores = [round(i*100, 2) for i in del_scores]
        ins_legend = [i + ": " + str(j) for i, j in zip(legend, ins_scores)]
        del_legend = [i + ": " + str(j) for i, j in zip(legend, del_scores)]

        n_steps = len(gd_ins_score)

        x = np.arange(n_steps) / n_steps
        plt.figure(figsize=(12, 5))

        plt.xlim(-0.1, 1.1)
        plt.ylim(0, 1.05)

        plt.subplot(121)
        plt.plot(x, rise_ins_score)
        plt.plot(x, gd_ins_score)
        plt.plot(x, gc_ins_score)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(ins_legend, loc='best', fontsize=15)
        plt.title("Insertion Curve", fontsize=15)

        plt.subplot(122)
        plt.plot(x, rise_del_score)
        plt.plot(x, gd_del_score)
        plt.plot(x, gc_del_score)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(del_legend, loc='best', fontsize=15)
        plt.title("Deletion Curve", fontsize=15)
        plt.show()


if __name__ == '__main__':
    main()
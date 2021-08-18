# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from utils import convert_to_gray


class GuidedBackProp(object):
    def __init__(self, model):
        super(GuidedBackProp, self).__init__()
        self.model = model.eval()

        for module in self.model.named_modules():
            module[1].register_backward_hook(self.bp_relu)

    def bp_relu(self, module, grad_in, grad_out):
        # Cut off negative gradients
        if isinstance(module, nn.ReLU):
            return (F.relu(grad_in[0]), )

    def forward(self, x, class_idx=None, retain_graph=False):
        x.requires_grad_()
        logit = self.model(x)

        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        saliency_map = x.grad  # [1, 3, H, W]

        saliency_map = convert_to_gray(saliency_map.cpu().data) # [1, 1, H, W]
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

        return saliency_map

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)

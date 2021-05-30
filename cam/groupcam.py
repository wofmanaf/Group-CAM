import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters.gaussian import gaussian_blur2d

blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))


class GroupCAM(object):
    def __init__(self, model, target_layer="layer4.2", groups=32):
        super(GroupCAM, self).__init__()
        self.model = model.eval().cuda()
        self.groups = groups
        self.gradients = dict()
        self.activations = dict()

        for module in self.model.named_modules():
            if module[0] == target_layer:
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activations['value'] = output

    def forward(self, x, class_idx=None, retain_graph=False):
        input = x.clone()
        input = input.cuda()
        b, c, h, w = input.size()
        logit = self.model(input)

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        predicted_class = predicted_class.cuda()
        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'].data
        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        activations = weights * activations

        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        masks = activations.chunk(self.groups, 1)
        with torch.no_grad():
            base_line = F.softmax(self.model(blur(x)), dim=-1)[0][predicted_class]
            for saliency_map in masks:
                saliency_map = saliency_map.sum(1, keepdims=True)
                saliency_map = F.relu(saliency_map)
                threshold = np.percentile(saliency_map.cpu().numpy(), 70)
                saliency_map = torch.where(
                    saliency_map > threshold, saliency_map, torch.full_like(saliency_map, 0))
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                if saliency_map.max() == saliency_map.min():
                    continue

                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # how much increase if keeping the highlighted region
                # predication on masked input
                blur_input = input * norm_saliency_map + blur(input) * (1 - norm_saliency_map)
                output = self.model(blur_input)
                output = F.softmax(output, dim=-1)
                score = output[0][predicted_class] - base_line

                # score_saliency_map += score * saliency_map
                score_saliency_map += score * norm_saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min) / (
                score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
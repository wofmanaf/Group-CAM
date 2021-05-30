import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters.gaussian import gaussian_blur2d

blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))

class GroupCAM(object):
    def __init__(self, model, target_layer="module.layer4.2", groups=16):
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

        masks = activations.chunk(self.groups, 1)
        # parallel implement
        masks = torch.cat(masks, dim=0)
        saliency_map = masks.sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        threshold = np.percentile(saliency_map.cpu().numpy(), 70)
        saliency_map = torch.where(
            saliency_map > threshold, saliency_map, torch.full_like(saliency_map, 0))

        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map = saliency_map.reshape(self.groups, -1)
        inter_min, inter_max = saliency_map.min(dim=-1, keepdim=True)[0], saliency_map.max(dim=-1, keepdim=True)[0]
        saliency_map = (saliency_map-inter_min) / (inter_max - inter_min)
        saliency_map = saliency_map.reshape(self.groups, 1, h, w)

        with torch.no_grad():
            blur_input = input * saliency_map + blur(input) * (1 - saliency_map)
            output = self.model(blur_input)
        output = F.softmax(output, dim=-1)
        score = output[:, predicted_class].unsqueeze(-1).unsqueeze(-1)
        score_saliency_map = torch.sum(saliency_map * score, dim=0, keepdim=True)

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min) / (
                score_saliency_map_max - score_saliency_map_min).data
        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

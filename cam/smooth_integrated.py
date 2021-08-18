import torch
from utils import convert_to_gray


class SmoothIntGrad(object):
    def __init__(self, model, stdev_spread=0.15, n_steps=20, magnitude=True):
        super(SmoothIntGrad, self).__init__()
        self.stdev_spread = stdev_spread
        self.n_steps = n_steps
        self.magnitude = magnitude

        self.model = model.eval()

    def forward(self, x, x_baseline=None, class_idx=None, retain_graph=False):
        if x_baseline is None:
            x_baseline = torch.zeros_like(x)
        else:
            x_baseline = x_baseline.cuda()
        assert x_baseline.size() == x.size()

        saliency_map = torch.zeros_like(x) # [1, 3, H, W]

        x_diff = x - x_baseline
        stdev = self.stdev_spread / (x_diff.max() - x_diff.min())
        for alpha in torch.linspace(0., 1., self.n_steps):
            x_step = x_baseline + alpha * x_diff
            noise = torch.normal(mean=torch.zeros_like(x_step), std=stdev)
            x_step_plus_noise = x_step + noise
            logit = self.model(x_step_plus_noise)

            if class_idx is None:
                score = logit[:, logit.max(1)[-1]].squeeze()
            else:
                score = logit[:, class_idx].squeeze()

            self.model.zero_grad()
            score.backward(retain_graph=retain_graph)
            saliency_map += x_step_plus_noise.grad

        saliency_map = saliency_map / self.n_steps
        saliency_map = convert_to_gray(saliency_map) # [1, 1, H, W]

        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min).data

        return saliency_map

    def __call__(self, x, x_baseline=None, class_idx=None, retain_graph=False):
        return self.forward(x, x_baseline, class_idx, retain_graph)

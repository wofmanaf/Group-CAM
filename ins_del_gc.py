# CUDA_VISIBLE_DEVICES=0
import json

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from kornia.filters.gaussian import gaussian_blur2d
from tqdm import tqdm

from cam import *

__all__ = ['CausalMetric', 'auc']

import warnings
warnings.filterwarnings("ignore")

HW = 224 * 224
img_label = json.load(open('./utils/resources/imagenet_class_index.json', 'r'))


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)


def explain_all(data_loader, explainer):
    """Get saliency maps for all Images in val loader
    Args:
        data_loader: torch data loarder
        explainer: gradcam, etc.
    Return:
        Images: list, length: len(data_loader), element: torch tensor with shape of (1, 3, H, W)
        explanations: np.ndarrays, with shape of (len(data_loader, H, W)
    """
    global vgg
    explanations = []
    images = []
    for i, (img, _) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Explaining Images')):
        try:
            # cls_idx = vgg(img.cuda()).max(1)[-1].item()
            saliency_maps = explainer(img.cuda(), class_idx=None).data
            explanations.append(saliency_maps.cpu().numpy())
            images.append(img)
        except Exception as e:
            continue

    explanations = np.array(explanations)
    return images, explanations


class CausalMetric(object):
    def __init__(self, model, mode, step, substrate_fn):
        """Create deletion/insertion metric instance.
        Args:
            model(nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model.eval().cuda()
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def evaluate(self, img, mask, cls_idx=None, verbose=0, save_to=None):
        """Run metric on one image-saliency pair.
        Args:
            img (Tensor): normalized image tensor.
            mask (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step.
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        if cls_idx is None:
            cls_idx = self.model(img.cuda()).max(1)[-1].item()

        n_steps = (HW + self.step - 1) // self.step
        if self.mode == 'del':
            title = 'Deletion Curve'
            ylabel = 'Pixels deleted'
            start = img.clone()
            finish = self.substrate_fn(img)
        elif self.mode == 'ins':
            title = 'Insertion Curve'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img)
            finish = img.clone()

        scores = np.empty(n_steps + 1, dtype='float32')
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(mask.reshape(-1, HW), axis=1), axis=-1)

        for i in range(n_steps + 1):
            logit = self.model(start.cuda())
            score = F.softmax(logit, dim=-1)[:, cls_idx].squeeze()
            scores[i] = score
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.axis('off')
                tensor_imshow(start[0])

                plt.subplot(122)
                plt.plot(np.arange(i + 1) / n_steps, scores[:i + 1])
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / n_steps, 0, scores[:i + 1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel(img_label[str(cls_idx)][-1])
                if save_to:
                    plt.savefig(save_to + '/{:06d}.jpg'.format(i))
                    plt.close()
                else:
                    plt.show()

            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start.cpu().numpy().reshape(1, 3, HW)[0, :, coords] = \
                    finish.cpu().numpy().reshape(1, 3, HW)[0, :, coords]
        return scores


def main():
    # hyper-parameters
    val_dir = '/path/to/imagenet/val/'
    batch_size = 1
    num_workers = 4
    batch = 0

    model_type = "vgg"
    saliency_type = 'group_cam'

    # sample_range = range(5 * batch, 5 * (batch + 1))
    sample_range = range(1 * batch, 1 * (batch + 1))

    vgg = models.vgg19(pretrained=True).eval()
    vgg = vgg.cuda()
    cam = GroupCAM(vgg, target_layer='features.35', groups=32)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=RangeSampler(sample_range)
    )

    images, exp = explain_all(val_loader, explainer=cam)
    # Function that blurs input image
    blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))

    # Evaluate a batch of explanations
    insertion = CausalMetric(vgg, 'ins', 224 * 2, substrate_fn=blur)
    deletion = CausalMetric(vgg, 'del', 224 * 2, substrate_fn=torch.zeros_like)

    scores = {'del': [], 'ins': []}
    del_tmps = []
    ins_tmps = []
    # Load saved batch of explanations
    for i in tqdm(range(len(images)), total=len(images), desc='Evaluating Saliency'):
        # Evaluate deletion
        del_score = deletion.evaluate(img=images[i], mask=exp[i], cls_idx=None, verbose=0)
        ins_score = insertion.evaluate(img=images[i], mask=exp[i], cls_idx=None, verbose=0)
        del_tmps.append(del_score)
        ins_tmps.append(ins_score)
        scores['del'].append(auc(del_score))
        scores['ins'].append(auc(ins_score))

    print('----------------------------------------------------------------')
    print('Final:\nDeletion - {:.5f}\nInsertion - {:.5f}'.format(np.mean(scores['del']), np.mean(scores['ins'])))


if __name__ == '__main__':
    main()
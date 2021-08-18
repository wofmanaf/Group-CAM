import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm


class RISE(nn.Module):
    def __init__(self, model, input_size, batch_size=100, N=8000, s=7, p1=0.1):
        super(RISE, self).__init__()
        assert N % batch_size == 0
        self.model = model.eval()
        self.input_size = input_size
        self.batch_size = batch_size
        self.N = N
        self.s = s
        self.p1 = p1

    def generate_masks(self, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / self.s)
        up_size = (self.s + 1) * cell_size

        grid = np.random.rand(self.N, self.s, self.s) < self.p1
        grid = grid.astype('float32')

        self.masks = np.empty((self.N, *self.input_size))

        for i in tqdm(range(self.N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float()
        self.N = self.masks.shape[0]

    def forward(self, x, class_idx=None):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        saliency = torch.zeros(1, 1, H, W).cuda()
        if class_idx is None:
            logit = self.model(x)
            class_idx = logit.max(1)[-1]
        else:
            class_idx = torch.LongTensor([class_idx])

        for i in range(0, self.N, self.batch_size):
            mask = self.masks[i: min(i+self.batch_size, N)]
            mask = mask.cuda()
            with torch.no_grad():
                logit = self.model(mask * x)
            score = logit[:, class_idx].unsqueeze(-1).unsqueeze(-1)
            saliency += (score * mask).sum(dim=0, keepdims=True)
        return saliency / N / self.p1
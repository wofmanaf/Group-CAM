import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import numpy as np

from misc import preprocess_img, tv_norm, get_idx_score, Heatmap_Revising, showimage, write_video, save_heatmap

import cv2


def Get_blurred_img(image_path, input_size=224, gaussian_params=(51, 50), median_params=11, blur_type='Gaussian'):
    raw_img = cv2.imread(image_path, 1)
    raw_img = cv2.resize(raw_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    raw_img = np.float32(raw_img) / 255

    if blur_type == 'Gaussian':
        k, sigmaX = gaussian_params
        blurred_img = cv2.GaussianBlur(raw_img, ksize=(k, k), sigmaX=sigmaX)
    elif blur_type == 'Median':
        blurred_img = cv2.medianBlur(raw_img, ksize=median_params)
    else:
        blurred_img = raw_img + 0.5

    img, image_norm = preprocess_img(raw_img)
    blurred_img, blurred_img_norm = preprocess_img(blurred_img)

    return img, image_norm, blurred_img, blurred_img_norm


def Integrated_Mask(model, image, blurred_image, init_mask, class_idx=None, max_iters=10, n_steps=20,
                    l1_coeff=0.01 * 100, tv_coeff=0.2 * 100, pixel_nums=200, use_cuda=True):

    # Initialize the mask
    if init_mask is None:
        init_mask = torch.ones(size=(1, 1, 28, 28), dtype=torch.float32)

    # optimizer = optim.SGD([init_mask], lr=0.1, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam([init_mask], lr=0.1)
    model = model.eval()
    if use_cuda:
        model = model.cuda()
        image, blurred_image = image.cuda(), blurred_image.cuda()
        init_mask = init_mask.cuda()

    init_mask.requires_grad_()
    upsample = nn.Upsample(size=(image.size(-2), image.size(-1)), mode='bilinear', align_corners=False)

    # get classification score, and top class_idx
    score, class_idx = get_idx_score(image, model, class_idx=class_idx)
    # print("class idx: {}, ""score: {:.3f}".format(class_idx, score))

    curve1 = []
    curve2 = []
    curvetop = []

    # Integrated gradient descent
    stdev_spread = 0.25

    alpha = 0.0001
    beta = 0.2

    for iters in range(max_iters):
        up_mask = upsample(init_mask)
        up_mask = up_mask.expand(1, 3, up_mask.size(-2), up_mask.size(-1))

        # the l1 term and the total variation term
        loss_1 = l1_coeff * torch.abs(torch.sub(1, init_mask)).mean() + tv_coeff * tv_norm(init_mask, tv_beta=2)
        loss_inte = loss_1.clone()

        # Compute the perturbated_image
        perturbated_image = torch.mul(image, up_mask) + torch.mul(blurred_image, 1 - up_mask)
        stdev = stdev_spread / (perturbated_image.max() - perturbated_image.min()).item()

        for inte_iter in range(n_steps):
            inte_mask = 0.0 + ((inte_iter + 1.0) / n_steps) * up_mask
            # test_blur_img = 0.0 + ((inte_iter + 1.0) / n_steps) * blurred_image
            inte_image = torch.mul(image, inte_mask) + torch.mul(blurred_image, 1 - inte_mask)
            # inte_image = torch.mul(image, inte_mask) + torch.mul(test_blur_img, 1 - inte_mask)
            # add noise using smoothgrad
            noise = torch.normal(mean=inte_image * 0.0, std=stdev).float().detach()
            step_plus_noise = torch.add(inte_image, noise)

            inte_output = F.softmax(model(step_plus_noise), dim=-1)
            inte_output = inte_output[:, class_idx]
            loss_inte = loss_inte + inte_output / n_steps

        # compute the integrated gradients for the given target, and compute the gradients for the l1 term and tv_norm
        model.zero_grad()
        optimizer.zero_grad()
        loss_inte.backward()
        grads = init_mask.grad.data.clone()

        loss_2 = F.softmax(model(perturbated_image), dim=-1)[0, class_idx]
        losses = loss_1 + loss_2

        # print("iter: {}, loss: {:.3f}".format(iters, losses.data.cpu().numpy()))

        # collect curve
        if iters == 0:
            curve1.append(loss_1.data.cpu().numpy())
            curve2.append(loss_2.data.cpu().numpy())
            curvetop.append(loss_2.data.cpu().numpy())

        # LINE SEARCH with revised Armijo condition
        step = 200.0
        new_mask = init_mask.data.clone()
        # new_mask -= step * grads
        new_mask -= step * grads
        new_mask.data.clamp_(0, 1)  # clamp the value of mask in [0,1]

        new_upmask = upsample(new_mask)  # Here the direction in the grads
        new_image = torch.mul(image, new_upmask) + torch.mul(blurred_image, 1 - new_upmask)
        ls_1 = F.softmax(model(new_image), dim=-1)[0, class_idx]
        ls_2 = l1_coeff * torch.mean(torch.abs(1 - new_mask)) + tv_coeff * tv_norm(new_mask, tv_beta=2)
        ls = ls_2 + ls_1
        ls = ls.data.cpu().numpy()

        new_condition = (grads ** 2).sum()
        new_condition = (alpha * step * new_condition).cpu().numpy()

        while ls > losses.data.cpu().numpy() - new_condition:
            step *= beta

            new_mask = init_mask.data.clone()
            new_mask -= step * grads
            new_mask.data.clamp_(0, 1)

            new_upmask = upsample(new_mask)  # Here the direction in the grads
            new_image = torch.mul(image, new_upmask) + torch.mul(blurred_image, 1 - new_upmask)
            ls_1 = F.softmax(model(new_image), dim=-1)[0, class_idx]
            ls_2 = l1_coeff * torch.mean(torch.abs(1 - new_mask)) + tv_coeff * tv_norm(new_mask, tv_beta=2)
            ls = ls_2 + ls_1
            ls = ls.data.cpu().numpy()

            new_condition = (grads ** 2).sum()
            new_condition = (alpha * step * new_condition).cpu().numpy()

            if step < 0.00001:
                break

        init_mask.data -= step * grads
        init_mask.data.clamp_(0, 1)

        curve1.append(loss_1.data.cpu().numpy())
        curve2.append(loss_2.data.cpu().numpy())

        # use the mask to perturbated the input image and calculate the loss
        del_mask = init_mask.data.cpu().squeeze().numpy()
        del_mask, img_ratio = Heatmap_Revising(del_mask, thre_num=pixel_nums, type='deleting')

        del_mask = torch.from_numpy(del_mask).unsqueeze(0).unsqueeze(0)
        if use_cuda:
            del_mask = del_mask.cuda()

        del_upmask = upsample(del_mask)
        del_image = torch.mul(image, del_upmask) + torch.mul(blurred_image, 1 - del_upmask)
        del_loss_1 = F.softmax(model(del_image), dim=-1)[0, class_idx]
        # del_loss_2 = l1_coeff * torch.mean(torch.abs(1 - del_mask)) + tv_coeff * tv_norm(del_mask, tv_beta)

        curvetop.append(del_loss_1.data.cpu().numpy())

        if max_iters > 3:
            if iters == int(max_iters / 2):
                if np.abs(curve2[0] - curve2[iters]) <= 0.001:
                    # print('Adjust Parameter l1 at iteration:', int(max_iters / 2))
                    l1_coeff = l1_coeff / 10
            elif iters == int(max_iters / 1.25):
                if np.abs(curve2[0] - curve2[iters]) <= 0.01:
                    # print('Adjust Parameters l1 again at iteration:', int(max_iters / 1.25))
                    l1_coeff = l1_coeff / 5

    init_mask = init_mask.data.cpu()
    up_mask = upsample(init_mask)

    return init_mask, up_mask, curvetop, curve1, curve2, class_idx


def Deletion_Insertion(mask, model, output_path, image, blurred_image, class_idx, pixel_nums=200, save_figure=True):
    # Here we use cpu to inference since we need to convert tensor to numpy frequently
    model = model.cpu()
    out_max, class_idx = get_idx_score(image, model, class_idx=class_idx)
    print("class idx: ", class_idx, ", score: ", out_max)

    classes = json.load(open("imagenet_class_index.json", 'r'))
    labels = classes[str(class_idx)][-1]

    upsample = nn.Upsample(size=(image.size(-2), image.size(-1)), mode='bilinear', align_corners=False)

    size_M = mask.size(-2) * mask.size(-1)
    n_steps = 1 if size_M < pixel_nums else int(size_M / pixel_nums)
    xtick = np.arange(0, int(size_M / 3.5), n_steps)
    x_num = xtick.shape[0]
    xtick = x_num + 10

    del_curve = []
    insert_curve = []
    del_image = torch.zeros_like(image)
    insert_image = torch.zeros_like(image)

    iter = 0
    for pix_num in range(0, int(size_M / 3.5), n_steps):
        iter += 1
        del_mask = mask.clone()
        del_mask, del_ratio = Heatmap_Revising(del_mask.squeeze().numpy(), pix_num, type='deleting')
        del_mask = torch.from_numpy(del_mask).reshape_as(mask)

        del_upmask = upsample(del_mask)
        del_image = torch.mul(image, del_upmask) + torch.mul(blurred_image, 1 - del_upmask)
        del_score, _ = get_idx_score(del_image, model, class_idx)
        print("blocks: {}, del score: {:.3f}".format(iter, del_score))
        # del_curve.append(del_score / out_max)
        del_curve.append(del_score)

        insert_mask = mask.clone()
        insert_mask, insert_ratio = Heatmap_Revising(insert_mask.squeeze().numpy(), pix_num, type='inserting')
        insert_mask = torch.from_numpy(insert_mask).reshape_as(mask)

        insert_upmask = upsample(insert_mask)
        insert_image = torch.mul(image, insert_upmask) + torch.mul(blurred_image, 1 - insert_upmask)
        insert_score, _ = get_idx_score(insert_image, model, class_idx)
        # insert_curve.append(insert_score / out_max)
        insert_curve.append(insert_score)

        if save_figure:
            # normalize and tranpose to numpy
            del_image = (del_image - del_image.min()) / (del_image.max() - del_image.min())
            del_image = np.transpose(del_image.squeeze(0).numpy(), (1, 2, 0))

            insert_image = (insert_image - insert_image.min()) / (insert_image.max() - insert_image.min())
            insert_image = np.transpose(insert_image.squeeze(0).numpy(), (1, 2, 0))

            showimage(del_image, insert_image, del_curve, insert_curve, output_path, xtick, labels)

    return del_image, insert_image, del_curve, insert_curve, out_max, x_num


if __name__ == '__main__':
    from torchvision.models import densenet201
    from utils import visualize_cam, save_images

    input_path = './images/'
    output_path = './Results/VIDEO/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    files = os.listdir(input_path)
    print(files)

    model = densenet201(pretrained=True)

    for img_name in files:
        input_img = input_path + img_name
        raw_image, image_norm, blurred_img, blurred_img_norm = Get_blurred_img(input_img)

        mask, up_mask, curvetop, curve1, curve2, class_idx = Integrated_Mask(
            model, image_norm, blurred_img_norm, init_mask=None, class_idx=None, max_iters=48, n_steps=20,
            l1_coeff=0.01 * 100, tv_coeff=0.1 * 100, pixel_nums=40, use_cuda=True)

        out_video_path = output_path + input_img.split('/')[-1].split('.')[0] + '/'
        if not os.path.exists(out_video_path):
            os.makedirs(out_video_path)
        output_save_path = output_path + input_img.split('/')[-1].split('.')[0] + '_IGOS_'

        save_heatmap(output_save_path, up_mask, raw_image, blurred_img)

        output_file = out_video_path + input_img.split('/')[-1].split('.')[0] + '_IGOS_'
        del_image, insert_image, del_curve, insert_curve, out_max, x_num = Deletion_Insertion(
            mask, model, output_file, image_norm, blurred_img_norm, class_idx, pixel_nums=200,
            save_figure=True)

        video_name = out_video_path + 'AllVideo_fps10' + input_img.split('/')[-1].split('.')[0] + '.avi'
        write_video(output_file, video_name, x_num, fps=3)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import random
import math
import scipy.io as scio


def handle_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("mkdir:", dir)


def kernel2png(kernel):
    kernel = cv2.resize(kernel, dsize=(0, 0), fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
    kernel = np.clip(kernel, 0, np.max(kernel))
    kernel = kernel / np.sum(kernel)
    mi = np.min(kernel)
    ma = np.max(kernel)
    kernel = (kernel - mi) / (ma - mi)
    kernel = np.round(np.clip(kernel * 255., 0, 255))
    kernel_png = np.stack([kernel, kernel, kernel], axis=2).astype('uint8')
    return kernel_png


def matlab_style_gauss2D(shape=(5, 5), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_blur_kernel(trian=True):
    if trian:
        gaussian_sigma = random.choice(
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    else:
        gaussian_sigma = 2.0
    gaussian_blur_kernel_size = int(math.ceil(gaussian_sigma * 3) * 2 + 1)
    kernel = matlab_style_gauss2D((gaussian_blur_kernel_size, gaussian_blur_kernel_size), gaussian_sigma)
    return kernel


def get_lr_blurdown(img_gt, kernel):
    img_gt = np.array(img_gt).astype('float32')
    gt_tensor = torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0).float()

    kernel_size = kernel.shape[0]
    psize = kernel_size // 2
    gt_tensor = F.pad(gt_tensor, (psize, psize, psize, psize), mode='replicate')

    gaussian_blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=1,
                              padding=int((kernel_size - 1) // 2), bias=False)
    nn.init.constant_(gaussian_blur.weight.data, 0.0)
    gaussian_blur.weight.data[0, 0, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[1, 1, :, :] = torch.FloatTensor(kernel)
    gaussian_blur.weight.data[2, 2, :, :] = torch.FloatTensor(kernel)

    blur_tensor = gaussian_blur(gt_tensor)
    blur_tensor = blur_tensor[:, :, psize:-psize, psize:-psize]

    lrx4_blur_tensor = blur_tensor[:, :, ::4, ::4]
    lrx4_blur_tensor = lrx4_blur_tensor.clamp(0, 255).round()
    lrx4_blur = lrx4_blur_tensor[0].detach().numpy().transpose(1, 2, 0).astype('uint8')

    return lrx4_blur


def add_gaussian_noise_numpy(input_numpy, level=5, range=255.):
    noise = np.random.randn(*input_numpy.shape) * range * 0.01 * level
    input_numpy = input_numpy.astype('float32')
    out = input_numpy + noise
    out = np.round(np.clip(out, 0, range)).astype('uint8')
    return out


def gene_dataset_blurdown(HR_root, save_root):
    save_root_kernel = os.path.join(save_root, 'kernel')
    save_root_kernel_png = os.path.join(save_root, 'kernel_png')
    save_root_LR = os.path.join(save_root, 'blurdown_x4')
    handle_dir(save_root)
    handle_dir(save_root_kernel)
    handle_dir(save_root_kernel_png)
    handle_dir(save_root_LR)

    video_names = sorted(os.listdir(HR_root))
    for vn in video_names:
        handle_dir(os.path.join(save_root_kernel, vn))
        handle_dir(os.path.join(save_root_kernel_png, vn))
        handle_dir(os.path.join(save_root_LR, vn))

        frame_names = sorted(os.listdir(os.path.join(HR_root, vn)))

        kernel = get_blur_kernel(trian=False)
        for fn in frame_names:
            HR_img = cv2.imread(os.path.join(HR_root, vn, fn))
            LRx4 = get_lr_blurdown(HR_img, kernel)

            basename = fn.split(".")[0]
            cv2.imwrite(os.path.join(save_root_LR, vn, "{}.png".format(basename)), LRx4)
            cv2.imwrite(os.path.join(save_root_kernel_png, vn, "{}.png".format(basename)), kernel2png(kernel))
            scio.savemat(os.path.join(save_root_kernel, vn, "{}.mat".format(basename)), {'Kernel': kernel})

            print("{}-{} done!".format(vn, fn))


if __name__ == '__main__':
    gene_dataset_blurdown(
        HR_root='../dataset/REDS4_BlurDown_Gaussian/sharp',
        save_root='../dataset/REDS4_BlurDown_Gaussian'
    )

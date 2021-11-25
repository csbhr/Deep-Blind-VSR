from model.utils_fft import *


def deconv_sisr_L2_fft(y, k, scale, gamma=0.01, eps=1e-2):
    '''
    y: LR image, tensor, NxCxWxH
    k: kernel, tensor, Nx(1,3)xwxh
    scale: int
    gamma: float
    eps: float
    '''

    # warp boundary
    w_ori, h_ori = y.shape[-2:]
    img = upsample_nearest(y, scale)
    img = wrap_boundary_tensor(img, [int(np.ceil(scale * w_ori / 8 + 2) * 8), int(np.ceil(scale * h_ori / 8 + 2) * 8)])
    img_wrap = img[:, :, ::scale, ::scale]
    img_wrap[:, :, :w_ori, :h_ori] = y
    y = img_wrap

    # initialization & pre-calculation
    w, h = y.shape[-2:]
    FB = p2o(k, (w * scale, h * scale))
    FBC = cconj(FB, inplace=False)
    F2B = r2c(cabs2(FB))

    g1_kernel = torch.from_numpy(np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).reshape((1, 1, 3, 3))).type_as(k)
    g2_kernel = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).reshape((1, 1, 3, 3))).type_as(k)
    FG1 = p2o(g1_kernel, (w * scale, h * scale))
    FG2 = p2o(g2_kernel, (w * scale, h * scale))
    F2G1 = r2c(cabs2(FG1))
    F2G2 = r2c(cabs2(FG2))
    F2G = F2G1 + F2G2
    F2G[F2G < eps] = eps

    STy = upsample(y, sf=scale)
    FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))

    FR = FBFy
    x1 = cdiv(cmul(FB, FR), F2G)
    FBR = torch.mean(splits(x1, scale), dim=-1, keepdim=False)
    invW = torch.mean(splits(cdiv(F2B, F2G), scale), dim=-1, keepdim=False)
    invWBR = cdiv(FBR, csum(invW, gamma))
    FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, scale, scale, 1))
    FX = cdiv((FR - FCBinvWBR), F2G) / gamma
    Xest = torch.irfft(FX, 2, onesided=False)

    Xest = Xest[:, :, :scale * w_ori, :scale * h_ori]

    return Xest


def deconv_batch(lr, kernel, scale, gamma=0.001):
    with torch.no_grad():
        b, c, h, w = lr.shape
        b_list = []
        for i in range(b):
            deconv = deconv_sisr_L2_fft(lr[i:i + 1, :, :, :], kernel[i:i + 1, :, :, :], scale=scale, gamma=gamma)
            b_list.append(deconv)
        deconv_result = torch.cat(b_list, dim=0)
        return deconv_result

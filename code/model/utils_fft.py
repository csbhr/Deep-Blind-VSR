import torch
import numpy as np
from scipy import fftpack
import torch.nn.functional as F

"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""


def splits(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxWxHx2
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x2x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b


def c2c(x):
    return torch.from_numpy(np.stack([np.float32(x.real), np.float32(x.imag)], axis=-1))


def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)


def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c ** 2 + d ** 2
    return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)


def crdiv(x, y):
    # complex/real division
    a, b = x[..., 0], x[..., 1]
    return torch.stack([a / y, b / y], -1)


def csum(x, y):
    # complex + real
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
    # modulus of a complex number
    return torch.pow(x[..., 0] ** 2 + x[..., 1] ** 2, 0.5)


def cabs2(x):
    return x[..., 0] ** 2 + x[..., 1] ** 2


def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def rfft(t):
    # Real-to-complex Discrete Fourier Transform
    return torch.rfft(t, 2, onesided=False)


def irfft(t):
    # Complex-to-real Inverse Discrete Fourier Transform
    return torch.irfft(t, 2, onesided=False)


def fft(t):
    # Complex-to-complex Discrete Fourier Transform
    return torch.fft(t, 2)


def ifft(t):
    # Complex-to-complex Inverse Discrete Fourier Transform
    return torch.ifft(t, 2)


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    otf_np = otf.cpu().numpy()
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf_np = np.roll(otf_np, -int(axis_size / 2), axis=axis + 2)
    otf = torch.from_numpy(otf_np).type_as(psf)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops * 2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def upsample_nearest(x, sf=3):
    '''nearest downsampler

    x: tensor image, NxCxWxH
    '''
    x_repeat = x.repeat(1, sf * sf, 1, 1)
    x_up = F.pixel_shuffle(x_repeat, sf)
    return x_up


"""
# --------------------------------------------
# warp boundary to reduce boundary artifacts in image deconvolution
# --------------------------------------------
"""


def solve_min_laplacian(boundary_image):
    (H, W) = np.shape(boundary_image)

    # Laplacian
    f = np.zeros((H, W))
    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0
    j = np.arange(2, H) - 1
    k = np.arange(2, W) - 1
    f_bp = np.zeros((H, W))
    f_bp[np.ix_(j, k)] = -4 * boundary_image[np.ix_(j, k)] + boundary_image[np.ix_(j, k + 1)] + boundary_image[
        np.ix_(j, k - 1)] + boundary_image[np.ix_(j - 1, k)] + boundary_image[np.ix_(j + 1, k)]

    del (j, k)
    f1 = f - f_bp  # subtract boundary points contribution
    del (f_bp, f)

    # DST Sine Transform algo starts here
    f2 = f1[1:-1, 1:-1]
    del (f1)

    # compute sine tranform
    if f2.shape[1] == 1:
        tt = fftpack.dst(f2, type=1, axis=0) / 2
    else:
        tt = fftpack.dst(f2, type=1) / 2

    if tt.shape[0] == 1:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1, axis=0) / 2)
    else:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1) / 2)
    del (f2)

    # compute Eigen Values
    [x, y] = np.meshgrid(np.arange(1, W - 1), np.arange(1, H - 1))
    denom = (2 * np.cos(np.pi * x / (W - 1)) - 2) + (2 * np.cos(np.pi * y / (H - 1)) - 2)

    # divide
    f3 = f2sin / denom
    del (f2sin, x, y)

    # compute Inverse Sine Transform
    if f3.shape[0] == 1:
        tt = fftpack.idst(f3 * 2, type=1, axis=1) / (2 * (f3.shape[1] + 1))
    else:
        tt = fftpack.idst(f3 * 2, type=1, axis=0) / (2 * (f3.shape[0] + 1))
    del (f3)
    if tt.shape[1] == 1:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt) * 2, type=1) / (2 * (tt.shape[0] + 1)))
    else:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt) * 2, type=1, axis=0) / (2 * (tt.shape[1] + 1)))
    del (tt)

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image
    img_direct[1:-1, 1:-1] = 0
    img_direct[1:-1, 1:-1] = img_tt
    return img_direct


def wrap_boundary(img, img_size):
    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    (H, W) = np.shape(img)
    H_w = int(img_size[0]) - H
    W_w = int(img_size[1]) - W

    # ret = np.zeros((img_size[0], img_size[1]));
    alpha = 1
    HG = img[:, :]

    r_A = np.zeros((alpha * 2 + H_w, W))
    r_A[:alpha, :] = HG[-alpha:, :]
    r_A[-alpha:, :] = HG[:alpha, :]
    a = np.arange(H_w) / (H_w - 1)
    # r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)
    r_A[alpha:-alpha, 0] = (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0]
    # r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end)
    r_A[alpha:-alpha, -1] = (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1]

    r_B = np.zeros((H, alpha * 2 + W_w))
    r_B[:, :alpha] = HG[:, -alpha:]
    r_B[:, -alpha:] = HG[:, :alpha]
    a = np.arange(W_w) / (W_w - 1)
    r_B[0, alpha:-alpha] = (1 - a) * r_B[0, alpha - 1] + a * r_B[0, -alpha]
    r_B[-1, alpha:-alpha] = (1 - a) * r_B[-1, alpha - 1] + a * r_B[-1, -alpha]

    if alpha == 1:
        A2 = solve_min_laplacian(r_A[alpha - 1:, :])
        B2 = solve_min_laplacian(r_B[:, alpha - 1:])
        r_A[alpha - 1:, :] = A2
        r_B[:, alpha - 1:] = B2
    else:
        A2 = solve_min_laplacian(r_A[alpha - 1:-alpha + 1, :])
        r_A[alpha - 1:-alpha + 1, :] = A2
        B2 = solve_min_laplacian(r_B[:, alpha - 1:-alpha + 1])
        r_B[:, alpha - 1:-alpha + 1] = B2
    A = r_A
    B = r_B

    r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w))
    r_C[:alpha, :] = B[-alpha:, :]
    r_C[-alpha:, :] = B[:alpha, :]
    r_C[:, :alpha] = A[:, -alpha:]
    r_C[:, -alpha:] = A[:, :alpha]

    if alpha == 1:
        C2 = C2 = solve_min_laplacian(r_C[alpha - 1:, alpha - 1:])
        r_C[alpha - 1:, alpha - 1:] = C2
    else:
        C2 = solve_min_laplacian(r_C[alpha - 1:-alpha + 1, alpha - 1:-alpha + 1])
        r_C[alpha - 1:-alpha + 1, alpha - 1:-alpha + 1] = C2
    C = r_C
    # return C
    A = A[alpha - 1:-alpha - 1, :]
    B = B[:, alpha:-alpha]
    C = C[alpha:-alpha, alpha:-alpha]
    ret = np.vstack((np.hstack((img, B)), np.hstack((A, C))))
    return ret


def wrap_boundary_liu(img, img_size):
    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    if img.ndim == 2:
        ret = wrap_boundary(img, img_size)
    elif img.ndim == 3:
        ret = [wrap_boundary(img[:, :, i], img_size) for i in range(3)]
        ret = np.stack(ret, 2)
    return ret


def wrap_boundary_tensor(img_tensor, img_size):
    b, c, _, _ = img_tensor.size()
    b_list = []
    for i in range(b):
        c_list = []
        for j in range(c):
            c_list.append(torch.from_numpy(
                wrap_boundary(img_tensor[i, j, :, :].cpu().numpy(), img_size)
            ).type_as(img_tensor))
        b_list.append(torch.stack(c_list, dim=0))
    ret = torch.stack(b_list, dim=0)
    return ret

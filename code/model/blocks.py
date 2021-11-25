import torch
import torch.nn as nn
import torch.nn.functional as F


###############################
# common
###############################

def get_act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def get_norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batchnorm':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instancenorm':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def get_sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
        else:
            raise Exception("Unsupport module type [{:s}]".format(type(module)))
    return nn.Sequential(*modules)


class CommonConv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 pad_type='same', norm_type=None, act_type=None, mode='CNA'):
        super(CommonConv, self).__init__()

        mode = mode.upper()
        pad_type = pad_type.lower()
        norm_type = norm_type.lower()
        act_type = act_type.lower()

        if pad_type == 'zero':
            padding = 0
        elif pad_type == 'same':
            padding = get_same_padding(kernel_size, dilation)
        else:
            raise NotImplementedError('padding type [{:s}] is not found'.format(pad_type))
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias, groups=groups)

        self.act = get_act(act_type=act_type) if act_type else None

        if mode == "CNA":
            self.norm = get_norm(norm_type=norm_type, nc=out_nc) if norm_type else None
        elif mode == "NAC":
            self.norm = get_norm(norm_type=norm_type, nc=in_nc) if norm_type else None
        else:
            raise NotImplementedError('convolution mode [{:s}] is not found'.format(mode))

        self.mode = mode
        self.pad_type = pad_type
        self.norm_type = norm_type
        self.act_type = act_type

    def forward(self, x):
        if self.mode == "CNA":
            x = self.conv(x)
            x = self.norm(x) if self.norm else x
            x = self.act(x) if self.act else x
        elif self.mode == "NAC":
            x = self.norm(x) if self.norm else x
            x = self.act(x) if self.act else x
            x = self.conv(x)
        else:
            x = x
        return x


###############################
# ResNet
###############################

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out


###############################
# Residual Dense Network
###############################

class DenseConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, act_type='relu'):
        super(DenseConv, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                              padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.act = get_act(act_type=act_type) if act_type else None

    def forward(self, x):
        output = self.act(self.conv(x))
        return torch.cat((x, output), 1)


class RDB(nn.Module):

    def __init__(self, inplanes, planes, midplanes=None, n_conv=6, kernel_size=3, stride=1, dilation=1):
        super(RDB, self).__init__()

        if not midplanes:
            midplanes = inplanes

        layers = []
        for i in range(n_conv):
            layers.append(DenseConv(inplanes + i * midplanes, midplanes,
                                    kernel_size=kernel_size, stride=1, dilation=dilation))
        layers.append(nn.Conv2d(inplanes + n_conv * midplanes, planes, kernel_size=1, stride=stride))
        self.layers = nn.Sequential(*layers)

        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.layers(x)

        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out


###############################
# U-net
###############################

class UnetBottleneck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, act_type='relu'):
        super(UnetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.act = get_act(act_type=act_type) if act_type else None

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        return out


class UnetDownBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, act_type='relu'):
        super(UnetDownBlock, self).__init__()
        self.down = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1)
        self.conv = UnetBottleneck(inplanes, planes, kernel_size=kernel_size, dilation=dilation, act_type=act_type)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class UnetUpBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, act_type='relu'):
        super(UnetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(inplanes, inplanes // 2, kernel_size=4, stride=2, padding=1)
        self.conv = UnetBottleneck(inplanes, planes, kernel_size=kernel_size, dilation=dilation, act_type=act_type)

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        out = self.conv(torch.cat([x2, x1_up], dim=1))
        return out


###############################
# SFT-net
###############################


class SFTLayer(nn.Module):
    def __init__(self, n_feat, n_cond):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(n_cond, n_cond, 1)
        self.SFT_scale_conv1 = nn.Conv2d(n_cond, n_feat, 1)
        self.SFT_shift_conv0 = nn.Conv2d(n_cond, n_cond, 1)
        self.SFT_shift_conv1 = nn.Conv2d(n_cond, n_feat, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_SFT(nn.Module):
    def __init__(self, n_feat, n_cond):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer(n_feat, n_cond)
        self.conv0 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.sft1 = SFTLayer(n_feat, n_cond)
        self.conv1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions

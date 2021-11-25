import torch
import torch.nn as nn
from model import flow_pwc
import model.blocks as blocks
from model.kernel import KernelNet
import torch.nn.functional as F
import math
import model.deconv_fft as deconv_fft


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    flow_pretrain_fn = args.pretrain_models_dir + 'network-default.pytorch'
    kernel_pretrain_fn = args.pretrain_models_dir + 'kernel_x4.pt'
    return PWC_Recons(n_colors=args.n_colors, n_sequence=args.n_sequence, extra_RBS=args.extra_RBS,
                      recons_RBS=args.recons_RBS, n_feat=args.n_feat, n_cond=args.n_cond, est_ksize=args.est_ksize,
                      scale=args.scale, flow_pretrain_fn=flow_pretrain_fn, kernel_pretrain_fn=kernel_pretrain_fn,
                      device=device)


class PWC_Recons(nn.Module):

    def __init__(self, n_colors=3, n_sequence=5, extra_RBS=1, recons_RBS=3, n_feat=32, n_cond=64, est_ksize=13,
                 kernel_size=3, scale=4, flow_pretrain_fn='.', kernel_pretrain_fn='.', device='cuda'):
        super(PWC_Recons, self).__init__()
        print("Creating PWC-Recons Net")

        self.n_sequence = n_sequence
        self.scale = scale

        In_conv = [nn.Conv2d(n_colors, n_feat, kernel_size=3, stride=1, padding=1)]

        Extra_feat = []
        Extra_feat.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1)
                           for _ in range(extra_RBS)])

        Fusion_conv = [nn.Conv2d(n_feat * n_sequence, n_feat, kernel_size=3, stride=1, padding=1)]

        Recons_net = []
        Recons_net.extend([blocks.ResBlock_SFT(n_feat, n_cond) for _ in range(recons_RBS)])
        Recons_net.extend([
            blocks.SFTLayer(n_feat, n_cond),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        ])

        Out_conv = [
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(n_feat, n_colors, kernel_size=3, stride=1, padding=1)
        ]

        CondNet = [
            nn.Conv2d(n_colors * scale * scale, n_feat, 5, 1, 4, dilation=2), nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 2, dilation=2), nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_cond, 1)
        ]

        Upsample_layers = []
        for _ in range(int(math.log2(scale))):
            Upsample_layers.append(nn.Conv2d(n_feat, n_feat * 4, 3, 1, 1, bias=True))
            Upsample_layers.append(nn.PixelShuffle(2))

        self.in_conv = nn.Sequential(*In_conv)
        self.extra_feat = nn.Sequential(*Extra_feat)
        self.fusion_conv = nn.Sequential(*Fusion_conv)
        self.recons_net = nn.Sequential(*Recons_net)
        self.out_conv = nn.Sequential(*Out_conv)
        self.upsample_layers = nn.Sequential(*Upsample_layers)
        self.flow_net = flow_pwc.Flow_PWC(pretrain_fn=flow_pretrain_fn, device=device)
        self.kernel_net = KernelNet(ksize=est_ksize)
        self.cond_net = nn.Sequential(*CondNet)

        if kernel_pretrain_fn != '.':
            self.kernel_net.load_state_dict(torch.load(kernel_pretrain_fn))
            print('Loading KernelNet pretrain model from {}'.format(kernel_pretrain_fn))

    def forward(self, input_dict):
        x = input_dict['x']
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]
        frame_feat_list = [self.extra_feat(self.in_conv(frame)) for frame in frame_list]

        kernel_list = [self.kernel_net(f) for f in frame_list]
        deconv = deconv_fft.deconv_batch(
            frame_list[self.n_sequence // 2],
            kernel_list[self.n_sequence // 2],
            self.scale
        )
        deconv_S2D = self.spatial2depth(deconv, self.scale)
        cond = self.cond_net(deconv_S2D)

        base = frame_list[self.n_sequence // 2]
        base = F.interpolate(base, scale_factor=self.scale, mode='bilinear', align_corners=False)

        warped_feat_list = []
        for i in range(self.n_sequence):
            if not i == self.n_sequence // 2:
                flow = self.flow_net(frame_list[self.n_sequence // 2], frame_list[i])
                warped_feat, _ = self.flow_net.warp(frame_feat_list[i], flow)
                warped_feat_list.append(warped_feat)
            else:
                warped_feat_list.append(frame_feat_list[i])

        fusion_feat = self.fusion_conv(torch.cat(warped_feat_list, dim=1))
        recons_feat = self.recons_net((fusion_feat, cond))
        recons = self.out_conv(self.upsample_layers(recons_feat))
        recons = recons + base

        mid_loss = None

        return {
            'recons': recons,
            'kernel_list': kernel_list
               }, mid_loss

    def spatial2depth(self, spatial, scale):
        depth_list = []
        for i in range(scale):
            for j in range(scale):
                depth_list.append(spatial[:, :, i::scale, j::scale])
        depth = torch.cat(depth_list, dim=1)
        return depth

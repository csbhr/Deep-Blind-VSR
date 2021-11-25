import decimal
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import data_utils
from trainer.trainer import Trainer


class Trainer_Flow_Video(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_Flow_Video, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer_Flow_Video")

        self.l1_loss = torch.nn.L1Loss()
        self.cycle_psnr_log = []
        self.mid_loss_log = []
        self.cycle_loss_log = []

        if args.load != '.':
            mid_logs = torch.load(os.path.join(ckp.dir, 'mid_logs.pt'))
            self.cycle_psnr_log = mid_logs['cycle_psnr_log']
            self.mid_loss_log = mid_logs['mid_loss_log']
            self.cycle_loss_log = mid_logs['cycle_loss_log']

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = optim.Adam([{"params": self.model.get_model().in_conv.parameters()},
                                {"params": self.model.get_model().extra_feat.parameters()},
                                {"params": self.model.get_model().fusion_conv.parameters()},
                                {"params": self.model.get_model().recons_net.parameters()},
                                {"params": self.model.get_model().upsample_layers.parameters()},
                                {"params": self.model.get_model().out_conv.parameters()},
                                {"params": self.model.get_model().flow_net.parameters(), "lr": 1e-6},
                                {"params": self.model.get_model().kernel_net.parameters(), "lr": 1e-6},
                                {"params": self.model.get_model().cond_net.parameters()}],
                               **kwargs)
        return optimizer

    def train(self):
        print("Now training")
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        mid_loss_sum = 0.
        cycle_loss_sum = 0.

        for batch, (input, gt, _) in enumerate(self.loader_train):

            input = input.to(self.device)
            gt_list = [gt[:, i, :, :, :].to(self.device) for i in range(self.args.n_sequence)]
            gt = gt_list[self.args.n_sequence // 2]

            output_dict, mid_loss = self.model({'x': input})
            output = output_dict['recons']
            kernel_list = output_dict['kernel_list']

            self.optimizer.zero_grad()

            loss = self.loss(output, gt)

            lr_cycle_list = [self.blur_down(g, k, self.args.scale) for g, k in zip(gt_list, kernel_list)]
            cycle_loss = 0.
            for i, lr_cycle in enumerate(lr_cycle_list):
                cycle_loss = cycle_loss + self.l1_loss(lr_cycle, input[:, i, :, :, :])
            cycle_loss_sum = cycle_loss_sum + cycle_loss.item()
            loss = loss + cycle_loss

            if mid_loss:  # mid loss is the loss during the model
                loss = loss + self.args.mid_loss_weight * mid_loss
                mid_loss_sum = mid_loss_sum + mid_loss.item()
            loss.backward()
            self.optimizer.step()

            self.ckp.report_log(loss.item())

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}[cycle: {:.4f}][mid: {:.4f}]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1),
                    self.loss.display_loss(batch),
                    cycle_loss_sum / (batch + 1),
                    mid_loss_sum / (batch + 1)
                ))

        self.loss.end_log(len(self.loader_train))
        self.mid_loss_log.append(mid_loss_sum / len(self.loader_train))
        self.cycle_loss_log.append(cycle_loss_sum / len(self.loader_train))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        cycle_psnr_list = []

        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (input, gt, filename) in enumerate(tqdm_test):

                filename = filename[self.args.n_sequence // 2][0]

                input = input.to(self.device)
                input_center = input[:, self.args.n_sequence // 2, :, :, :]
                gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)

                output_dict, _ = self.model({'x': input})
                output = output_dict['recons']
                kernel_list = output_dict['kernel_list']
                est_kernel = kernel_list[self.args.n_sequence // 2]

                lr_cycle_center = self.blur_down(gt, est_kernel, self.args.scale)

                cycle_PSNR = data_utils.calc_psnr(input_center, lr_cycle_center, rgb_range=self.args.rgb_range,
                                                  is_rgb=True)
                PSNR = data_utils.calc_psnr(gt, output, rgb_range=self.args.rgb_range, is_rgb=True)
                self.ckp.report_log(PSNR, train=False)
                cycle_psnr_list.append(cycle_PSNR)

                if self.args.save_images:
                    gt, input_center, output, lr_cycle_center = data_utils.postprocess(
                        gt, input_center, output, lr_cycle_center,
                        rgb_range=self.args.rgb_range,
                        ycbcr_flag=False,
                        device=self.device)

                    est_kernel = self.process_kernel(est_kernel)

                    save_list = [gt, input_center, output, lr_cycle_center, est_kernel]
                    self.ckp.save_images(filename, save_list, epoch)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage Cycle-PSNR: {:.3f} PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test,
                sum(cycle_psnr_list) / len(cycle_psnr_list),
                self.ckp.psnr_log[-1],
                best[0], best[1] + 1))
            self.cycle_psnr_log.append(sum(cycle_psnr_list) / len(cycle_psnr_list))
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
                self.ckp.plot_log(self.cycle_psnr_log, filename='cycle_psnr.pdf', title='Cycle PSNR')
                self.ckp.plot_log(self.mid_loss_log, filename='mid_loss.pdf', title='Mid Loss')
                self.ckp.plot_log(self.cycle_loss_log, filename='cycle_loss.pdf', title='Cycle Loss')
                torch.save({
                    'cycle_psnr_log': self.cycle_psnr_log,
                    'mid_loss_log': self.mid_loss_log,
                    'cycle_loss_log': self.cycle_loss_log,
                }, os.path.join(self.ckp.dir, 'mid_logs.pt'))

    def conv_func(self, input, kernel, padding='same'):
        b, c, h, w = input.size()
        assert b == 1, "only support b=1!"
        _, _, ksize, ksize = kernel.size()
        if padding == 'same':
            pad = ksize // 2
        elif padding == 'valid':
            pad = 0
        else:
            raise Exception("not support padding flag!")

        conv_result_list = []
        for i in range(c):
            conv_result_list.append(F.conv2d(input[:, i:i + 1, :, :], kernel, bias=None, stride=1, padding=pad))
        conv_result = torch.cat(conv_result_list, dim=1)
        return conv_result

    def blur_down(self, x, kernel, scale):
        b, c, h, w = x.size()
        _, kc, ksize, _ = kernel.size()
        psize = ksize // 2
        assert kc == 1, "only support kc=1!"

        # blur
        x = F.pad(x, (psize, psize, psize, psize), mode='replicate')
        blur_list = []
        for i in range(b):
            blur_list.append(self.conv_func(x[i:i + 1, :, :, :], kernel[i:i + 1, :, :, :]))
        blur = torch.cat(blur_list, dim=0)
        blur = blur[:, :, psize:-psize, psize:-psize]

        # down
        blurdown = blur[:, :, ::scale, ::scale]

        return blurdown

    def process_kernel(self, kernel):
        mi = torch.min(kernel)
        ma = torch.max(kernel)
        kernel = (kernel - mi) / (ma - mi)
        kernel = torch.cat([kernel, kernel, kernel], dim=1)
        kernel = kernel.mul(255.).clamp(0, 255).round()
        return kernel

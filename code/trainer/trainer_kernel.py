import decimal
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import data_utils
from trainer.trainer import Trainer


class Trainer_Kernel(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_Kernel, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer_Kernel")

        self.mid_loss_log = []

        if args.load != '.':
            mid_logs = torch.load(os.path.join(ckp.dir, 'mid_logs.pt'))
            self.mid_loss_log = mid_logs['mid_loss_log']

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = optim.Adam([{"params": self.model.get_model().parameters()}],
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

        for batch, (input, gt, _) in enumerate(self.loader_train):

            input = input[:, self.args.n_sequence // 2, :, :, :].to(self.device)
            gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)

            kernel = self.model(input)
            lr_cycle = self.blur_down(gt, kernel, self.args.scale)

            self.optimizer.zero_grad()

            loss = self.loss(lr_cycle, input)

            loss.backward()
            self.optimizer.step()

            self.ckp.report_log(loss.item())

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}[mid: {:.4f}]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1),
                    self.loss.display_loss(batch),
                    mid_loss_sum / (batch + 1)
                ))

        self.loss.end_log(len(self.loader_train))
        self.mid_loss_log.append(mid_loss_sum / len(self.loader_train))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)

        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (input, gt, filename) in enumerate(tqdm_test):

                filename = filename[self.args.n_sequence // 2][0]

                input = input[:, self.args.n_sequence // 2, :, :, :].to(self.device)
                gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)

                est_kernel = self.model(input)

                lr_cycle_center = self.blur_down(gt, est_kernel, self.args.scale)

                PSNR = data_utils.calc_psnr(input, lr_cycle_center, rgb_range=self.args.rgb_range, is_rgb=True)
                self.ckp.report_log(PSNR, train=False)

                if self.args.save_images:
                    est_kernel = self.process_kernel(est_kernel)

                    save_list = [est_kernel]
                    self.ckp.save_images(filename, save_list, epoch)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test,
                self.ckp.psnr_log[-1],
                best[0], best[1] + 1))
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
                self.ckp.plot_log(self.mid_loss_log, filename='mid_loss.pdf', title='Mid Loss')
                torch.save({
                    'mid_loss_log': self.mid_loss_log,
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

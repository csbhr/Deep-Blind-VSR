import os
import glob
import utils.data_utils as utils
import numpy as np
import imageio
import torch
import cv2
import torch.utils.data as data


class VIDEODATA_HRLR(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.n_seq = args.n_sequence
        self.n_frames_per_video = args.n_frames_per_video
        print("n_seq:", args.n_sequence)
        print("n_frames_per_video:", args.n_frames_per_video)

        self.n_frames_video = []

        if train:
            self._set_filesystem(args.dir_data)
        else:
            self._set_filesystem(args.dir_data_test)

        self.images_gt, self.images_input = self._scan()

        self.num_video = len(self.images_gt)
        self.num_frame = sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        print("Number of frames to load:", self.num_frame)

        if train:
            self.repeat = max(args.test_every // max((self.num_frame // self.args.batch_size), 1), 1)
            print("Dataset repeat:", self.repeat)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'GT')
        self.dir_input = os.path.join(self.apath, 'LR')
        print("DataSet gt path:", self.dir_gt)
        print("DataSet lr path:", self.dir_input)

    def _scan(self):
        vid_gt_names = sorted(glob.glob(os.path.join(self.dir_gt, '*')))
        vid_input_names = sorted(glob.glob(os.path.join(self.dir_input, '*')))
        assert len(vid_gt_names) == len(vid_input_names), "len(vid_gt_names) must equal len(vid_input_names)"

        images_gt = []
        images_input = []

        for vid_gt_name, vid_input_name in zip(vid_gt_names, vid_input_names):
            if self.train:
                gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))[:self.args.n_frames_per_video]
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))[:self.args.n_frames_per_video]
            else:
                gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))
            images_gt.append(gt_dir_names)
            images_input.append(input_dir_names)
            self.n_frames_video.append(len(gt_dir_names))

        return images_gt, images_input

    def __getitem__(self, idx):
        inputs, gts, filenames = self._load_file(idx)

        inputs_list = [inputs[i, :, :, :] for i in range(self.n_seq)]
        inputs_concat = np.concatenate(inputs_list, axis=2)
        gts_list = [gts[i, :, :, :] for i in range(self.n_seq)]
        gts_concat = np.concatenate(gts_list, axis=2)
        inputs_concat, gts_concat = self.get_patch(inputs_concat, gts_concat,
                                                   self.args.size_must_mode, scale=self.args.scale)
        inputs_list = [inputs_concat[:, :, i * self.args.n_colors:(i + 1) * self.args.n_colors] for i in
                       range(self.n_seq)]
        gts_list = [gts_concat[:, :, i * self.args.n_colors:(i + 1) * self.args.n_colors] for i in range(self.n_seq)]

        inputs = np.array(inputs_list)
        gts = np.array(gts_list)

        input_tensors = utils.np2Tensor(*inputs, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        gt_tensors = utils.np2Tensor(*gts, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)

        return torch.stack(input_tensors), torch.stack(gt_tensors), filenames

    def __len__(self):
        if self.train:
            return self.num_frame * self.repeat
        else:
            return self.num_frame

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_frame
        else:
            return idx

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)  # test时，根据idx获取对应的视频id和帧id
        f_gts = self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        f_inputs = self.images_input[video_idx][frame_idx:frame_idx + self.n_seq]
        gts = np.array([imageio.imread(hr_name) for hr_name in f_gts], dtype=np.float)
        inputs = np.array([imageio.imread(lr_name) for lr_name in f_inputs], dtype=np.float)
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in f_gts]

        return inputs, gts, filenames

    def get_patch(self, input, gt, size_must_mode=1, scale=1):
        if self.train:
            input_patch, gt_patch = utils.get_patch(input, gt, patch_size=self.args.patch_size, scale=scale)
            mid_b, mid_e = (self.n_seq // 2) * self.args.n_colors, (self.n_seq // 2 + 1) * self.args.n_colors
            mean_edge = self.cal_smooth(gt_patch[:, :, mid_b:mid_e])
            n_loop = 1
            while mean_edge < 7 and n_loop < 5:  # drop smooth patch
                input_patch, gt_patch = utils.get_patch(input, gt, patch_size=self.args.patch_size, scale=scale)
                mean_edge = self.cal_smooth(gt_patch[:, :, mid_b:mid_e])
                n_loop += 1

            h, w, c = input_patch.shape
            size_must_mode = size_must_mode
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input_patch, gt_patch = input_patch[:new_h, :new_w, :], gt_patch[:new_h * scale, :new_w * scale, :]
            if not self.args.no_augment:
                input_patch, gt_patch = utils.data_augment(input_patch, gt_patch)
        else:
            input_patch, gt_patch = input, gt
            h, w, c = input_patch.shape
            size_must_mode = size_must_mode
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input_patch, gt_patch = input_patch[:new_h, :new_w, :], gt_patch[:new_h * scale, :new_w * scale, :]
        return input_patch, gt_patch

    def cal_smooth(self, img):
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        smooth = np.mean(dst)
        return smooth

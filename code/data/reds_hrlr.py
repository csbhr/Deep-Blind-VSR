import os
from data.videodata_hrlr import VIDEODATA_HRLR as VIDEODATA


class REDS_HRLR(VIDEODATA):
    def __init__(self, args, name='REDS_HRLR', train=True):
        super(REDS_HRLR, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'sharp')
        self.dir_input = os.path.join(self.apath, 'blurdown_x4')
        print("DataSet gt path:", self.dir_gt)
        print("DataSet lr path:", self.dir_input)

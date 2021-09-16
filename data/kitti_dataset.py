import os.path
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import glob
import imutils
from data.base_dataset import BaseDataset

def Random_Rotation(img,label, degrees=10):

    angle = np.random.uniform(-degrees, degrees)
    img = imutils.rotate(img, angle,  center=(img.shape[1]/ 2, img.shape[0]))
    label = imutils.rotate(label, angle, center=(label.shape[1] / 2, label.shape[0]))

    return img, label


class kittidataset(BaseDataset):
    """dataloader for kitti dataset"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.root = opt.dataroot # path for the dataset
        self.num_labels = 2
        self.use_size = (opt.useWidth, opt.useHeight)

        if opt.phase == "train":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'training', 'image_2', '*.png')))
        elif opt.phase == "val":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'validation', 'image_2', '*.png')))
        else:
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'testing', 'image_2', '*.png')))





    def __getitem__(self, index):
        useDir = "/".join(self.image_list[index].split('/')[:-2])
        name = self.image_list[index].split('/')[-1]

        rgb_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'image_2', name)), cv2.COLOR_BGR2RGB)
        oriHeight, oriWidth, _ = rgb_image.shape
        # resize image to enable sizes divide 32
        rgb_image = cv2.resize(rgb_image, self.use_size)

        if self.opt.phase == 'test' and self.opt.no_label:
            # Since we have no gt label (e.g., kitti submission), we generate pseudo gt labels to
            # avoid destroying the code architecture
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
            label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)

        elif self.opt.phase == "val" or self.opt.phase == 'test':

            label_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'gt_image_2', name)),
                                       cv2.COLOR_BGR2RGB)
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
            label[label_image[:, :, 0] > 0] = 1
            label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)
            label[label > 0] = 1

        else:#train

            label_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'gt_image_2', name)),
                                       cv2.COLOR_BGR2RGB)
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
            label[label_image[:, :, 0] > 0] = 1
            label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)
            label[label > 0] = 1

            if self.opt.data_augment:
                if np.random.random() <= self.opt.rotate_prob:
                    rgb_image,label =Random_Rotation(rgb_image,label,degrees=10)#Random_Rotation [-degrees,degrees]
                if np.random.random() <= self.opt.mirror_prob:
                    rgb_image = cv2.flip(rgb_image ,1, dst=None) #Random_mirror
                    label= cv2.flip(label, 1, dst=None)


        rgb_image = rgb_image.astype(np.float32) / 255
        rgb_image = transforms.ToTensor()(rgb_image)
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        # return a dictionary containing useful information
        # input rgb images and labels for training;
        # 'path': image name for saving predictions
        # 'oriSize': original image size for evaluating and saving predictions
        return {'rgb_image': rgb_image, 'label': label,
                'path': name, 'oriSize': (oriWidth, oriHeight)}

    def __len__(self):
        return len(self.image_list)

    def name(self):
        return 'kitti'

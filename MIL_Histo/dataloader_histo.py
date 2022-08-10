' 使用的是COLON CANCER数据集 '

import glob
import os
import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from PIL import Image
from patchify import patchify, unpatchify
import scipy.io as scio


class train_dataloader_histo(data_utils.Dataset):
    def __init__(self, k):
        # self.dataset_path = 'C:/Users/mi/Desktop/Lab/code/MIL/MIL_Histo/datasets_histo/PATCHES COLON CANCER/*'
        self.dataset_path = 'datasets_histo/PATCHES COLON CANCER/*'
        self.k = k
        self.bags, self.bags_labels, self.instances_labels, self.patches_length = self.get_bags(self.dataset_path)

    def get_bags(self, dataset_path):  # 获取包
        bags_list = []
        bags_labels_list = []
        instances_labels_list = []
        patches_length = 0  # 所有patch长度
        # test_signal1 = True
        # test_signal2 = True

        for path in glob.glob(dataset_path):  # path是0、1文件夹
            # print('dataset_path', dataset_path)
            # print('path', path)
            label = int(path.split('/')[-1])  # 包的标签
            # print('label', label)
            for each_path in glob.glob(path + '/*'):  # each_path是img文件夹
                # if test_signal1:
                #     # print('each_path', each_path)
                #     test_signal1 = False
                slide_patches = []  # 存放一张bmp图所有patch
                slide_patches_labels = []  # 存放一张bmp图所有patch标签
                patch_path = glob.glob(each_path + '/*.bmp')  # 获取的是所有patch的路径
                # if test_signal2:
                #     # print('patch_path', patch_path)
                #     test_signal2 = False
                img_length = len(patch_path)  # 每张图包含的的有效patch数
                # print('img_length', img_length)
                patches_length += img_length

                if label == 0:  # 如果这个包是个负包
                    for file in patch_path:  # 开始读取patch
                        patch = Image.open(file)  # 先打开图片

                        instance = self.patch_normalization(patch)  # 图片归一化
                        instance = np.array(instance)  # 图片转为numpy数组
                        slide_patches.append(instance)  # instance（patch）组成包
                        slide_patches_labels.append(label)  # 包的标签

                        instance_ = self.patch_augmentation(patch)
                        instance_ = np.array(instance_)
                        slide_patches.append(instance_)
                        slide_patches_labels.append(label)

                elif label == 1:  # 表示这个包内可能包含有colon cancer对应的块
                    for file in patch_path:
                        patch = Image.open(file)

                        instance = self.patch_normalization(patch)
                        instance = np.array(instance)
                        slide_patches.append(instance)
                        if (os.path.basename(file).split(".")[-2]).split("-")[-1] == "epithelial":
                            slide_patches_labels.append(int(1))  # 包含有colon cancer对应的块，正包
                        else:
                            slide_patches_labels.append(int(0))  # 不包含有colon cancer对应的块，负包

                        instance_ = self.patch_augmentation(patch)
                        instance_ = np.array(instance_)
                        slide_patches.append(instance_)
                        slide_patches_labels.append(label)
                        if (os.path.basename(file).split(".")[-2]).split("-")[-1] == "epithelial":
                            slide_patches_labels.append(int(1))
                        else:
                            slide_patches_labels.append(int(0))

                slide_patches = np.array(slide_patches)
                slide_patches_labels = np.array(slide_patches_labels)

                slide_patches = torch.from_numpy(slide_patches).float()  # numpy数组转为tensor格式
                slide_patches_labels = torch.from_numpy(slide_patches_labels).float()
                # slide_patches = torch.unsqueeze(slide_patches, 1)

                bags_list.append(slide_patches)
                bags_labels_list.append(max(slide_patches_labels))
                instances_labels_list.append(slide_patches_labels)
        # print('有效patch数量(包括增强部分)：', len(patches))
        # print('patches.shape', patches.shape)
        # print('patches_labels.shape', patches_labels.shape)
        # length = len(bags_list)

        del_index = list(range(self.k-1, 99, 10))
        bags_list = [i for index, i in enumerate(bags_list) if index not in del_index]
        bags_labels_list = [i for index, i in enumerate(bags_labels_list) if index not in del_index]
        instances_labels_list = [i for index, i in enumerate(instances_labels_list) if index not in del_index]

        patches_length = int(patches_length * 0.9)
        return bags_list, bags_labels_list, instances_labels_list, patches_length

    def patch_augmentation(self, patch_aug):  # 数据增强：随机旋转，镜像后归一化
        patch_transforms = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        patch_ = patch_transforms(patch_aug)
        return patch_

    def patch_normalization(self, patch):  # 数据归一化
        patch_transforms = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                               ])
        patch_ = patch_transforms(patch)
        return patch_

    def __len__(self):
        # print('生成的包长度为：', len(self.bags))
        return len(self.bags)

    def __getitem__(self, index):
        # if index not in list(range((self.k - 1) * 10, self.k * 10 - 1)):
        bag = self.bags[index]
        label = [self.bags_labels[index], self.instances_labels[index]]
        return bag, label


# class test_dataloader_histo(data_utils.Dataset):
#     def __init__(self, tile_size=27, seed=1):
#         self.tile_size = tile_size
#         self.r = np.random.RandomState(seed)
#         # self.dataset_path = 'C:/Users/mi/Desktop/Lab/code/MIL/MIL_Histo/datasets_histo/SOURCE COLON CANCER/Classification/*'
#         self.dataset_path = 'datasets_histo/SOURCE COLON CANCER/Classification/*'
#         self.bags, self.bags_labels = self.get_bags(self.dataset_path)
#
#     def get_bags(self, dataset_path):
#         bags_list = []
#         bags_labels_list = []
#         # instances_labels_list = []
#         for path in glob.glob(dataset_path):    # path: classification
#             for each_path in glob.glob(path):   # path: img1
#                 slide_patches = []
#                 slide_path = glob.glob(each_path + "/*.bmp")  # 一张图就是一个包
#                 for slide_path_ in slide_path:
#                     slide = Image.open(slide_path_)
#                     data = np.array(slide)
#                     patches = patchify(data, (self.tile_size, self.tile_size, 3), step=self.tile_size)
#                     for row in range(patches.shape[0]-1):
#                         for col in range(patches.shape[1]-1):
#                             patch = Image.fromarray(patches[row][col][0], 'RGB')
#                             patch_ = self.patch_normalization(patch)
#                             patch_ = np.array(patch_)
#                             slide_patches.append(patch_)
#
#                     slide_patches = np.array(slide_patches)
#                     slide_patches = torch.from_numpy(slide_patches).float()
#                     bags_list.append(slide_patches)
#
#                     slide_mat_path = glob.glob(each_path + "/*epithelial.mat")
#                     # print('slide_mat_path', slide_mat_path)
#                     for slide_mat in slide_mat_path:
#                         data_mat = scio.loadmat(slide_mat)
#                         if data_mat['detection'].size > 0:
#                             bags_labels_list.append(torch.FloatTensor([1]))
#                         else:
#                             bags_labels_list.append(torch.FloatTensor([0]))
#
#         return bags_list, bags_labels_list
#
#     def patch_normalization(self, patch_aug):
#         patch_transforms = transforms.Compose([transforms.ToTensor(),
#                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                                                ])
#         patch_ = patch_transforms(patch_aug)
#         return patch_
#
#     def __len__(self):
#         # print('生成的包长度为：', len(self.bags))
#         return len(self.bags)
#
#     def __getitem__(self, index):
#         bag = self.bags[index]
#         label = self.bags_labels[index]
#         return bag, label

class test_dataloader_histo(data_utils.Dataset):
    def __init__(self, k):
        # self.dataset_path = 'C:/Users/mi/Desktop/Lab/code/MIL/MIL_Histo/datasets_histo/PATCHES COLON CANCER/*'
        self.dataset_path = 'datasets_histo/PATCHES COLON CANCER/*'
        self.k = k
        self.bags, self.bags_labels, self.instances_labels, self.patches_length = self.get_bags(self.dataset_path)

    def get_bags(self, dataset_path):  # get_bags
        bags_list = []
        bags_labels_list = []
        instances_labels_list = []
        patches_length = 0  # 所有patch长度
        # test_signal1 = True
        # test_signal2 = True

        for path in glob.glob(dataset_path):  # path是0、1文件夹
            # print('dataset_path', dataset_path)
            # print('path', path)
            label = int(path.split('/')[-1])  # 包的标签
            # print('label', label)
            for each_path in glob.glob(path + '/*'):  # each_path是img文件夹
                # if test_signal1:
                #     # print('each_path', each_path)
                #     test_signal1 = False
                slide_patches = []  # 存放一张bmp图所有patch
                slide_patches_labels = []  # 存放一张bmp图所有patch标签
                patch_path = glob.glob(each_path + '/*.bmp')  # 获取的是所有patch的路径
                # if test_signal2:
                #     # print('patch_path', patch_path)
                #     test_signal2 = False
                img_length = len(patch_path)  # 每张图包含的的有效patch数
                # print('img_length', img_length)
                patches_length += img_length

                if label == 0:
                    for file in patch_path:
                        patch = Image.open(file)

                        instance = self.patch_normalization(patch)
                        instance = np.array(instance)
                        slide_patches.append(instance)
                        slide_patches_labels.append(label)

                elif label == 1:
                    for file in patch_path:
                        patch = Image.open(file)

                        instance = self.patch_normalization(patch)
                        instance = np.array(instance)
                        slide_patches.append(instance)
                        if (os.path.basename(file).split(".")[-2]).split("-")[-1] == "epithelial":
                            slide_patches_labels.append(int(1))
                        else:
                            slide_patches_labels.append(int(0))

                slide_patches = np.array(slide_patches)
                slide_patches_labels = np.array(slide_patches_labels)

                slide_patches = torch.from_numpy(slide_patches).float()
                slide_patches_labels = torch.from_numpy(slide_patches_labels).float()
                # slide_patches = torch.unsqueeze(slide_patches, 1)

                bags_list.append(slide_patches)
                bags_labels_list.append(max(slide_patches_labels))
                instances_labels_list.append(slide_patches_labels)
        # print('有效patch数量(包括增强部分)：', len(patches))
        # print('patches.shape', patches.shape)
        # print('patches_labels.shape', patches_labels.shape)

        del_index = list(range(self.k-1, 99, 10))
        bags_list = [i for index, i in enumerate(bags_list) if index in del_index]
        bags_labels_list = [i for index, i in enumerate(bags_labels_list) if index in del_index]
        instances_labels_list = [i for index, i in enumerate(instances_labels_list) if index in del_index]

        patches_length = int(patches_length * 0.1)
        return bags_list, bags_labels_list, instances_labels_list, patches_length

    def patch_normalization(self, patch):
        patch_transforms = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                               ])
        patch_ = patch_transforms(patch)
        return patch_

    def __len__(self):
        # print('生成的包长度为：', len(self.bags))
        return len(self.bags)

    def __getitem__(self, index):
        bag = self.bags[index]
        label = [self.bags_labels[index], self.instances_labels[index]]
        return bag, label


if __name__ == "__main__":
    train_loader = train_dataloader_histo()
    print('有效patch数量(不包括包括增强部分):', train_loader.patches_length)
    print('有效包的数量:', len(train_loader.bags))
    print('其中1个包的尺寸：', train_loader.bags[10].shape)

    print('有效包标签的数量:', len(train_loader.bags_labels))
    print('包标签分布：', train_loader.bags_labels)
    print('1个包的示例：', train_loader.bags[10])
    print('1个包标签的示例：', train_loader.bags_labels[0])

    # test_loader = test_dataloader_histo()
    # # print('有效patch数量(不包括包括增强部分):', test_loader.patches_length)
    # print('有效包的数量:', len(test_loader.bags))
    # print('包标签分布：', test_loader.bags_labels)
    #
    # print('1个包的示例：', test_loader.bags[0])
    # print('1个包标签的示例：', test_loader.bags_labels[0])

"""
正负包比例随机
"""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=50, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)    # 伪随机数种子，保证生成的包不换含有相同的样本

        self.num_in_train = 60000   # 训练样本数量(手写数字图片的张数)
        self.num_in_test = 10000   # 测试样本数量(手写数字图片的张数)

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()  # 生成训练用的包
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()  # 生成测试用的包

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('./datasets',
                                                          train=True,
                                                          download=False,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('./datasets',
                                                          train=False,
                                                          download=False,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:   # batch_size=self.num_in_train，一次性取完所有图片
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))  # 生成一个包(包与包不会重合)
            if bag_length < 1:
                bag_length = 1

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))  # 随机生成索引，用于构建训练包
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))  # 随机生成索引，用于构建测试包

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)  # 返回总训练包的个数
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):   # __getitem__用于向迭代器dataloader传送数据
        if self.train:
            bag = self.train_bags_list[index]   # 返回第index个训练包
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=100,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=100,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
            mnist_bags_train, len(train_loader),
            np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
            mnist_bags_test, len(test_loader),
            np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))

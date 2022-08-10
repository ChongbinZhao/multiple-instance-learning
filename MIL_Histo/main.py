from __future__ import print_function
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
import datetime
from dataloader_histo import train_dataloader_histo, test_dataloader_histo
from model_histo import Attention, GatedAttention

# 训练参数设置
parser = argparse.ArgumentParser(description='Histopathology bags Example')
parser.add_argument('--epochs', type=int, default=100, metavar='N',  # 训练的epochs
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',  # 学习率
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=5e-4, metavar='R',  # 权重衰减
                    help='weight decay')
parser.add_argument('--num_bags_train', type=int, default=89, metavar='NTrain',  # 训练的包的数量
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=10, metavar='NTest',  # 测试的包的数量
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',  # 随机数种子
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,  # 是否使用gpu训练
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')  # 模型

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
k = 10

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Init Model')
if args.model == 'attention':  # 选择attention模型
    model = Attention()
elif args.model == 'gated_attention':  # 选择门控attention模型
    model = GatedAttention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)  # 亚当优化器
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}  # 线程数

print('Load Train and Test Set')

train_loader = data_utils.DataLoader(dataset=train_dataloader_histo(k),  # 迭代器获取训练集
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(dataset=test_dataloader_histo(k),  # 迭代器获取测试集
                                    batch_size=1,
                                    shuffle=True,
                                    **loader_kwargs)


def train(epoch):
    # print('训练包长度：',len(train_loader))
    model.train()  # 设置为训练模式，这样训练过程中就会保留dropout和BN
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):  # 获取每个包的和它的索引
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        optimizer.zero_grad()  # 训练前梯度置零

        loss, _ = model.calculate_objective(data, bag_label)  # 计算目标函数（负对数损失函数）
        train_loss += loss.item()  # 计算目标函数（负对数损失函数）
        error, _, _ = model.calculate_classification_error(data, bag_label)  # 计算误差（误差和目标函数不同）
        train_error += error

        loss.backward()  # 梯度反向传播
        optimizer.step()  # 根据梯度更新权重

    train_loss /= len(train_loader)  # 训练损失
    train_error /= len(train_loader)  # 训练误差

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))


def test():
    # print('测试包长度：', len(test_loader))
    model.eval()
    test_loss = 0.
    test_error = 0.
    y_probs = []
    y_probs_labels = []
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        y_prob_label = bag_label
        # y_prob_label = label
        y_probs_labels.append(int(y_prob_label.cpu().data.numpy()[0]))  # 可能性对应的标签(测试包的标签)
        instance_labels = label[1]  # 实例标签是one-hot的形式
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        # 这的attention_weights指的是αk(softmax后的)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.item()
        error, predicted_label, y_prob = model.calculate_classification_error(data, bag_label)
        y_probs.append(np.round(y_prob.cpu().data.numpy()[0][0], decimals=4))  # 可能性
        test_error += error

        # if batch_idx >= args.num_bags_test - 2:  # plot bag labels and instance labels for the last bag
        #     if batch_idx == args.num_bags_test - 2:
        #         print('\nplot bag labels and instance labels for the last bag')
        #     bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
        #     # instance_level: zip函数让每个元素实例标签与其权重绑在一起
        #     # instance_level = list(np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist())
        #     instance_level = list(zip(instance_labels.numpy()[0].tolist(),
        #                               np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))
        #     print('\nTrue Bag Label, Predicted Bag Label: {}\n'
        #           'Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    y_probs_acc = np.array(y_probs) > 0.5
    y_probs_acc = y_probs_acc == np.array(y_probs_labels)
    acc = np.round(sum(y_probs_acc) / len(y_probs_labels), decimals=3)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))
    print(80 * '=')
    print('y_probs', y_probs)
    print('y_probs_labels', y_probs_labels)

    # 绘制ROC曲线与并计算AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_probs_labels, y_probs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(80 * '=')
    print('AUC', auc)
    print('accuracy', acc)
    end = datetime.datetime.now()
    print("运行耗时：", end - start)
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % auc)
    # plt.legend(loc='lower right')
    # # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([-0.1, 1.1])
    # plt.ylim([-0.1, 1.1])
    # plt.xlabel('False Positive Rate')  # 横坐标是fpr
    # plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    # plt.title('Histopathology: ' + args.model)
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    start = datetime.datetime.now()
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test()
    print('-'*60)
    print('k=', k)
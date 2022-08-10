import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # 全连接层的隐含单元
        self.L = 512
        self.D = 128
        self.K = 1

        #
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 36, kernel_size=4),    # （输入通道：单通道灰度图, 输出通道, kernel_size）,pad默认为0, str默认为1
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # 27x27的patch卷积后为（48，5，5）
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(48 * 5 * 5, self.L),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.feature_extractor_part3 = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # attention将提取到的特征生成K列的embedding
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        # 给包分类:给出可能性大小
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    # 最终输出对一个包的可能性预测以及根据预测给出的包标签
    def forward(self, x):
        x = x.squeeze(0)    # x好像没发生变化

        # with torch.no_grad():
        # 特征提取部分
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 48 * 5 * 5)  # 每行代表一个实例的特征
        H = self.feature_extractor_part2(H)  # NxL（一个包N个实例，一个实例L列）
        H = self.feature_extractor_part3(H)

        A = self.attention(H)  # NxK（一个L列的实例变成K列的embedding）
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N : embedding转化为权重向量αk

        M = torch.mm(A, H)  # KxN NxL -> KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()   # 包为正可能性大于0.5就认定为包标签为1

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat, Y_prob

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)   # 使得y_prob介于0和1之内，不取两端
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        # 全连接层的隐含单元
        self.L = 512
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 36, kernel_size=4),  # （输入通道：单通道灰度图, 输出通道, kernel_size）,pad默认为0, str默认为1
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, 48, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # 28x24的patch卷积后为（48，5，5）
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(48 * 5 * 5, self.L),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.feature_extractor_part3 = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 48 * 5 * 5)
        H = self.feature_extractor_part2(H)  # NxL
        H = self.feature_extractor_part3(H)

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        Y_prob, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat, Y_prob

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

import torch
import torch.nn as nn


class CbamAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=0.25):
        super(CbamAttention, self).__init__()

        assert 0 < reduction_ratio <= 1, "Reduction ratio must be between 0 and 1"
        self.reduction_ratio = reduction_ratio

        # f和g通道数采用降维方式
        reduced_channels = int(in_channels * self.reduction_ratio)

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 降维投影
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True)
        # 升维project
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=True)
        # 最后一个sigmoid不包含在有包含 Attention 的 Sequence 中
        self.sigmoid_channel = nn.Sigmoid()

        # spaial attention
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1),
            # 不需要缩小 fc 网络中的输出空间尺寸， 7*7 的卷积核已经很大了
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1),
            # 最后一个sigmoid不包含在有包含 Attention 的 Sequence 中
            nn.Sigmoid()
        )

    def forward(self, x):
        # 全局特征
        avg_pool = self.avg_pool(x)
        f_channel = self.fc1(avg_pool)
        f_channel = nn.ReLU(inplace=True)(f_channel)
        f_channel = self.fc2(f_channel)
        f_channel = self.sigmoid_channel(f_channel)  # shape: [N, C, 1, 1]
        # f_channel.mul(x) 在第 C 维上做 broadcasting，然后和 x 相乘
        x_attended_global = torch.mul(x, f_channel)

        # 空间特征
        f_spatial = self.conv_spatial(x)
        x_attended_local = torch.mul(x, f_spatial)

        # Rescale
        attended_combined = x_attended_local + x_attended_global
        return attended_combined

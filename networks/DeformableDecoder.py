import torch
import torch.nn as nn
from .deform_conv import DeformConv2d  # 假设可变卷积的实现为 DeformConv2d

class DeformableRoadDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, target_channels=None):
        """
        带有可变卷积的道路解码器
        :param in_channels: 输入特征图的通道数
        :param out_channels: 输出特征图的通道数
        :param target_channels: 目标输出通道数（用于特征合并后的调整）
        """
        super(DeformableRoadDecoder, self).__init__()

        # 可变卷积层
        self.deform_conv1 = DeformConv2d(in_channels, 256, kernel_size=3, padding=1)
        self.deform_conv2 = DeformConv2d(256, 128, kernel_size=3, padding=1)
        self.deform_conv3 = DeformConv2d(128, 64, kernel_size=3, padding=1)

        # 上采样层
        self.upsample1 = nn.ConvTranspose2d(256, out_channels[0], kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(128, out_channels[1], kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(64, out_channels[2], kernel_size=2, stride=2)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        channels=out_channels[0]+out_channels[1]+out_channels[2]
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        """
        前向传播
        :param x: 输入特征图
        :return: 输出特征图
        """
        # 可变卷积
        x1 = self.relu(self.deform_conv1(x))  # 输出形状: (batch_size, 256, H, W)
        x2 = self.relu(self.deform_conv2(x1))  # 输出形状: (batch_size, 128, H, W)
        x3 = self.relu(self.deform_conv3(x2))  # 输出形状: (batch_size, 64, H, W)

        # 上采样
        x1_up = self.upsample1(x1)  # 输出形状: (batch_size, c, H*2, W*2)
        x2_up = self.upsample2(x2)  # 输出形状: (batch_size, c, H*2, W*2)
        x3_up = self.upsample3(x3)  # 输出形状: (batch_size, out_channels, H*2, W*2)

        # 合并特征
        output = torch.cat([x1_up, x2_up, x3_up], dim=1)  # 输出形状: (batch_size, 256, H*2, W*2)
        # 特征合并后的调整
        output = self.conv(output)  # 调整通道数
        return output


# 测试代码
if __name__ == "__main__":
    upsample_out_channels=[256,64,64]
    # 输入特征图，形状为 (batch_size, in_channels, height, width)
    batch_size, in_channels, height, width = 2, 512, 16, 16
    x = torch.randn(batch_size, in_channels, height, width)
    # decoder4:  (10, 512, 16, 16) -> (10, 256, 32, 32)
    decoder4 = DeformableRoadDecoder(in_channels=512, out_channels=[128,64,64])
    # decoder3: (10, 256, 32, 32) -> (10, 128, 64, 64)
    decoder3 = DeformableRoadDecoder(in_channels=256, out_channels=[64,32,32])
    # decoder2:  (10, 128, 64, 64)->  (10, 64, 128, 128)
    decoder2 = DeformableRoadDecoder(in_channels=128, out_channels=[32,16,16])
    # decoder1:  (10, 64, 128, 128)->  (10, 64, 256, 256)
    decoder1 = DeformableRoadDecoder(in_channels=64, out_channels=[32,16,16])
    # 前向传播
    d_4= decoder4(x)#(10, 512, 16, 16) -> (10, 256, 32, 32)
    d_3= decoder3(d_4)# (10, 256, 32, 32) -> (10, 128, 64, 64)
    d_2 = decoder2(d_3)# (10, 128, 64, 64)->  (10, 64, 128, 128)
    d_1 = decoder1(d_2)# (10, 64, 128, 128)->  (10, 64, 256, 256)

    print("Output Shape:", x.shape)  # 输出形状应为 (2, 256, 32, 32)
    #模仿decoder1构造过程。请根据下面要求输出完整的decoder1、decoder2，decoder3，decoder4过程代码，
    # 现有道路编码特征，x=torch.randn(2, 512, 16, 16),decoder1输入为(2,512, 16, 16)，输出为 (2, 256, 32, 32)。
    #要求decoder2的输入形状为 (2, 256, 32, 32)，输出为（10,128,64,64)；decoder3的输入形状为 (2, 256, 32, 32)，输出为10,64,128,128
    #decoder4的输入形状为 （10,64,128,128），输出为（10,64,256,256)
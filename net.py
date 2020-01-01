import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义上采样层
class UpsampleLayer(nn.Module):

    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


# 定义DBL层
class DBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()

        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)


# 定义残差层
class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.sub_module = nn.Sequential(
            DBL(in_channels, in_channels // 2, 1, 1, 0),
            DBL(in_channels // 2, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.sub_module(x)


# 定义下采样层
class DownsamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sub_module = nn.Sequential(
            DBL(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)


# 定义主网络
class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down256 = nn.Sequential(
            DBL(3, 32, 3, 1, 1),
            DBL(32, 32, 3, 1, 1),
            ResidualLayer(32)
        )# 256 * 256
        self.downto128 = nn.Sequential(
            DBL(32, 32, 3, 1, 1),
            DBL(32, 64, 3, 1, 1),
            DownsamplingLayer(64, 64),
            ResidualLayer(64),
        )# 128 * 128
        self.downto64 = nn.Sequential(
            DBL(64, 64, 3, 1, 1),
            DBL(64, 128, 3, 1, 1),
            DownsamplingLayer(128, 128),
            ResidualLayer(128),
        )# 64 * 64
        self.downto32 = nn.Sequential(
            DBL(128, 128, 3, 1, 1),
            DBL(128, 256, 3, 1, 1),
            DownsamplingLayer(256, 256),
            ResidualLayer(256),
        )# 32 * 32
        self.downto16 = nn.Sequential(
            DBL(256, 256, 3, 1, 1),
            DBL(256, 512, 3, 1, 1),
            DownsamplingLayer(512, 512),
            ResidualLayer(512),
        )# 16 * 16
        self.downto8 = nn.Sequential(
            DBL(512, 512, 3, 1, 1),
            DBL(512, 1024, 3, 1, 1),
            DownsamplingLayer(1024, 1024),
            ResidualLayer(1024),
        )# 8 * 8


        self.upto16 = nn.Sequential(
            DBL(1024, 1024, 3, 1, 1),
            DBL(1024, 512, 3, 1, 1),
            UpsampleLayer()
        )# 16 * 16
        self.upto32 = nn.Sequential(
            DBL(1024, 512, 3, 1, 1),
            DBL(512, 256, 3, 1, 1),
            UpsampleLayer()
        )# 32 * 32
        self.upto64 = nn.Sequential(
            DBL(512, 256, 3, 1, 1),
            DBL(256, 128, 3, 1, 1),
            UpsampleLayer()
        )# 64 * 64
        self.upto128 = nn.Sequential(
            DBL(256, 128, 3, 1, 1),
            DBL(128, 64, 3, 1, 1),
            UpsampleLayer()
        )# 128 * 128
        self.upto256 = nn.Sequential(
            DBL(128, 64, 3, 1, 1),
            DBL(64, 32, 3, 1, 1),
            UpsampleLayer()
        )# 256 * 256

        self.output = DBL(64, 1, 3, 1, 1)
        self.out = nn.Sigmoid()


    def forward(self, x):
        x1 = self.down256(x)
        x2 = self.downto128(x1)
        x3 = self.downto64(x2)
        x4 = self.downto32(x3)
        x5 = self.downto16(x4)
        x6 = self.downto8(x5)
        x = self.upto16(x6)
        x = torch.cat([x, x5], dim=1)
        x = self.upto32(x)
        x = torch.cat([x, x4], dim=1)
        x = self.upto64(x)
        x = torch.cat([x, x3], dim=1)
        x = self.upto128(x)
        x = torch.cat([x, x2], dim=1)
        x = self.upto256(x)
        x = torch.cat([x, x1], dim=1)


        output = self.output(x)
        out = self.out(output)



        return out

if __name__ == '__main__':
    a = torch.Tensor([10, 3, 256, 256])
    Unet = Unet()
    print(Unet(a))
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.act = nn.LeakyReLU(0.2)

        pad = (1, 1)
        kernel_size = (3, 3)
        stride = (1, 1)
        out_channels = 3 * (upscale_factor ** 2)

        # 仅 3 层卷积，通道数小，适合 GLSL 逐像素计算
        self.conv1 = nn.Conv2d(3, 32, kernel_size, stride, pad)
        self.conv2 = nn.Conv2d(32, 32, kernel_size, stride, pad)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size, stride, pad)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        x = torch.sigmoid(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.conv3:
                    # 最后一层（PixelShuffle 前）使用较小增益，避免输出值过大
                    init.orthogonal_(m.weight, gain=0.1)
                elif m is self.conv1 or m is self.conv2:
                    init.orthogonal_(m.weight, init.calculate_gain('leaky_relu', 0.2))
                else:
                    init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

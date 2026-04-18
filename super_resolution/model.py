import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.act = nn.LeakyReLU(0.2)
        
        pad = 'same'
        kernel_size = (4, 4)
        stride = (1, 1)
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size, stride, pad)
        self.conv2 = nn.Conv2d(64, 128, kernel_size, stride, pad)
        self.conv3 = nn.Conv2d(128, 128, kernel_size, stride, pad)
        self.conv4 = nn.Conv2d(128, 64, kernel_size, stride, pad)
        self.conv5 = nn.Conv2d(64, 32, kernel_size, stride, pad)
        self.conv6 = nn.Conv2d(32, upscale_factor**2, kernel_size, stride, pad)
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m == self.conv6:
                    init.orthogonal_(m.weight, gain=0.1)
                elif m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
                    init.orthogonal_(m.weight, init.calculate_gain('leaky_relu', 0.2))
                else:
                    init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

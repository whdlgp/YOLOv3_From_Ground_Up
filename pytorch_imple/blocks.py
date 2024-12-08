import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding
                 , use_norm = True
                 , activation = nn.LeakyReLU(negative_slope=0.1)):
        super().__init__()
        # out_channel == Num of Filters
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=(not use_norm)) # No bias for this
        self.norm = nn.BatchNorm2d(out_channel) if use_norm else None
        self.activation = activation
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        # Reducing Channel, Bottleneck
        # channel shoud be more than 1
        reduced_channel = max(1, in_channel // 2)
        self.conv1 = ConvBlock(in_channel, reduced_channel, 1, 1, 0)
        self.conv2 = ConvBlock(reduced_channel, in_channel, 3, 1, 1) # use padding for same input-output shapes

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        assert x.shape == out.shape, f"Input shape not match! input {x.shape}, output {out.shape}"
        return x + out


class ResidualBlocks(nn.Module):
    def __init__(self, in_channel, num_blocks):
        super().__init__()
        # Torch not support list argument. unpack list for it
        self.layers = nn.Sequential(*[ResidualBlock(in_channel) for _ in range(num_blocks)]) 

    def forward(self, x):
        return self.layers(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # use stride 2 for reducing output resolution to half width-height
        # use padding for same input-output shapes
        # we can change output channel.
        #  ex) input chan 32, output is half of input width-height, output chan can be set 64 -> [Batch, output chan, H // 2, W // 2]
        self.conv = ConvBlock(in_channel, out_channel, 3, 2, 1)

    def forward(self, x):
        out = self.conv(x)
        assert out.shape[2] == x.shape[2] // 2 and out.shape[3] == x.shape[3] // 2, \
            f"Output shape does not match! expected {(x.shape[2] // 2, x.shape[3] // 2)} but {(out.shape[2], out.shape[3])}"
        return out


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # use scale_factor 2 for upscale output resolution to double width-height
        # we can change output channel.
        #  ex) input chan 64, output is double of input width-height, output chan can be set 32 -> [Batch, output chan, H * 2, W * 2]
        self.conv = ConvBlock(in_channel, out_channel, 1, 1, 0)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.upsample(self.conv(x))


class Conv5Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # 5 conv block For FPN implementation
        self.conv1 = ConvBlock(in_channel, out_channel, 1, 1, 0)
        self.conv2 = ConvBlock(out_channel, out_channel * 2, 3, 1, 1)
        self.conv3 = ConvBlock(out_channel * 2, out_channel, 1, 1, 0)
        self.conv4 = ConvBlock(out_channel, out_channel * 2, 3, 1, 1)
        self.conv5 = ConvBlock(out_channel * 2, out_channel, 1, 1, 0)

    def forward(self, x):
        return self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))

from blocks import *
from backbone import Darknet53
from neck import FPN
from head import YOLOHead

import torch.nn.init as init


# Init function for My YOLOv3 initial weights
def yolo_weight_init(m):
    if isinstance(m, ConvBlock):
        # Conv2d init
        if m.activation is None:  # No activation
            init.xavier_uniform_(m.conv.weight)
        elif isinstance(m.activation, nn.LeakyReLU):  # Leaky ReLU
            init.kaiming_normal_(m.conv.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.1)
        else:
            print(f"Unknown activation type: {type(m.activation)}. Using default Xavier Uniform initialization.")
            init.xavier_uniform_(m.conv.weight)
        # BatchNorm init
        if isinstance(m.norm, nn.BatchNorm2d):
            init.constant_(m.norm.weight, 1)  # gamma (scale factor)
            init.constant_(m.norm.bias, 0)  # beta (shift factor)


class YOLOv3(nn.Module):
    def __init__(self, num_class, num_anchor):
        super().__init__()

        # [Batch, 3(channel), H, W]
        self.backbone = Darknet53(3, False)

        # Each output of Darknet53 is,
        # P1: [Batch, 64, H/2, W/2]
        # P2: [Batch, 128, H/4, W/4]
        # P3: [Batch, 256, H/8, W/8]
        # P4: [Batch, 512, H/16, W/16]
        # P5: [Batch, 1024, H/32, W/32]
        # Here, We use P3, P4, P5
        self.neck = FPN(input_channels=[256, 512, 1024])
        
        # Each YOLO Head's input channels are half of input channels of Neck
        self.head_low_scale = YOLOHead(in_channel=256 // 2, num_class=num_class, num_anchor=num_anchor)
        self.head_mid_scale = YOLOHead(in_channel=512 // 2, num_class=num_class, num_anchor=num_anchor)
        self.head_high_scale = YOLOHead(in_channel=1024 // 2, num_class=num_class, num_anchor=num_anchor)

    def forward(self, x):
        p = self.backbone(x)
        x_low, x_mid, x_high = self.neck([p[2], p[3], p[4]])
        x_low = self.head_low_scale(x_low)
        x_mid = self.head_mid_scale(x_mid)
        x_high = self.head_high_scale(x_high)
        
        return x_low, x_mid, x_high


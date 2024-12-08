from blocks import *

# Darknet53 Backbone
# Output shape, Low scale(High resolution) to High scale(Low resolution) order
# [Batch, 64, H/2, W/2]
# [Batch, 128, H/4, W/4]
# [Batch, 256, H/8, W/8]
# [Batch, 512, H/16, W/16]
# [Batch, 1024, H/32, W/32]
class Darknet53(nn.Module):
    def __init__(self, in_channel = 3, classification = False, num_class = 1000):
        super().__init__()
        self.conv1 = ConvBlock(in_channel, 32, 3, 1, 1)
        self.down1 = DownsamplingBlock(32, 64)
        self.residual1 = ResidualBlocks(64, 1)
        self.down2 = DownsamplingBlock(64, 128)
        self.residual2 = ResidualBlocks(128, 2)
        self.down3 = DownsamplingBlock(128, 256)
        self.residual3 = ResidualBlocks(256, 8)
        self.down4 = DownsamplingBlock(256, 512)
        self.residual4 = ResidualBlocks(512, 8)
        self.down5 = DownsamplingBlock(512, 1024)
        self.residual5 = ResidualBlocks(1024, 4)

        # For classification mode
        # AAP with 1,1 output size means GAP. output is [Batch, Num input chan]
        self.classification = classification
        self.gap = nn.AdaptiveAvgPool2d((1,1)) if classification else None
        self.fc1 = nn.Linear(1024, num_class) if classification else None

    def forward(self, x):
        x_conv1     = self.conv1(x)
        x_down1     = self.down1(x_conv1)
        x_residual1 = self.residual1(x_down1)
        x_down2     = self.down2(x_residual1)
        x_residual2 = self.residual2(x_down2)
        x_down3     = self.down3(x_residual2)
        x_residual3 = self.residual3(x_down3)
        x_down4     = self.down4(x_residual3)
        x_residual4 = self.residual4(x_down4)
        x_down5     = self.down5(x_residual4)
        x_residual5 = self.residual5(x_down5)
        
        if self.classification:
            out = self.gap(x_residual5)
            # gap's outout is [batch, chan, 1, 1], each output chan has each input chan's avgpool
            # flatten [batch, chan, 1, 1] to [batch, chan]. 1 means second idx, [chan, 1, 1] to [chan]
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            return out
        
        # Return each scale's results. 
        return x_residual1, x_residual2, x_residual3, x_residual4, x_residual5

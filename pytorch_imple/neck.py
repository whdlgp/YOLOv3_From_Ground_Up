from blocks import *

# FeaturePyramidNetwork
# Basic implementation for 3 in-out structure
# init input channel order : Low scale(High resoultion) to High scale (Low Resolution)
class FPN_3InOut(nn.Module):
    def __init__(self, in_channel_low_scale, in_channel_mid_scale, in_channel_high_scale):
        super().__init__()
        self.conv_sequence1 = Conv5Block(in_channel_high_scale, in_channel_high_scale // 2)
        self.upsample1 = UpsamplingBlock(in_channel_high_scale // 2, in_channel_high_scale // 4)
        self.conv_sequence2 = Conv5Block(in_channel_high_scale // 4 + in_channel_mid_scale, in_channel_mid_scale // 2)
        self.upsample2 = UpsamplingBlock(in_channel_mid_scale // 2, in_channel_mid_scale // 4)
        self.conv_sequence3 = Conv5Block(in_channel_mid_scale // 4 + in_channel_low_scale, in_channel_low_scale // 2)

    # input features order : Low scale(High resoultion) to High scale (Low Resolution)
    def forward(self, features_low_scale, features_mid_scale, features_high_scale):
        x_high = self.conv_sequence1(features_high_scale)

        x_mid = torch.cat([self.upsample1(x_high), features_mid_scale], dim=1)
        x_mid = self.conv_sequence2(x_mid)
        
        x_low = torch.cat([self.upsample2(x_mid), features_low_scale], dim=1)
        x_low = self.conv_sequence3(x_low)

        return x_low, x_mid, x_high


# FeaturePyramidNetwork
# init input channel order : Low scale(High resoultion) to High scale (Low Resolution)
# Modified version of FPN_3InOut
class FPN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.conv_sequences = nn.ModuleList()
        self.upsample = nn.ModuleList()

        # Conv5 for Most High scale 
        self.conv_sequences.append(Conv5Block(input_channels[-1], input_channels[-1] // 2))

        # Upscale and Conv5
        self.num_scales = len(input_channels)
        for i in range(self.num_scales - 1, 0, -1):
            in_channel_high_scale = input_channels[i]
            in_channel_low_scale = input_channels[i-1]
            self.upsample.append(UpsamplingBlock(in_channel_high_scale // 2, in_channel_high_scale // 4))
            self.conv_sequences.append(Conv5Block(in_channel_high_scale // 4 + in_channel_low_scale, in_channel_low_scale // 2))

    # input features order : Low scale(High resoultion) to High scale (Low Resolution)
    def forward(self, features):
        assert len(features) == self.num_scales, "Number of input features not match with initialized number of scales"
        
        # Most High Scale
        output = []
        output.append(self.conv_sequences[0](features[-1]))

        # Upscale and Cat, Conv5
        count = 0
        for i in range(self.num_scales - 2, -1, -1):
            x = torch.cat([self.upsample[count](output[-1]), features[i]], dim=1)
            x = self.conv_sequences[count+1](x)
            count += 1
            output.append(x)

        # return Low scale -> High scale order
        output.reverse()
        return output

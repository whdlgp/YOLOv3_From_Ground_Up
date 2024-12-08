from blocks import *

# YOLO head
class YOLOHead(nn.Module):
    def __init__(self, in_channel, num_class, num_anchor):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channel, in_channel * 2, 3, 1, 1)
        # Output channels = num_anchors * (4(x,y,w,h) + 1(Confidence) + num_classes)
        # Final result not use Batchnorm and Linear Activation(y = x) 
        self.conv2 = ConvBlock(in_channel * 2, num_anchor * (5 + num_class), 1, 1, 0, False, None)

    def forward(self, x):
        return self.conv2(self.conv1(x))

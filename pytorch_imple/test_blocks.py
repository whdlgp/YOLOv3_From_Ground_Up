from backbone import *
from neck import *
from head import *
from yolov3 import YOLOv3
from postprocess import decode_yolov3_output, generate_dummy_data, gt_to_yolo
from loss import YoloLoss

import torch
from torchinfo import summary

def test_conv_block():
    print("\n[ConvBlock Test]")
    input_tensor = torch.randn(1, 3, 32, 32)
    
    # Test 1: With BatchNorm and Activation
    print("Test 1: With BatchNorm and Activation")
    block1 = ConvBlock(3, 16, 3, 1, 1)
    output = block1(input_tensor)
    print(block1)
    print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")

    # Test 2: Without BatchNorm
    print("\nTest 2: Without BatchNorm")
    block2 = ConvBlock(3, 16, 3, 1, 1, use_norm=False)
    output = block2(input_tensor)
    print(block2)
    print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")

    # Test 3: Without Activation
    print("\nTest 3: Without Activation")
    block3 = ConvBlock(3, 16, 3, 1, 1, activation=None)
    output = block3(input_tensor)
    print(block3)
    print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")


def test_residual_block():
    print("\n[ResidualBlock Test]")
    input_tensor = torch.randn(1, 64, 32, 32)
    block = ResidualBlock(64)
    output = block(input_tensor)
    print(block)
    print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")


def test_residual_blocks():
    print("\n[ResidualBlocks Test]")
    input_tensor = torch.randn(1, 64, 32, 32)
    blocks = ResidualBlocks(64, num_blocks=3)
    output = blocks(input_tensor)
    print(blocks)
    print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")


def test_downsampling_block():
    print("\n[DownsamplingBlock Test]")
    # Valid input test
    input_tensor = torch.randn(1, 64, 32, 32)
    down_block = DownsamplingBlock(64, 128)
    output = down_block(input_tensor)
    print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")

    # Invalid input test
    print("\nTest 2: Invalid Input")
    invalid_input = torch.randn(1, 64, 31, 31)
    try:
        down_block(invalid_input)
    except AssertionError as e:
        print(f"Error: {e}")


def test_darknet53():
    print("\n[Darknet53 Test]")
    dummy_image = torch.randn(1, 3, 416, 416)
    input_chan = dummy_image.shape[1]

    # Backbone mode
    print("Backbone Mode Test")
    darknet53_backbone = Darknet53(input_chan)
    outputs_backbone = darknet53_backbone(dummy_image)
    print(darknet53_backbone)
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shapes(For Multi-Scale):")
    for output in outputs_backbone:
        print(output.shape)

    # Classification mode
    print("\nClassification Mode Test")
    darknet53_classification = Darknet53(input_chan, classification=True, num_class=1000)
    output_classification = darknet53_classification(dummy_image)
    print(darknet53_classification)
    print(f"Input shape: {dummy_image.shape}, Output shape (Classification): {output_classification.shape}")


def test_upsample():
    # Parameters for test
    batch_size = 1
    in_channel = 64
    out_channel = 32
    height, width = 32, 32

    # Dummy input
    x = torch.randn(batch_size, in_channel, height, width)

    # Initialize UpsamplingBlock
    upsample_block = UpsamplingBlock(in_channel, out_channel)

    # Forward pass
    output = upsample_block(x)

    # Print input and output shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def test_conv5():
    # Parameters for test
    batch_size = 1
    in_channel = 384
    out_channel = 128
    height, width = 52, 52

    # Dummy input
    x = torch.randn(batch_size, in_channel, height, width)

    # Initialize Conv5Block
    conv5_block = Conv5Block(in_channel, out_channel)

    # Forward pass
    output = conv5_block(x)

    # Print input and output shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


def test_FPN_3InOut():
    # Parameters
    batch_size = 1
    low_scale_channels = 256
    mid_scale_channels = 512
    high_scale_channels = 1024

    # Feature Map resolutions
    low_scale_res = (52, 52)  # High resolution
    mid_scale_res = (26, 26)  # Medium resolution
    high_scale_res = (13, 13)  # Low resolution

    # Dummy inputs
    features_low_scale = torch.randn(batch_size, low_scale_channels, *low_scale_res)
    features_mid_scale = torch.randn(batch_size, mid_scale_channels, *mid_scale_res)
    features_high_scale = torch.randn(batch_size, high_scale_channels, *high_scale_res)

    # Initialize FPN
    fpn = FPN_3InOut(
        in_channel_low_scale=low_scale_channels,
        in_channel_mid_scale=mid_scale_channels,
        in_channel_high_scale=high_scale_channels
    )

    # Forward pass
    output_low, output_mid, output_high = fpn(features_low_scale, features_mid_scale, features_high_scale)

    # Print input and output shapes
    print(f"Input Low Scale Shape: {features_low_scale.shape}")
    print(f"Input Mid Scale Shape: {features_mid_scale.shape}")
    print(f"Input High Scale Shape: {features_high_scale.shape}")
    print(f"Output Low Scale Shape: {output_low.shape}")
    print(f"Output Mid Scale Shape: {output_mid.shape}")
    print(f"Output High Scale Shape: {output_high.shape}")


def test_FPN():
    # Parameters
    batch_size = 1
    input_channels = [256, 512, 1024]  # Low to High scale channels
    resolutions = [(52, 52), (26, 26), (13, 13)]  # High to Low resolutions (H, W)

    # Dummy inputs for FPN (Low scale to High scale)
    features = [
        torch.randn(batch_size, input_channels[i], *resolutions[i])
        for i in range(len(input_channels))
    ]

    # Initialize FPN
    fpn = FPN(input_channels)

    # Forward pass
    outputs = fpn(features)

    # Print input shapes
    for i, feature in enumerate(features):
        print(f"Input Scale {i + 1} Shape (Low to High): {feature.shape}")

    # Print output shapes
    for i, output in enumerate(outputs):
        print(f"Output Scale {i + 1} Shape (Low to High): {output.shape}")

    # Verify the number of outputs
    assert len(outputs) == len(input_channels), "Mismatch in the number of outputs"

    # Verify output shapes
    for i in range(len(outputs)):
        expected_channels = input_channels[i] // 2
        expected_height, expected_width = resolutions[i]
        assert outputs[i].shape == (batch_size, expected_channels, expected_height, expected_width), \
            f"Output shape mismatch at scale {i}: {outputs[i].shape} vs {(batch_size, expected_channels, expected_height, expected_width)}"

    print("FPN test passed successfully!")


def test_YOLOHead():
    # Parameters for test
    num_classes = 80
    batch_size = 1
    num_anchors = 3

    # Dummy feature maps for each scale
    f3 = torch.randn(batch_size, 256, 52, 52)  # Small scale (F3)
    f4 = torch.randn(batch_size, 512, 26, 26)  # Medium scale (F4)
    f5 = torch.randn(batch_size, 1024, 13, 13)  # Large scale (F5)

    # Initialize YOLOHead for each scale
    yolo_head_small = YOLOHead(f3.shape[1], num_classes, num_anchors)  # For F3
    yolo_head_medium = YOLOHead(f4.shape[1], num_classes, num_anchors)  # For F4
    yolo_head_large = YOLOHead(f5.shape[1], num_classes, num_anchors)  # For F5

    # Forward pass
    out_small = yolo_head_small(f3)
    out_medium = yolo_head_medium(f4)
    out_large = yolo_head_large(f5)

    # Print output shapes
    print(f"Small scale output: {out_small.shape}")
    print(f"Medium scale output: {out_medium.shape}")
    print(f"Large scale output: {out_large.shape}")


def test_YOLOv3():
    # Parameters
    batch_size = 1
    num_classes = 80
    num_anchors = 3
    image_size = 416

    # Initialize YOLOv3
    model = YOLOv3(num_class=num_classes, num_anchor=num_anchors)

    # Dummy input image
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    # Forward pass
    output_low, output_mid, output_high = model(dummy_input)

    # Print input and output shapes
    print(f"Input shape: {dummy_input.shape}")
    print(f"Low scale output shape: {output_low.shape}")
    print(f"Mid scale output shape: {output_mid.shape}")
    print(f"High scale output shape: {output_high.shape}")


def register_hooks(model, file):
    hooks = []

    def hook_fn(module, input, output):
        def format_tensor(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.shape
            elif isinstance(tensor, list):
                return [format_tensor(t) for t in tensor]
            elif isinstance(tensor, tuple):
                return tuple(format_tensor(t) for t in tensor)
            else:
                return str(type(tensor))  # Other types fallback

        # Write hook information to the file
        file.write(f"{module.__class__.__name__}\t{format_tensor(input)}\t{format_tensor(output)}\n")

    for layer in model.modules():
        hooks.append(layer.register_forward_hook(hook_fn))

    return hooks


def show_YOLOv3():
    # Parameters
    batch_size = 1
    num_classes = 80
    num_anchors = 3
    image_size = 416

    # Initialize YOLOv3
    model = YOLOv3(num_class=num_classes, num_anchor=num_anchors)

    # Open file for writing
    with open("yolov3_layer_info.txt", "w") as file:
        file.write("Layer Name\tInput\tOutput\n")
        hooks = register_hooks(model, file)

        # Dummy input image
        dummy_input = torch.randn(batch_size, 3, image_size, image_size)

        # Forward pass
        output_low, output_mid, output_high = model(dummy_input)

        # Remove hooks after use
        for hook in hooks:
            hook.remove()

    print("Layer information saved to yolov3_layer_info.txt")


def test_YOLOv3_postprocess():
    # Parameters
    batch_size = 1
    num_classes = 80
    num_anchors = 3
    image_size = 416

    # anchor
    anchors_small = [[10,13], [16,30], [33,23]]
    anchors_middle = [[30,61], [62,45], [59,119]]
    anchors_high = [[116,90], [156,198], [373,326]]

    # Initialize YOLOv3
    model = YOLOv3(num_class=num_classes, num_anchor=num_anchors)
    
    # Dummy input image
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    # Forward pass
    # [batch_size, num_anchors * (5 + num_classes), grid_height, grid_width]
    output_low, output_mid, output_high = model(dummy_input)
    
    # Print input shapes
    print(f"Input shape: {dummy_input.shape}")
    print()

    # Print output shapes
    print(f"Low scale output shape: {output_low.shape}")
    print(f"Mid scale output shape: {output_mid.shape}")
    print(f"High scale output shape: {output_high.shape}")
    print()
    
    # Low scale output shape: torch.Size([1, 255, 52, 52])
    # Mid scale output shape: torch.Size([1, 255, 26, 26])
    # High scale output shape: torch.Size([1, 255, 13, 13])

    # Postprocess
    # Decoded output shape:
    # [batch_size, grid_height * grid_width * num_anchors, obj data]
    #   obj data : [xmin, ymin, xmax, ymax, obj_conf, class_conf_1,,,,,class_conf_N]
    # xmin, ymin, xmax, ymax : Pixel coord BBox LT and RB point
    decoded_low = decode_yolov3_output(output_low, anchors_small, (image_size, image_size))
    decoded_middle = decode_yolov3_output(output_mid, anchors_middle, (image_size, image_size))
    decoded_high = decode_yolov3_output(output_high, anchors_high, (image_size, image_size))

    # Print decoded output shapes
    print(f"Low scale decoded shape: {decoded_low.shape}")
    print(f"Mid scale decoded shape: {decoded_middle.shape}")
    print(f"High scale decoded shape: {decoded_high.shape}")

    # Low scale decoded shape: torch.Size([1, 8112, 85])
    # Mid scale decoded shape: torch.Size([1, 2028, 85])
    # High scale decoded shape: torch.Size([1, 507, 85])


def test_GT_convert():
    # Parameters
    batch_size = 1
    num_classes = 10
    num_anchors = 3
    image_size = 416

    # anchor
    anchors_small = [[10,13], [16,30], [33,23]]
    anchors_middle = [[30,61], [62,45], [59,119]]
    anchors_high = [[116,90], [156,198], [373,326]]

    # Initialize YOLOv3
    model = YOLOv3(num_class=num_classes, num_anchor=num_anchors)
    
    # Generate dumy input and GT
    dummy_image, dummy_target = generate_dummy_data(batch_size=batch_size, image_size=(3, image_size, image_size), num_objects=3, num_classes=num_classes)

    # Forward pass
    # [batch_size, num_anchors * (5 + num_classes), grid_height, grid_width]
    output_low, output_mid, output_high = model(dummy_image)

    # Print output shapes
    print(f"Low scale output shape: {output_low.shape}")
    print(f"Mid scale output shape: {output_mid.shape}")
    print(f"High scale output shape: {output_high.shape}")
    print()

    converted_low, converted_middle, converted_high = gt_to_yolo(num_batch=batch_size,
                                                                 num_class=num_classes,
                                                                 input_size= (image_size, image_size),
                                                                 target_gt = dummy_target,
                                                                 YOLO_output=[output_low, output_mid, output_high],
                                                                 anchors = [anchors_small, anchors_middle, anchors_high])
    
    # Print decoded output shapes
    # must be 
    # [batch_size, num_anchors * (5 + num_classes), grid_height, grid_width]
    print(f"Low scale converted GT shape: {converted_low.shape}")
    print(f"Mid scale converted GT shape: {converted_middle.shape}")
    print(f"High scale converted GT shape: {converted_high.shape}")


def test_yolo_loss():
    # Parameters
    batch_size = 1
    num_classes = 10
    num_anchors = 3
    image_size = 416

    # anchor
    anchors_small = [[10,13], [16,30], [33,23]]
    anchors_middle = [[30,61], [62,45], [59,119]]
    anchors_high = [[116,90], [156,198], [373,326]]

    # Initialize YOLOv3
    model = YOLOv3(num_class=num_classes, num_anchor=num_anchors)
    
    # Generate dumy input and GT
    dummy_image, dummy_target = generate_dummy_data(batch_size=batch_size, image_size=(3, image_size, image_size), num_objects=3, num_classes=num_classes)

    # Forward pass
    # [batch_size, num_anchors * (5 + num_classes), grid_height, grid_width]
    output_low, output_mid, output_high = model(dummy_image)

    # Print output shapes
    print(f"YOLO v3 output shape")
    print(f"Low scale output shape: {output_low.shape}")
    print(f"Mid scale output shape: {output_mid.shape}")
    print(f"High scale output shape: {output_high.shape}")
    print()

    converted_low, converted_mid, converted_high = gt_to_yolo(num_batch=batch_size,
                                                              num_class=num_classes,
                                                              input_size= (image_size, image_size),
                                                              target_gt = dummy_target,
                                                              YOLO_output=[output_low, output_mid, output_high],
                                                              anchors = [anchors_small, anchors_middle, anchors_high])
    
    # Print decoded output shapes
    # must be 
    # [batch_size, num_anchors * (5 + num_classes), grid_height, grid_width]
    print(f"GT label converted shape")
    print(f"Low scale converted GT shape: {converted_low.shape}")
    print(f"Mid scale converted GT shape: {converted_mid.shape}")
    print(f"High scale converted GT shape: {converted_high.shape}")

    loss_fn = YoloLoss(num_classes)

    total_loss_low, bbox_loss_low, obj_loss_low, class_loss_low = loss_fn(output_low, converted_low)
    total_loss_mid, bbox_loss_mid, obj_loss_mid, class_loss_mid = loss_fn(output_mid, converted_mid)
    total_loss_high, bbox_loss_high, obj_loss_high, class_loss_high = loss_fn(output_high, converted_high)

    total_loss = total_loss_low + total_loss_mid + total_loss_high

    print(f"Total Loss: {total_loss}")
    print(f"Low Scale - BBox: {bbox_loss_low}, Obj: {obj_loss_low}, Class: {class_loss_low}")
    print(f"Mid Scale - BBox: {bbox_loss_mid}, Obj: {obj_loss_mid}, Class: {class_loss_mid}")
    print(f"High Scale - BBox: {bbox_loss_high}, Obj: {obj_loss_high}, Class: {class_loss_high}")



if __name__ == "__main__":
    print("YOLO v3 Implementation Testing")

    # Run all tests
    #test_conv_block()
    #test_residual_block()
    #test_residual_blocks()
    #test_downsampling_block()
    #test_darknet53()
    #test_upsample()
    #test_conv5()
    #test_FPN_3InOut()
    #test_FPN()
    #test_YOLOHead()
    #test_YOLOv3()
    #show_YOLOv3()
    #test_YOLOv3_postprocess()
    #test_GT_convert()
    test_yolo_loss()
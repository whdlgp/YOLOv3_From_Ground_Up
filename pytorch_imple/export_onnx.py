from yolov3 import YOLOv3
import torch
import onnxruntime as ort
from pathlib import Path

def export(params):
    # Output Path
    output_path = params["checkpoint_dir"] / params["onnx_file_name"]
    print(f"Start export onnx file to: {str(output_path)}")

    # Create model
    checkpoint_path = params["checkpoint_dir"] / params["src_checkpoint"]
    model = YOLOv3(params["num_class"], len(params["anchors"][0]))
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # dumy input
    # (batch_size, channels, height, width)
    dummy_input = torch.randn(1, params["input_chan"], params["input_height"], params["input_width"])

    # ONNX convert
    torch.onnx.export(
        model,                      # model
        dummy_input,                # input tensor
        str(output_path),           # save path
        export_params=True,         # include learning params
        opset_version=11,           # ONNX opset version
        input_names=['input'],      # input layer name
        output_names=['output_small_scale', 'output_medium_scale', 'output_large_scale'],  # output layer name
        dynamic_axes={              # dynamic axis
            'input': {0: 'batch_size'},
            'output_small_scale': {0: 'batch_size'},
            'output_medium_scale': {0: 'batch_size'},
            'output_large_scale': {0: 'batch_size'}
        }
    )


def test(params):
    # saved ONNX file name
    output_path = params["checkpoint_dir"] / params["onnx_file_name"]
    print(f"Start test onnx file from: {str(output_path)}")

    # ORT session
    ort_session = ort.InferenceSession(str(output_path))

    # dumy input
    # (batch_size, channels, height, width)
    dummy_input = torch.randn(1, params["input_chan"], params["input_height"], params["input_width"])
    onnx_input = {"input": dummy_input.numpy()}

    # inference
    outputs = ort_session.run(None, onnx_input)

    print("Output Small:", outputs[0].shape)
    print("Output Medium:", outputs[1].shape)
    print("Output Large:", outputs[2].shape)


if __name__ == "__main__":
    # export params
    params = {
        # Model
        "num_class": 3,
        "anchors": [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
            ],

        # Input
        "input_width": 416,
        "input_height": 416,
        "input_chan": 3,
        "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
        "std": [0.229, 0.224, 0.225],   # ImageNet normalization

        # Checkpoint paths
        "checkpoint_dir": Path("./checkpoints_yolo"),
        "src_checkpoint": "best.pth",
        "onnx_file_name": "yolov3.onnx"
    }

    export(params=params)
    test(params=params)
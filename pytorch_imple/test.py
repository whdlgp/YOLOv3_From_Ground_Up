from yolov3 import YOLOv3
from preprocess import image_to_blob
import torch
import torchvision
from torchvision.transforms.functional import resize, pad, to_tensor
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# prediction: output of YOLO, one scale
#             shape = [Batch, num_anchors * (5 + num_classes), grid_height, grid_width]
# anchor: achor boxes for prediction, corresponding prediction
def decode_yolo_out(prediction, anchor, num_class, target_width, target_height, conf_th):
    batch_size, _, grid_height, grid_width = prediction.shape
    num_anchor = len(anchor)
    # view to [Batch, num_anchors, (5 + num_classes), grid_height, grid_width]
    pred = prediction.view(batch_size, num_anchor, num_class + 5, grid_height, grid_width)
    anchor = torch.tensor(anchor).to(prediction.device).view(1, num_anchor, 2, 1, 1)

    grid_x = torch.arange(grid_width).repeat(grid_height, 1).view(1, 1, grid_height, grid_width).to(pred.device)
    grid_y = torch.arange(grid_height).repeat(grid_width, 1).t().view(1, 1, grid_height, grid_width).to(pred.device)

    tx = pred[:, :, 0, :, :]
    ty = pred[:, :, 1, :, :]
    tw = pred[:, :, 2, :, :]
    th = pred[:, :, 3, :, :]
    obj_conf = pred[:, :, 4, :, :]
    class_conf = pred[:, :, 5:, :, :]

    # pixel coord bx, by
    bx = (torch.sigmoid(tx) + grid_x) * target_width / grid_width
    by = (torch.sigmoid(ty) + grid_y) * target_height / grid_height
    # pixel coord bw, bh
    bw = torch.exp(tw) * anchor[:, :, 0, :, :]
    bh = torch.exp(th) * anchor[:, :, 1, :, :]
    x_min = bx - bw/2
    y_min = by - bh/2
    x_max = bx + bw/2
    y_max = by + bh/2
    # Objectness confidence
    obj_conf = torch.sigmoid(obj_conf)
    # Class probabilities
    class_conf = torch.sigmoid(class_conf)

    # [Batch, num_anchors, grid_height, grid_width, 4]
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    # Find high score object
    max_class_conf, max_class_idx = class_conf.max(dim=2) # 0: max val, 1: index
    score = obj_conf * max_class_conf
    mask = score > conf_th
    #mask = obj_conf > conf_th
    #mask = max_class_conf > conf_th

    print(f"Objectness Confidence mean: {obj_conf.mean().item():.4f}")
    print(f"Objectness Confidence max: {obj_conf.max().item():.4f}")
    print(f"Objectness Confidence min: {obj_conf.min().item():.4f}")
    print(f"class Confidence mean: {class_conf.mean().item():.4f}")
    print(f"class Confidence max: {class_conf.max().item():.4f}")
    print(f"class Confidence min: {class_conf.min().item():.4f}")
    
    box_masked = boxes[mask]
    score_masked = score[mask]
    idx_masked = max_class_idx[mask]

    return box_masked, score_masked, idx_masked


# rescale boxes
def rescale_boxes(boxes, input_width, input_height, origin_width, origin_height):
    scale_x = origin_width / input_width
    scale_y = origin_height / input_height

    boxes[:, 0] *= scale_x  # bx
    boxes[:, 1] *= scale_y  # by
    boxes[:, 2] *= scale_x  # bw
    boxes[:, 3] *= scale_y  # bh

    return boxes


# draw
def draw_boxes(image, boxes, scores, idxs, class_names):
    """
    Draw bounding boxes on the image using center-based coordinates.

    Args:
        image (PIL.Image.Image): Original image.
        boxes (Tensor): Decoded bounding boxes [N, 4] (bx, by, bw, bh).
        scores (Tensor): Confidence scores [N].
        class_labels (list): Class names.
        score_threshold (float): Minimum score to display a box.
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    ax = plt.gca()

    for box, score, idx in zip(boxes, scores, idxs):
        # Bounding box parameters
        x_min, y_min, x_max, y_max = box

        # Draw rectangle
        rect = patches.Rectangle(
            (x_min, y_min), x_max-x_min, y_max-y_min,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        # Add confidence score
        name = class_names[int(idx)]
        label = f"{name}: {score:.2f}"
        plt.text(
            x_min, y_min - 10, label,
            color="white", fontsize=10, bbox=dict(facecolor="red", alpha=0.5)
        )

    plt.axis("off")
    plt.show()


# Run for one image
def run(image):
    # Training params
    params = {
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

        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # Checkpoint paths
        "checkpoint_dir": Path("./checkpoints_yolo"),
    }

    # Create model
    checkpoint_path = params["checkpoint_dir"] / "best.pth"
    model = YOLOv3(params["num_class"], len(params["anchors"][0]))
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    origin_width, origin_height = image.size
    #plt.imshow(image)
    #plt.show()

    with torch.no_grad():
        blob = image_to_blob(image, params["input_width"], params["input_height"], params["mean"], params["std"])
        print(f"Blob shape: {blob.shape}")

        # Outputs from YOLO
        # P3: [Batch, num_anchors * (5 + num_classes), H/8, W/8]
        # P4: [Batch, num_anchors * (5 + num_classes), H/16, W/16]
        # P5: [Batch, num_anchors * (5 + num_classes), H/32, W/32]
        print("YOLO v3 output shape")
        yolo_outputs = model(blob)
        for scale_idx, output in enumerate(yolo_outputs):
            print(f"Scale {scale_idx}: {output.shape}")

        # Decode Outputs
        boxes = torch.zeros([0, 4])
        scores = torch.zeros([0])
        idxs = torch.zeros([0])
        for scale_idx, output in enumerate(yolo_outputs):
            box, score, idx = decode_yolo_out(output, params["anchors"][scale_idx], params["num_class"], params["input_width"], params["input_height"], 0.1)
            if len(box) > 0:
                boxes = torch.cat((boxes, box), dim=0)
                scores = torch.cat((scores, score), dim=0)
                idxs = torch.cat((idxs, idx), dim=0)
        
        # show
        # Rescale boxes to original image size
        boxes = rescale_boxes(boxes, params["input_width"], params["input_height"], origin_width, origin_height)

        # Apply Non-Maximum Suppression
        iou_threshold = 0.5  # Adjust as needed
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        idxs = idxs[keep_indices]

        # VOC Dataset Classes
        '''
        class_names = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        '''
        class_names = ["Gun", "Knife", "Pistol"]

        # Draw boxes on the image
        draw_boxes(image, boxes, scores, idxs, class_names)
        


if __name__ == "__main__":
    # Read image and show
    image_files = list(Path("testset").glob("*.jpg"))
    for image_file in image_files:
        image = Image.open(image_file)
        run(image)
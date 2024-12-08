import torch
import sys


# Postprocess of YOLOv3 output
# predictions : [batch_size, num_anchors * (5 + num_classes), grid_height, grid_width]
# anchor_boxes : [[anchor width, anchor height] ,,,, []]
# image_size : [image width, image height]
# Output
# [batch_size, grid_height * grid_width * num_anchors, obj data]
#   obj data : [xmin, ymin, xmax, ymax, obj_conf, class_conf_1,,,,,class_conf_N]
def decode_yolov3_output(predictions, anchor_boxes, image_size):
    # input shape 
    # [batch_size, num_anchors * (5 + num_classes), grid_height, grid_width]
    batch, obj_len, grid_height, grid_width = predictions.shape
    num_anchor = len(anchor_boxes)
    num_class = obj_len // num_anchor - 5

    # [batch_size, num_anchors, 5 + num_classes, grid_height, grid_width]
    pred = predictions.view(batch, num_anchor, num_class + 5, grid_height, grid_width)

    # for each grid
    results = []
    for batch_i in range(batch):
        batch_result = []
        for grid_i in range(grid_height):
            for grid_j in range(grid_width):
                for anchor_i in range(num_anchor):
                    # [5 + num_classes]
                    obj = pred[batch_i, anchor_i, :, grid_i, grid_j]
                    
                    # tx, ty, tw, th, confidence, class scores
                    tx = obj[0]
                    ty = obj[1]
                    tw = obj[2]
                    th = obj[3]
                    obj_conf = obj[4]
                    class_confs = obj[5:]

                    # bbox coord in pixel
                    # [center x, center y, bbox width, bbox height]
                    bx = (grid_j + torch.sigmoid(tx)) / grid_width * image_size[0]
                    by = (grid_i + torch.sigmoid(ty)) / grid_height * image_size[1]
                    bw = torch.exp(tw) * anchor_boxes[anchor_i][0]
                    bh = torch.exp(th) * anchor_boxes[anchor_i][1]

                    # bbox coord in [xmin, ymin, xmax, ymax]
                    xmin = bx - (bw / 2)
                    ymin = by - (bh / 2)
                    xmax = bx + (bw / 2)
                    ymax = by + (bh / 2)

                    obj_conf = torch.sigmoid(obj_conf)  
                    class_confs = torch.sigmoid(class_confs)

                    combined = torch.cat([torch.tensor([xmin, ymin, xmax, ymax, obj_conf]), class_confs])
                    batch_result.append(combined)
        results.append(torch.stack(batch_result))

    # [batch_size, grid_height * grid_width * num_anchors, obj data]
    #   obj data : [xmin, ymin, xmax, ymax, obj_conf, class_conf_1,,,,,class_conf_N]
    #   bbox coord : Pixel coord
    final_result = torch.stack(results)
    return final_result


# Test function, Dummy GT data
def generate_dummy_data(batch_size=2, image_size=(3, 224, 224), num_objects=3, num_classes=20):
    images = torch.randn(batch_size, *image_size)

    targets = []
    for _ in range(batch_size):
        # 랜덤 바운딩 박스 생성 (xmin, ymin, xmax, ymax)
        boxes = torch.randint(0, image_size[1], (num_objects, 4)).float()
        boxes[:, 2:] = boxes[:, :2] + torch.abs(boxes[:, 2:] - boxes[:, :2])  # xmax, ymax > xmin, ymin 
        
        # 이미지 크기 내로 클리핑
        boxes[:, 0].clamp_(0, image_size[2])  # xmin
        boxes[:, 1].clamp_(0, image_size[1])  # ymin
        boxes[:, 2].clamp_(0, image_size[2])  # xmax
        boxes[:, 3].clamp_(0, image_size[1])  # ymax
        
        # 랜덤 클래스 레이블 생성
        labels = torch.randint(0, num_classes, (num_objects,)).long()

        targets.append({
            "boxes": boxes,  # Shape: [num_objects, 4]
            "labels": labels  # Shape: [num_objects]
        })
    
    return images, targets


# Calculate IoU between two boxes: box1 (w1, h1), box2 (w2, h2).
def calculate_anchor_iou(box1, box2):
    inter_w = torch.min(box1[0], box2[0])
    inter_h = torch.min(box1[1], box2[1])
    inter_area = inter_w * inter_h

    union_area = box1[0] * box1[1] + box2[0] * box2[1] - inter_area
    return inter_area / (union_area + 1e-6)  # Avoid division by zero


# Find best match of anchor with GT bbox
# box: [xmin, xmax, ymin, ymax] in pixel coord, GT bbox
# anchors shape: [scales, num_anchors]
def find_best_anchor(box, anchors):
    # Compute GT box dimensions (w, h)
    gt_width = box[2] - box[0]
    gt_height = box[3] - box[1]
    gt_size = torch.tensor([gt_width, gt_height])

    best_iou = 0
    best_scale_idx = -1
    best_anchor_idx = -1
    # Find the best achor match with GT
    for scale_idx, anchor_set in enumerate(anchors):
        # Compare anchors for one scale
        for anchor_idx, anchor in enumerate(anchor_set):
            anchor_size = torch.tensor(anchor)
            iou = calculate_anchor_iou(gt_size, anchor_size)

            if iou > best_iou:
                best_iou = iou
                best_scale_idx = scale_idx
                best_anchor_idx = anchor_idx

    return best_scale_idx, best_anchor_idx, best_iou


# Convert GT to YOLO output format
# GT : PascalVOC data but converted to COCO format
#      bbox = [int(obj["bndbox"]["xmin"]),
#              int(obj["bndbox"]["ymin"]),
#              int(obj["bndbox"]["xmax"]),
#              int(obj["bndbox"]["ymax"])
#      label = self.class_to_idx[obj["name"]]  # Convert label name to index
#      {
#        "boxes": torch.tensor(bboxes, dtype=torch.float32),
#        "labels": torch.tensor(labels, dtype=torch.int64)
#      }
# num_batch: Number of batch
# num_class: Number of class
# input_size: Input size of DNN(Image) input. ex) (input_width, input_height), (416, 416)
# target_gt: above format gt input
# YOLO_output: Output from YOLO. It should be list of [batch_size, num_anchor*(num_class + 5), grid_height, grid_width]
# anchors: Pixel coord anchors, should be same with scale order of YOLO_output
# device: device to use pytorch tensor
def gt_to_yolo(num_batch, num_class, input_size, target_gt, YOLO_output, anchors, device="cpu"):
    #print(f"GT: {target_gt}")

    #for scale_idx in range(len(YOLO_output)):
    #    print(f"Scale {scale_idx} output: {YOLO_output[scale_idx].shape}")
    #    print(f"Scale {scale_idx} anchors: {anchors[scale_idx]}")

    # make zerofill memory for result
    gt_tensors = []
    for output in YOLO_output:
        _, _, grid_height, grid_width = output.shape
        zero_tensor = torch.zeros_like(output)  # YOLO_output's shape
        gt_tensors.append(zero_tensor)

    for batch_idx in range(num_batch):
        for gt_idx, (box, label) in enumerate(zip(target_gt[batch_idx]['boxes'], target_gt[batch_idx]["labels"])):
            # Find best anchor for current GT BBox
            best_scale_idx, best_anchor_idx, best_iou = find_best_anchor(box, anchors)

            # If IOU of GT BBox and anchor is too low, we ignore it 
            if best_iou < 0.5:
                continue

            #print(f"Found Best anchor for GT {gt_idx}, [{box[2] - box[0]}, {box[3] - box[1]}]: Scale {best_scale_idx}'s {anchors[best_scale_idx][best_anchor_idx]}")

            # Origin image size
            input_width, input_height = input_size
            _, _, grid_height, grid_width = YOLO_output[best_scale_idx].shape

            # YOLO Head shape
            # [batch_size, num_anchor * (num_class + 5), grid_height, grid_width]
            #   for "num_class + 5": [tx, ty, tw, th, obj_conf, class_confs], class_confs is vector of class confidense

            # Grid coord bx, by of GT BBox
            bx = ((box[2] + box[0]) / 2) / input_width * grid_width
            by = ((box[3] + box[1]) / 2) / input_height * grid_height
            # YOLO Head coord cx, cy of GT BBox
            cx = torch.floor(bx)
            cy = torch.floor(by)
            # YOLO Head coord tx, ty of GT BBox
            # clamp to prevent problem when b* - c* is 0 or 1
            tx = torch.logit(torch.clamp(bx - cx, min=1e-6, max=1-1e-6))
            ty = torch.logit(torch.clamp(by - cy, min=1e-6, max=1-1e-6))
            #tx_tmp = torch.log((bx - cx) / (1- (bx - cx) + 1e-6))
            #ty_tmp = torch.log((by - cy) / (1- (by - cy) + 1e-6))
            #tx_tmp2 = torch.log((bx - cx) + 1e-6) - torch.log(1- (bx - cx) + 1e-6)
            #ty_tmp2 = torch.log((by - cy) + 1e-6) - torch.log(1- (by - cy) + 1e-6)
            
            # Pixel coord pw, ph, anchor box width and height
            pw, ph = anchors[best_scale_idx][best_anchor_idx]
            # Pixel coord bw, bh, of GT BBox
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            # YOLO Head coord tw, th of GT BBox
            tw = torch.log(bw / pw)
            th = torch.log(bh / ph)

            # For object conf, I will use BCE. No need to use logit
            obj_conf = 1

            # For class conf, I will use BCE. No need to use logit
            # use one hot encoding 
            class_confs = torch.zeros(num_class)
            class_confs[int(label)] = 1.0

            # Now, "5 + class_num" length of data complete
            coord_and_conf = torch.tensor([tx, ty, tw, th, obj_conf])
            result_tensor = torch.cat([coord_and_conf, class_confs])

            # channel offset
            anchor_offset = best_anchor_idx * (num_class + 5)

            # Insert the result into YOLO head tensor
            try:
                gt_tensors[best_scale_idx][batch_idx,
                                            anchor_offset:anchor_offset + (num_class + 5),
                                            int(cy),
                                            int(cx)] = result_tensor
            except Exception as e:
                print(f"Error when inserting tensor: {e}")
                print(f"Best scale index: {best_scale_idx}")
                print(f"Batch index: {batch_idx}, Anchor offset: {anchor_offset}")
                print(f"Coordinates (cy, cx): ({cy}, {cx})")
                print(f"GT Tensor shape: {gt_tensors[best_scale_idx].shape}")
                print(f"Result tensor: {result_tensor}")
                sys.exit(1)
            
            #print(result_tensor)
            #print(gt_tensors[best_scale_idx][batch_idx, :, int(cy), int(cx)])

    gt_tensors = [tensor.to(device, non_blocking=True) for tensor in gt_tensors]
    return gt_tensors


if __name__ == "__main__":
    # Parameter
    batch_size = 1
    num_anchors = 3
    num_classes = 2
    grid_height = 3
    grid_width = 3
    image_size = [416, 416]  # 이미지 크기 (pixel)

    # Anchor box ([width, height])
    anchor_boxes = [[10, 13], [16, 30], [33, 23]]

    # dumy input
    predictions = torch.randn(batch_size, num_anchors * (5 + num_classes), grid_height, grid_width)

    # run postprocess
    result = decode_yolov3_output(predictions, anchor_boxes, image_size)

    # show result
    print("Final result shape:", result.shape)
    print("Sample result for first object:", result[0][0])
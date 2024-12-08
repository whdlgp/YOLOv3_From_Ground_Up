import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    # alpha: Weighting factor for positive class (default: 0.25).
    # gamma: Focusing parameter to reduce relative loss for well-classified examples (default: 2.0).
    # reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")  # BCE without reduction to handle weighting

    def forward(self, inputs, targets):
        # BCE with logits loss calculation
        bce_loss = self.bce_with_logits(inputs, targets)

        # Convert logits to probabilities
        # probs: Sigmoid probabilities from logits
        probs = torch.sigmoid(inputs)

        # Calculate p_t for focal loss
        # p_t = probs if target == 1 else 1 - probs
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate focal weight
        # focal_weight: (1 - p_t)^gamma, reduces the loss contribution of well-classified examples
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight to BCE loss
        loss = focal_weight * bce_loss

        # Apply alpha balancing factor
        # alpha_factor: Balances positive and negative sample contributions to the loss
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss *= alpha_factor

        # Apply reduction method
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class YoloLoss(nn.Module):
    def __init__(self, num_classes, lambda_obj=1.0, lambda_noobj=1.0, lambda_bbox=5.0, lambda_class=1.0):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_bbox = lambda_bbox
        self.lambda_class = lambda_class
        
        # Loss functions
        # We will use 'sum' or 'mean' later. 
        self.mse_loss = nn.MSELoss(reduction="none")
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction="none")
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")
        self.focal_loss = FocalLoss(reduction="none")
    
    # yolo_output: YOLO model output tensor, shape: [batch_size, num_anchors * (5 + num_classes), grid_height, grid_width]
    # converted_gt: Converted GT tensor in YOLO format, shape: [batch_size, num_anchors * (5 + num_classes), grid_height, grid_width]
    def forward(self, yolo_output, converted_gt):
        # Split outputs and GT into components
        # Shape: [batch_size, num_anchors * (5 + num_classes), grid_height, grid_width]
        batch_size, _, grid_height, grid_width = yolo_output.shape
        feature_size = 5 + self.num_classes
        num_anchors = yolo_output.shape[1] // feature_size
        
        # Reshape output and GT to match feature components
        # Shape: [batch_size, num_anchors, 5 + num_classes, grid_height, grid_width]
        yolo_output = yolo_output.view(batch_size, num_anchors, feature_size, grid_height, grid_width)
        converted_gt = converted_gt.view(batch_size, num_anchors, feature_size, grid_height, grid_width)
        
        # Extract components
        pred_bbox_txty = yolo_output[:, :, 0:2, :, :] # [tx, ty]
        pred_bbox_twth = yolo_output[:, :, 2:4, :, :] # [tw, th]
        pred_obj_conf = yolo_output[:, :, 4:5, :, :]  # Object confidence (logit)
        pred_class_conf = yolo_output[:, :, 5:, :, :]  # Class confidences (logits)

        # [tx, ty, tw, th]
        gt_bbox_txty = converted_gt[:, :, 0:2, :, :]
        gt_bbox_twth = converted_gt[:, :, 2:4, :, :]
        gt_obj_conf = converted_gt[:, :, 4:5, :, :]  # Object confidence
        gt_class_conf = converted_gt[:, :, 5:, :, :]  # One-hot class confidences

        # Find object
        # [batch_size, num_anchors, 1, grid_height, grid_width]
        obj = torch.abs(gt_obj_conf - 1.0) < 1e-6
        noobj = torch.abs(gt_obj_conf - 0.0) < 1e-6

        # Objectness Loss where object exist
        obj_loss = self.bce_with_logits(pred_obj_conf, gt_obj_conf)
        obj_loss *= obj
        # batch mean loss
        obj_loss = obj_loss.sum(dim=(1, 2, 3, 4))
        obj_loss = obj_loss.mean()

        # Objectness Loss where object exist
        noobj_loss = self.bce_with_logits(pred_obj_conf, gt_obj_conf)
        noobj_loss *= noobj
        # batch mean loss
        noobj_loss = noobj_loss.sum(dim=(1, 2, 3, 4))
        noobj_loss = noobj_loss.mean()

        # BBox Loss (SmoothL1)
        bbox_loss_txty = self.smooth_l1_loss(pred_bbox_txty, gt_bbox_txty)
        bbox_loss_twth = self.smooth_l1_loss(pred_bbox_twth, gt_bbox_twth)
        # Apply mask and sum
        bbox_mask = obj.expand_as(pred_bbox_txty)
        bbox_loss_txty = bbox_loss_txty * bbox_mask
        bbox_loss_twth = bbox_loss_twth * bbox_mask
        # batch mean loss
        bbox_loss_txty = bbox_loss_txty.sum(dim=(1, 2, 3, 4))
        bbox_loss_twth = bbox_loss_twth.sum(dim=(1, 2, 3, 4))
        bbox_loss = bbox_loss_txty + bbox_loss_twth
        bbox_loss = bbox_loss.mean()
        
        # Class Confidence Loss (BCE with logits output)
        #class_loss = self.bce_with_logits(pred_class_conf, gt_class_conf)
        class_loss = self.focal_loss(pred_class_conf, gt_class_conf)
        class_mask = obj.expand_as(pred_class_conf)
        class_loss = class_loss * class_mask # Apply mask and sum
        # batch mean loss
        class_loss = class_loss.sum(dim=(1, 2, 3, 4))
        class_loss = class_loss.mean()
        
        # Total Loss
        total_loss = (self.lambda_obj * obj_loss +
                      self.lambda_noobj * noobj_loss +
                      self.lambda_bbox * bbox_loss + 
                      self.lambda_class * class_loss)
        
        return total_loss, obj_loss, noobj_loss, bbox_loss, class_loss
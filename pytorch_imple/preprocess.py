import torch
from torchvision.transforms.functional import resize, pad, to_tensor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def image_to_blob(image, target_width, target_height, mean, std, use_letterbox=False):
    image = to_tensor(image) # PIL image to Torch tensor
    mean = torch.tensor(mean).contiguous().view(-1, 1, 1)
    std = torch.tensor(std).contiguous().view(-1, 1, 1)
    
    # Only support 3 or 4 dims
    num_dims = len(image.shape)
    assert num_dims == 3 or num_dims == 4, "dims of image must be 3 or 4 dims"

    # Check user confused dim order, HWC not allowed
    assert image.shape[-1] != 3, "Input tensor is likely in HWC format. Convert to CHW using to_tensor()."

    # Add dims [C, H, W] to [1, C, H, W]
    images = image
    if(num_dims == 3):
        images = image.unsqueeze(0)

    num_image, origin_chan, origin_height, origin_width = images.shape

    results = []
    for image in images:
        # images : [N, C, H, W]
        # image : [C, H, W]

        if use_letterbox:
            # Resize
            scale = min(target_width/origin_width, target_height/origin_height)
            new_width= int(origin_width * scale)
            new_height = int(origin_height * scale)
            resized_image = resize(image, [new_height, new_width])

            # Letterbox
            pad_top = (target_height - new_height) // 2
            pad_bottom = target_height - new_height - pad_top
            pad_left = (target_width - new_width) // 2
            pad_right = target_width - new_width - pad_left
            resized_image = pad(resized_image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
        else:
            resized_image = resize(image, [target_height, target_width])

        # Normalization
        norm_image = (resized_image - mean) / std

        results.append(norm_image)

    results = torch.stack(results, dim=0)

    return results


class PreProcessorLagacy:
    def __init__(self, target_width = 416, target_height = 416, target_chan = 3, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], use_letterbox=False):
        self.target_width = target_width
        self.target_height = target_height
        self.target_chan = target_chan
        self.mean = torch.tensor(mean).contiguous().view(-1, 1, 1)
        self.std = torch.tensor(std).contiguous().view(-1, 1, 1)
        self.use_letterbox = use_letterbox
    
    # Torch image or PIL image type
    def run(self, images):
        if not isinstance(images, torch.Tensor):
            images = to_tensor(images) # PIL image to Torch tensor

        # Only support 3 or 4 dims
        num_dims = len(images.shape)
        assert num_dims == 3 or num_dims == 4, "dims of image must be 3 or 4 dims"

        # Check user confused dim order, HWC not allowed
        assert images.shape[-1] != 3, "Input tensor is likely in HWC format. Convert to CHW using to_tensor()."

        # Add dims [C, H, W] to [1, C, H, W]
        if(num_dims == 3):
            images = images.unsqueeze(0)

        num_image, origin_chan, origin_height, origin_width = images.shape

        results = []
        for image in images:
            # images : [N, C, H, W]
            # image : [C, H, W]

            # Resize
            scale = min(self.target_width/origin_width, self.target_height/origin_height)
            new_width= int(origin_width * scale)
            new_height = int(origin_height * scale)
            resized_image = resize(image, [new_height, new_width])

            # Letterbox
            if self.use_letterbox:
                pad_top = (self.target_height - new_height) // 2
                pad_bottom = self.target_height - new_height - pad_top
                pad_left = (self.target_width - new_width) // 2
                pad_right = self.target_width - new_width - pad_left
                resized_image = pad(resized_image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

            # Normalization
            norm_image = (resized_image - self.mean) / self.std

            results.append(norm_image)

        results = torch.stack(results, dim=0)

        return results

def test_preprocessor_lagacy():
    dummy_image = torch.randn(3, 720, 1280)  # (C, H, W)
    preprocessor = PreProcessorLagacy()
    processed_image = preprocessor.run(dummy_image)
    print(f"Processed single image shape: {processed_image.shape}")  # [1, C, H, W]

    batch_images = torch.randn(4, 3, 720, 1280)  # (N, C, H, W)
    processed_batch = preprocessor.run(batch_images)
    print(f"Processed batch shape: {processed_batch.shape}")  # [N, C, H, W]

    dummy_image = torch.randn(720, 1280, 3)  # HWC format (incorrect)
    try:
        processed_image = preprocessor.run(dummy_image)
    except AssertionError as e:
        print(f"AssertionError: {e}")


class VOCPreProcessor:
    def __init__(self, target_width=416, target_height=416, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # Normalization
        self.target_width = target_width
        self.target_height = target_height
        self.mean = mean
        self.std = std
        
        # Albumentations
        transform_list = [
            A.LongestMaxSize(max_size=target_width),
            A.PadIfNeeded(min_height=target_height, min_width=target_width, border_mode=cv2.BORDER_CONSTANT, value=0), 
            A.BBoxSafeRandomCrop(erosion_rate=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=1.5, hue=0.1, p=0.5),
            A.SafeRotate(limit=10, p=0.5),
            #A.GaussNoise(var_limit=(10.0, 30.0), p=1),
            A.MedianBlur(blur_limit=3, p=0.2),
            A.Resize(height=target_height, width=target_width, always_apply=True),
            A.Normalize(mean=self.mean, std=self.std, always_apply=True),
            ToTensorV2()
        ]
        self.transform = A.Compose(transform_list, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.2))

    # image: HWC, PIL or Numpy
    # bboxes: [[xmin, ymin, xmax, ymax],,,]
    # labels: 
    def __call__(self, image, bboxes, labels):
        # Albumentations
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        return transformed['image'], transformed['bboxes'], transformed['labels']
    
    # Visualize the preprocessing results
    # image: Pre-processed Image (NumPy array or PIL image)
    # bboxes: Pre-Processed BBox list [[xmin, ymin, xmax, ymax], ...]
    # labels: class label list [label1, label2, ...]
    # class_names: list of class names corresponding to the labels
    def visualize(self, transformed_image, transformed_bboxes, transformed_labels, class_names):
        # De-normalize the image for visualization
        mean = np.array(self.mean)  # Convert mean list to NumPy array
        std = np.array(self.std)    # Convert std list to NumPy array
        de_transformed_image = transformed_image.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
        de_transformed_image = (de_transformed_image * std + mean) * 255  # De-normalize and scale to [0, 255]
        de_transformed_image = np.clip(de_transformed_image, 0, 255).astype(np.uint8)  # Clip and convert to uint8

        # Use Matplotlib to visualize the image and bboxes
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(de_transformed_image)

        # Draw BBoxes and labels on the image
        for bbox, label in zip(transformed_bboxes, transformed_labels):
            xmin, ymin, xmax, ymax = bbox
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                xmin, ymin - 5, class_names[int(label)], color='g', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5)
            )

        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    test_preprocessor_lagacy()
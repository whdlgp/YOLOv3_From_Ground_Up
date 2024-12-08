import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Flowers102, VOCDetection, Caltech101, CocoDetection
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from xml.etree import ElementTree as ET
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm

from preprocess import PreProcessorLagacy, VOCPreProcessor


class DatasetWrapper:
    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        processed_img = self.preprocessor.run(image)
        return processed_img.squeeze(0), label
    

class Flowers102Loader:
    def __init__(self, preprocessor, batch_size, num_workers, shuffle):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.classes = [
            "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
            "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
            "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
            "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
            "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
            "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
            "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox",
            "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
            "cape flower", "great masterwort", "siam tulip", "lenten rose", "barberton daisy",
            "daffodil", "sword lily", "poinsettia", "bolero deep blue", "wallflower",
            "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia",
            "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff",
            "gaura", "geranium", "orange dahlia", "pink-yellow dahlia?", "cautleya spicata",
            "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
            "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
            "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory",
            "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis",
            "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen",
            "watercress", "canna lily", "hippeastrum", "bee balm", "pink quill",
            "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia",
            "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"
        ]
        self.num_classes = len(self.classes)

    def get_dataloaders(self):
        train_dataset = Flowers102(root='./dataset', split="train", download=True)
        val_dataset = Flowers102(root='./dataset', split="val", download=True)
        test_dataset = Flowers102(root='./dataset', split="test", download=True)

        # Custom Dataloader
        train_loader = DataLoader(
            DatasetWrapper(train_dataset, self.preprocessor),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=True
        )
        val_loader = DataLoader(
            DatasetWrapper(val_dataset, self.preprocessor),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
        test_loader = DataLoader(
            DatasetWrapper(test_dataset, self.preprocessor),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
        
        return train_loader, val_loader, test_loader


def test_flowers102():
    # PreProcessor
    preprocessor = PreProcessorLagacy(target_width=416, target_height=416, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # DataLoader
    flowers102_loader = Flowers102Loader(preprocessor, batch_size=64, num_workers=4, shuffle=True)
    train_loader, val_loader, test_loader = flowers102_loader.get_dataloaders()

    # test show
    for images, labels in train_loader:
        print(f"Batch Images Shape: {images.shape}")  # Expected: [batch_size, C, H, W]
        print(f"Batch Labels Shape: {labels.shape}")  # Expected: [batch_size]
        break


class Caltech101Loader:
    def __init__(self, preprocessor, batch_size, num_workers, shuffle):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        # Caltech101 classes
        self.classes = Caltech101(root="./dataset", download=True).categories
        self.num_classes = len(self.classes)

    def get_dataloaders(self):
        train_dataset = Caltech101(root='./dataset', download=True, target_type='category', transform=None)

        # Split into training and validation datasets
        total_train = len(train_dataset)
        val_split = 0.1
        val_size = int(total_train * val_split)
        train_size = total_train - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        # Custom DataLoaders
        train_loader = DataLoader(
            DatasetWrapper(train_dataset, self.preprocessor),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=True
        )
        val_loader = DataLoader(
            DatasetWrapper(val_dataset, self.preprocessor),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
        return train_loader, val_loader


def test_caltech101():
    # PreProcessor
    preprocessor = PreProcessorLagacy(target_width=416, target_height=416, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # DataLoader
    caltech_loader = Caltech101Loader(preprocessor, batch_size=64, num_workers=4, shuffle=True)
    train_loader, val_loader = caltech_loader.get_dataloaders()

    # Test show
    for images, labels in train_loader:
        print(f"Batch Images Shape: {images.shape}")  # Expected: [batch_size, C, H, W]
        print(f"Batch Labels Shape: {labels.shape}")  # Expected: [batch_size]
        break


class VOCDetectionLoaderLagacy:
    def __init__(self, preprocessor, batch_size, num_workers, shuffle, input_width, input_height):
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.input_width = input_width
        self.input_height = input_height

        # VOC classes
        self.classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # Map class name to index

    def get_dataloaders(self):
        train_dataset = VOCDetection(
            root='./dataset',
            year="2007",
            image_set="train",
            download=True,
            transform=None,
            target_transform=self._target_transform
        )
        val_dataset = VOCDetection(
            root='./dataset',
            year="2007",
            image_set="val",
            download=True,
            transform=None,
            target_transform=self._target_transform
        )

        # Custom DataLoaders
        train_loader = DataLoader(
            DatasetWrapper(train_dataset, self.preprocessor),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )
        val_loader = DataLoader(
            DatasetWrapper(val_dataset, self.preprocessor),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True
        )
        return train_loader, val_loader

    def _target_transform(self, target):
        """Transform VOC target to COCO-style format."""
        bboxes = []
        labels = []
        orig_width = int(target["annotation"]["size"]["width"])
        orig_height = int(target["annotation"]["size"]["height"])
        scale_x = self.input_width / orig_width
        scale_y = self.input_height / orig_height

        for obj in target["annotation"]["object"]:
            bbox = [
                int(int(obj["bndbox"]["xmin"]) * scale_x),
                int(int(obj["bndbox"]["ymin"]) * scale_y),
                int(int(obj["bndbox"]["xmax"]) * scale_x),
                int(int(obj["bndbox"]["ymax"]) * scale_y)
            ]
            label = self.class_to_idx[obj["name"]]  # Convert label name to index
            bboxes.append(bbox)
            labels.append(label)

        return {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

    @staticmethod
    def collate_fn(batch):
        """Collate function to handle variable-sized targets."""
        images, targets = zip(*batch)
        images = torch.stack(images)  # Stack images into a single tensor
        return images, list(targets)


def test_VOC_lagacy():
    # PreProcessor
    preprocessor = PreProcessorLagacy(target_width=416, target_height=416, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # DataLoader
    voc_loader = VOCDetectionLoaderLagacy(preprocessor, batch_size=16, num_workers=4, shuffle=True)
    train_loader, val_loader = voc_loader.get_dataloaders()

    # Test show
    for images, targets in train_loader:
        print(f"Batch Images Shape: {images.shape}")  # Expected: [batch_size, C, H, W]
        print("Batch Targets:")
        
        for i, target in enumerate(targets):  # Iterate over batch
            print(f"  Image {i + 1}:")
            print(f"    Boxes: {target['boxes']}")  # Bounding boxes
            print(f"    Labels: {target['labels']}")  # Corresponding labels
        break


class VOCDetectionDataset(Dataset):
    def __init__(self, root, year, image_set, preprocessor, download):
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.preprocessor = preprocessor

        # VOC classes
        self.classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Image and Target
        image, target = self.dataset[idx]
        orig_width = int(target["annotation"]["size"]["width"])
        orig_height = int(target["annotation"]["size"]["height"])

        # BBox, Labels
        bboxes, labels = [], []
        for obj in target["annotation"]["object"]:
            xmin = int(obj["bndbox"]["xmin"])
            ymin = int(obj["bndbox"]["ymin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymax = int(obj["bndbox"]["ymax"])
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[obj["name"]])

        # Albumentations
        transformed_image, transformed_bboxes, transformed_labels = self.preprocessor(np.array(image), bboxes, labels)

        return transformed_image, {"boxes": torch.tensor(transformed_bboxes, dtype=torch.float32),
                                   "labels": torch.tensor(transformed_labels, dtype=torch.int64)}


class VOCDetectionLoader:
    def __init__(self, batch_size=16, num_workers=4, shuffle=True, target_width=416, target_height=416, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], download = True):
        # Preprocessor
        self.preprocessor = VOCPreProcessor(target_width=target_width, target_height=target_height, mean=mean, std=std)

        # Train, Validation
        self.train_dataset = VOCDetectionDataset(root='./dataset', year='2007', image_set='train', preprocessor=self.preprocessor, download=download)
        self.val_dataset = VOCDetectionDataset(root='./dataset', year='2007', image_set='val', preprocessor=self.preprocessor, download=download)
        
        # DataLoader
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=True,
                                collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True,
                                collate_fn=self.collate_fn)
        
        # Classes
        self.classes = self.train_dataset.classes
        self.num_class = len(self.train_dataset.classes)

    def get_dataloaders(self):
        return self.train_loader, self.val_loader, self.classes
    
    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, list(targets)


def test_VOC():
    # Training params
    params = {
        # Default training params
        "batch_size": 16,               # Typical batch size

        # Input
        "input_width": 416,
        "input_height": 416,
        "input_chan": 3,
        "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
        "std": [0.229, 0.224, 0.225],   # ImageNet normalization

        # DataLoader
        "num_workers": 4,
        "shuffle": True,
    }
    # Load train and validation dataloaders
    dataloader = VOCDetectionLoader(batch_size=params["batch_size"],
                                    num_workers=params["num_workers"],
                                    shuffle=params["shuffle"],
                                    target_width=params["input_width"],
                                    target_height=params["input_height"],
                                    mean=params["mean"],
                                    std=params["std"])
    train_loader, val_loader, class_names = dataloader.get_dataloaders()

    # Iterate through one batch of data
    for images, targets in train_loader:
        print(f"Images shape: {images.shape}")  # [batch_size, 3, 416, 416]
        print(f"Targets: {targets}")

        # Preprocessor
        preprocessor = train_loader.dataset.preprocessor

        # Visualize each image in the batch
        for i in range(len(images)):
            image = images[i]
            bboxes = targets[i]["boxes"].cpu().numpy().tolist()  # Convert to list
            labels = targets[i]["labels"].cpu().numpy().tolist()  # Convert to list

            preprocessor.visualize(image, bboxes, labels, class_names)

        break  # Stop after the first batch


class COCODetectionDataset(Dataset):
    def __init__(self, root, ann_file, preprocessor):
        self.dataset = CocoDetection(root=root, annFile=ann_file)
        self.preprocessor = preprocessor

        # COCO classes
        coco = COCO(ann_file)
        # Class names
        self.classes = [coco.cats[cat_id]['name'] for cat_id in sorted(coco.cats.keys())]
        # Category ID to class index
        self.cat_id_to_classes_idx = {cat_id: self.classes.index(coco.cats[cat_id]['name']) for cat_id in coco.cats.keys()}
        # class name to class idx
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Image and Target
        image, target = self.dataset[idx]

        # Extract BBox and Labels
        bboxes, labels = [], []
        for obj in target:
            bbox = obj['bbox']  # COCO format: [x_min, y_min, width, height]
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.cat_id_to_classes_idx[int(obj['category_id'])])

        # Albumentations
        transformed_image, transformed_bboxes, transformed_labels = self.preprocessor(np.array(image), bboxes, labels)

        return transformed_image, {"boxes": torch.tensor(transformed_bboxes, dtype=torch.float32),
                                   "labels": torch.tensor(transformed_labels, dtype=torch.int64)}


class COCODetectionLoader:
    def __init__(self, batch_size=16, num_workers=4, shuffle=True, target_width=416, target_height=416, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], download=True):
        if download:
            self.download_coco_dataset('./dataset/coco')

        # Preprocessor
        self.preprocessor = VOCPreProcessor(target_width=target_width, target_height=target_height, mean=mean, std=std)

        # Train, Validation
        self.train_dataset = COCODetectionDataset(root='./dataset/coco/train2017',
                                                  ann_file='./dataset/coco/annotations/instances_train2017.json',
                                                  preprocessor=self.preprocessor)
        self.val_dataset = COCODetectionDataset(root='./dataset/coco/val2017',
                                                ann_file='./dataset/coco/annotations/instances_val2017.json',
                                                preprocessor=self.preprocessor)
        
        # DataLoader
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=True,# pin_memory=True,
                                       collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True,# pin_memory=True,
                                     collate_fn=self.collate_fn)
        
        # Classes
        self.classes = self.train_dataset.classes
        self.num_class = len(self.train_dataset.classes)

    def get_dataloaders(self):
        return self.train_loader, self.val_loader, self.classes
    
    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images).pin_memory()
        return images, list(targets)
    
    @staticmethod
    def download_coco_dataset(root):
        urls = {
            "train_images": "http://images.cocodataset.org/zips/train2017.zip",
            "val_images": "http://images.cocodataset.org/zips/val2017.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        }

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        for key, url in urls.items():
            print(f"Downloading {key}...")
            zip_path = root / f"{key}.zip"
            extract_path = root if key == "annotations" else root / (key.split("_")[0] + "2017")
            extract_path.mkdir(parents=True, exist_ok=True)

            if not zip_path.exists():
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with zip_path.open('wb') as f:
                        for chunk in tqdm(r.iter_content(chunk_size=8192), desc=f"Downloading {key}"):
                            f.write(chunk)
            else:
                print(f"{zip_path} already exists. Skipping download.")

            print(f"Extracting {key}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(root)

        print("COCO dataset download and extraction completed.")


def test_COCO():
    # Training params
    params = {
        # Default training params
        "batch_size": 16,               # Typical batch size

        # Input
        "input_width": 416,
        "input_height": 416,
        "input_chan": 3,
        "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
        "std": [0.229, 0.224, 0.225],   # ImageNet normalization

        # DataLoader
        "num_workers": 4,
        "shuffle": True,
    }
    # Load train and validation dataloaders
    dataloader = COCODetectionLoader(batch_size=params["batch_size"],
                                     num_workers=params["num_workers"],
                                     shuffle=params["shuffle"],
                                     target_width=params["input_width"],
                                     target_height=params["input_height"],
                                     mean=params["mean"],
                                     std=params["std"],
                                     download=False)
    train_loader, val_loader, class_names = dataloader.get_dataloaders()

    # Iterate through one batch of data
    for images, targets in train_loader:
        print(f"Images shape: {images.shape}")  # [batch_size, 3, 416, 416]
        print(f"Targets: {targets}")

        # Preprocessor
        preprocessor = train_loader.dataset.preprocessor

        # Visualize each image in the batch
        for i in range(len(images)):
            image = images[i]
            bboxes = targets[i]["boxes"].cpu().numpy().tolist()  # Convert to list
            labels = targets[i]["labels"].cpu().numpy().tolist()  # Convert to list

            preprocessor.visualize(image, bboxes, labels, class_names)

        break  # Stop after the first batch


class RoboFlowPascalVOCDataset(Dataset):
    def __init__(self, root, split, preprocessor, classes):
        self.image_dir = Path(root) / split
        self.annotation_dir = Path(root) / split
        self.image_files = sorted(self.image_dir.glob("*.jpg"))  # Assume images are in JPG format
        self.preprocessor = preprocessor

        # Class names (can be modified if needed)
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size

        # Load annotation
        annotation_path = self.annotation_dir / f"{image_path.stem}.xml"
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Parse bounding boxes and labels
        bboxes, labels = [], []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in self.class_to_idx:
                continue
            labels.append(self.class_to_idx[class_name])

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text) - 1
            ymax = int(bndbox.find("ymax").text) - 1
            bboxes.append([xmin, ymin, xmax, ymax])

        # Apply preprocessor
        transformed_image, transformed_bboxes, transformed_labels = self.preprocessor(
            np.array(image), bboxes, labels
        )

        return transformed_image, {
            "boxes": torch.tensor(transformed_bboxes, dtype=torch.float32),
            "labels": torch.tensor(transformed_labels, dtype=torch.int64),
        }


class RoboFlowPascalVOCLoader:
    def __init__(self, root, classes, batch_size=16, num_workers=4, shuffle=True, target_width=416, target_height=416,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # Preprocessor
        self.preprocessor = VOCPreProcessor(target_width=target_width, target_height=target_height, mean=mean, std=std)

        # Datasets
        self.train_dataset = RoboFlowPascalVOCDataset(root, "train", self.preprocessor, classes)
        self.val_dataset = RoboFlowPascalVOCDataset(root, "valid", self.preprocessor, classes)

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
            collate_fn=self.collate_fn
        )

    def get_dataloaders(self):
        return self.train_loader, self.val_loader, self.train_dataset.classes

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, list(targets)


def test_RoboFlowPascalVOC():
    # Training params
    params = {
        # Default training params
        "batch_size": 16,               # Typical batch size

        # Input
        "input_width": 416,
        "input_height": 416,
        "input_chan": 3,
        "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
        "std": [0.229, 0.224, 0.225],   # ImageNet normalization

        # DataLoader
        "num_workers": 0,
        "shuffle": True,
    }

    data_dir = "./dataset/Weapon_Detection.v1i.voc"
    classes = ["Gun", "Knife", "Pistol"]
    dataloader = RoboFlowPascalVOCLoader(
        root=data_dir,
        classes=classes,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        shuffle=params["shuffle"],
        target_width=params["input_width"],
        target_height=params["input_height"],
        mean=params["mean"],
        std=params["std"]
    )
    train_loader, val_loader, class_names = dataloader.get_dataloaders()

    # Iterate through one batch of data
    for images, targets in train_loader:
        print(f"Images shape: {images.shape}")  # [batch_size, 3, 416, 416]
        print(f"Targets: {targets}")

        # Preprocessor
        preprocessor = train_loader.dataset.preprocessor

        # Visualize each image in the batch
        for i in range(len(images)):
            image = images[i]
            bboxes = targets[i]["boxes"].cpu().numpy().tolist()  # Convert to list
            labels = targets[i]["labels"].cpu().numpy().tolist()  # Convert to list

            preprocessor.visualize(image, bboxes, labels, class_names)

        break  # Stop after the first batch


if __name__ == "__main__":
    #test_flowers102()
    #test_caltech101()
    #test_VOC_lagacy()
    #test_VOC()
    #test_COCO()
    test_RoboFlowPascalVOC()
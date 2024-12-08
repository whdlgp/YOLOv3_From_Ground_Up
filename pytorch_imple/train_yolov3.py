from pathlib import Path
import natsort.natsort
from tqdm import tqdm
import natsort
from torch import nn, optim
import torch
from torch_lr_finder import LRFinder

from yolov3 import YOLOv3, yolo_weight_init
from loss import YoloLoss
from preprocess import PreProcessorLagacy
from postprocess import gt_to_yolo
from dataloaders import VOCDetectionLoaderLagacy, VOCDetectionLoader, COCODetectionLoader, RoboFlowPascalVOCLoader

class YOLOv3Trainer:
    def __init__(self, model, num_class, train_loader, val_loader, loss_fn, gt_to_yolo, params):
        self.model = model
        self.num_class = num_class
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.gt_to_yolo = gt_to_yolo

        # Training params
        self.params = params

        # Currently Total 5 Losses out.(total_loss, obj_loss, noobj_loss, bbox_loss, class_loss)
        self.NUM_LOSS_OUT = 5 

        # Optimizer: Adam
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params.get("weight_decay", 0.0001)
        )

        # Scheduler: CosineAnnealingLR
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #    self.optimizer,
        #    T_max=self.params["epoch"]
        #)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=self.params["learning_rate"]*0.1
        )

        # Device
        self.device = self.params["device"]
        self.model = self.model.to(self.device)

        # Ensure checkpoint directory exists
        self.params["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

        # Load checkpoint if exists
        last_checkpoint_path = self.params["checkpoint_dir"] / self.params["last_checkpoint"]
        if last_checkpoint_path.exists():
            self.load_checkpoint(last_checkpoint_path)
        else:
            init_checkpoint_path = self.params["checkpoint_dir"] / self.params["init_checkpoint"]
            if init_checkpoint_path.exists():
                self.load_checkpoint(init_checkpoint_path, False)
            else:
                print("No checkpoint found. Training will start from scratch.")
                print("Applying YOLO custom weight initialization")
                self.model.apply(yolo_weight_init)

        # Check NaN or Inf
        torch.autograd.set_detect_anomaly(True)

        # use AMP
        if self.params["use_amp"]:
            self.scaler = torch.cuda.amp.GradScaler()

    # Save current model to pth file
    def save_checkpoint(self, epoch, filename):
        checkpoint_path = self.params["checkpoint_dir"] / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    # Load model from pth file
    def load_checkpoint(self, checkpoint_path, resume=True):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
        if resume:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            print(f"Starting from epoch {self.start_epoch}")

    # Calc Loss of multi scale out of YOLO
    # return list of tensor, [total_loss, obj_loss, noobj_loss, bbox_loss, class_loss]
    def YOLOLoss_calc(self, num_batch, outputs, targets):
        # Convert targets to YOLO format
        converted_targets = self.gt_to_yolo(
            num_batch=num_batch,
            num_class=self.num_class,
            input_size=(self.params["input_width"], self.params["input_height"]),
            target_gt=targets,
            YOLO_output=outputs,
            anchors=self.params["anchors"],
            device=self.device
        )

        # Compute YOLO loss for each scale
        losses = [0] * self.NUM_LOSS_OUT
        for output, converted_target in zip(outputs, converted_targets):
            losses_per_scale = self.loss_fn(output, converted_target)
            losses = [x + y for x, y in zip(losses, losses_per_scale)]

        return losses

    # Backward Loss
    def backward_loss(self, loss):
        if self.params["use_amp"]:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    # Step optimizer
    def step_optimizer(self):
        # Step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        if self.params["use_amp"]:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()

    # Check GPU Memory and clear
    def check_gpu_mem(self):
        # Clear GPU Memory if needed
        mem_reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        allocated = torch.cuda.memory_allocated()
        if mem_reserved > 0 and (max_allocated / mem_reserved > 0.9 or allocated / mem_reserved > 0.9):
            torch.cuda.empty_cache()

    # Train YOLO v3 model
    def train(self):
        print("Start Training")

        checkpoint_dir = self.params["checkpoint_dir"]
        results_file = checkpoint_dir / "train_results.txt"
        valid_result = checkpoint_dir / "valid_results_per_epoch.txt"

        with results_file.open("w") as train_log:
            train_log.write("Epoch\tLoss\tObj Loss\tNo-Obj Loss\tBBox Loss\tclass Loss\n")

        with valid_result.open("w") as valid_log:
            valid_log.write("Checkpoint\tLoss\tObj Loss\tNo-Obj Loss\tBBox Loss\tclass Loss\n")

        self.start_epoch = getattr(self, "start_epoch", 0)
        for epoch in range(self.start_epoch, self.params["epoch"]):
            # Set train mode
            self.model.train()
            
            # Epoch Losses for log
            epoch_losses = [0] * self.NUM_LOSS_OUT

            # Train Loader, Train bar for visulalize progress
            train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.params['epoch']}")

            # Mini-Batch processing
            mini_batch_count = 0

            # Start mini batch
            for images, targets in train_bar:
                images = images.to(self.device, non_blocking=True)

                if torch.cuda.amp.autocast(enabled=self.params["use_amp"]):
                    outputs = self.model(images)

                    # losses from YOLOLoss = (total sum loss) / (mini batch size)
                    losses = self.YOLOLoss_calc(len(images), outputs, targets)

                    # Backword loss should be (total sum loss) / (mini batch size) / (number of mini batch)
                    batch_losses = [x / self.params["num_mini_batch"] for x in losses]
                    self.backward_loss(batch_losses[0])

                    # epoch loss for log
                    epoch_losses = [x + y.item() for x, y in zip(epoch_losses, batch_losses)]

                    # update progress bar postfix
                    train_bar.set_postfix(loss=f"{losses[0].item():.4f}")
                    
                # count mini batch
                mini_batch_count += 1

                if (mini_batch_count % self.params["num_mini_batch"]) == 0:
                    # Clear mini batch count
                    mini_batch_count = 0 

                    # Step
                    self.step_optimizer()

            # Remained mini batch results. update it
            if mini_batch_count > 0:
                # Clear mini batch count
                mini_batch_count = 0 
                
                # Step
                self.step_optimizer()

            # Step LR scheduler
            self.scheduler.step()

            # Print Epoch loss for debug
            print(f"Epoch {epoch + 1}/{self.params['epoch']}, Loss: {epoch_losses[0]:.4f}, Obj Loss: {epoch_losses[1]:.4f}, No-Obj Loss: {epoch_losses[2]:.4f}, BBox Loss: {epoch_losses[3]:.4f}, class Loss: {epoch_losses[4]:.4f}")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, f"checkpoint_epoch_{epoch + 1}.pth")
                # Do validation
                with valid_result.open("a") as valid_log: 
                    self.validate_current(valid_log, epoch+1)

            # Save checkpoint every epoch, "last.pth"
            self.save_checkpoint(epoch + 1, self.params["last_checkpoint"])

            # Log train loss log
            with results_file.open("a") as train_log:
                train_log.write(f"{epoch + 1}\t{epoch_losses[0]:.2f}\t{epoch_losses[1]:.2f}\t{epoch_losses[2]:.2f}\t{epoch_losses[3]:.2f}\t{epoch_losses[4]:.2f}\n")

            # Check GPU mem and clear
            self.check_gpu_mem()

    # Validate saved pth files
    def validate(self):
        print("Start Validation for All Checkpoints")

        checkpoint_dir = self.params["checkpoint_dir"]
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        checkpoint_files = natsort.natsort.natsorted(checkpoint_files)
        results_file = checkpoint_dir / "validation_results.txt"

        with results_file.open("w") as f:
            f.write("Checkpoint\tLoss\tObj Loss\tNo-Obj Loss\tBBox Loss\tclass Loss\n")

        for checkpoint_path in tqdm(checkpoint_files, desc="Validating Checkpoints"):
            # Load checkpoint
            self.load_checkpoint(checkpoint_path, False)

            # Set model to evaluation mode
            self.model.eval()

            # Loss for log
            batch_losses = [0] * self.NUM_LOSS_OUT

            with torch.no_grad():
                val_bar = tqdm(self.val_loader, desc=f"Validating {checkpoint_path.name}")
                for images, targets in val_bar:
                    images = images.to(self.device)

                    try:
                        outputs = self.model(images)

                        losses = self.YOLOLoss_calc(len(images), outputs, targets)
                        batch_losses = [x + y for x, y in zip(batch_losses, losses)]

                    except Exception as e:
                        print(f"Error during validation at checkpoint {checkpoint_path.name}: {e}")
                        continue

            batch_losses = [x.item() for x in batch_losses]
            print(f"Checkpoint {checkpoint_path.name}: Loss: {batch_losses[0]:.4f}, Obj Loss: {batch_losses[1]:.4f}, No-Obj Loss: {batch_losses[2]:.4f}, BBox Loss: {batch_losses[3]:.4f}, class Loss: {batch_losses[4]:.4f}")

            # Append results to file
            with results_file.open("a") as f:
                f.write(f"{checkpoint_path.name}\t{batch_losses[0]:.2f}\t{batch_losses[1]:.2f}\t{batch_losses[2]:.2f}\t{batch_losses[3]:.2f}\t{batch_losses[4]:.2f}\n")

        print(f"Validation results saved to {results_file}")

    # Validate currently loaded model
    def validate_current(self, valid_log_f, epoch):
        print("Start Validation for Current Checkpoints")

        # Set model to evaluation mode
        self.model.eval()

        # Loss for log
        batch_losses = [0] * self.NUM_LOSS_OUT

        # Start validation
        with torch.no_grad():
            val_bar = tqdm(self.val_loader, desc=f"Validating {epoch}")
            for images, targets in val_bar:
                images = images.to(self.device)

                try:
                    outputs = self.model(images)

                    losses = self.YOLOLoss_calc(len(images), outputs, targets)
                    batch_losses = [x + y for x, y in zip(batch_losses, losses)]

                except Exception as e:
                    print(f"Error during validation at checkpoint {epoch}: {e}")
                    continue
        
        # Log Loss
        batch_losses = [x.item() for x in batch_losses]
        print(f"Checkpoint {epoch}: Loss: {batch_losses[0]:.4f}, Obj Loss: {batch_losses[1]:.4f}, No-Obj Loss: {batch_losses[2]:.4f}, BBox Loss: {batch_losses[3]:.4f}, class Loss: {batch_losses[4]:.4f}")

        valid_log_f.write(f"{epoch}\t{batch_losses[0]:.2f}\t{batch_losses[1]:.2f}\t{batch_losses[2]:.2f}\t{batch_losses[3]:.2f}\t{batch_losses[4]:.2f}\n")

        # revert model to train mode
        self.model.train()

    # Find LR for first learning
    def find_lr(self):
        print("Starting Learning Rate Range Test...")

        # Wrapper loss fn for test LR
        def wrapper_loss_fn(outputs, targets):
            converted_targets = self.gt_to_yolo(
                num_batch=len(targets),
                num_class=self.num_class,
                input_size=(self.params["input_width"], self.params["input_height"]),
                target_gt=targets,
                YOLO_output=outputs,
                anchors=self.params["anchors"],
                device=self.device
                )
            total_loss = 0
            for output, converted_target in zip(outputs, converted_targets):
                loss, _, _, _, _ = self.loss_fn(output, converted_target)
                total_loss += loss
            return total_loss
        
        # optimizer for test LR
        lr_finder_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params.get("weight_decay", 0.0001)
        )

        # LR Finder
        lr_finder = LRFinder(self.model, lr_finder_optimizer, wrapper_loss_fn, device=self.device)

        try:
            lr_finder.range_test(
                self.train_loader,
                end_lr=self.params.get("end_lr", 10),
                num_iter=self.params.get("lr_find_iter", 100),
                step_mode="exp"
            )
            lr_finder.plot() # Show loss graph and best LR
        except Exception as e:
            print(f"Error during LR Finder: {e}")
        finally:
            lr_finder.reset()


if __name__ == "__main__":
    # Training params
    params = {
        # Default training params
        "batch_size": 64,              # Typical batch size
        "num_mini_batch": 4,           # Num of Mini batch  
        "epoch": 270,                  # Darknet default max_batches / iterations
        "learning_rate": 0.001,        # Initial learning rate
        "weight_decay": 0.0005,
        "num_class": 3,
        "anchors": [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
            ],
        "loss_lambdas": [1, 1, 1, 1],    # Lambda for sum loss, [obj, noobj, bbox, class]

        # Input
        "input_width": 416,
        "input_height": 416,
        "input_chan": 3,
        "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
        "std": [0.229, 0.224, 0.225],   # ImageNet normalization

        # DataLoader
        "num_workers": 8,
        "shuffle": True,
        "download": False,

        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_amp": True,

        # Checkpoint paths
        "checkpoint_dir": Path("./checkpoints_yolo"),
        "last_checkpoint": "last.pth",
        "init_checkpoint": "init.pth"
    }

    # Initialize components

    # Check Batch size and Mini batch size
    assert params["batch_size"] % params["num_mini_batch"] == 0, "Batch size must be divisible by the number of mini-batches"
    mini_batch_size = params["batch_size"] // params["num_mini_batch"]

    '''
    preprocessor = PreProcessorLagacy(target_width=params["input_width"],
                                target_height=params["input_height"],
                                target_chan=params["input_chan"],
                                mean=params["mean"],
                                std=params["std"])
    loader = VOCDetectionLoaderLagacy(preprocessor, params["batch_size"], params["num_workers"], params["shuffle"], params["input_width"], params["input_height"])
    train_loader, val_loader = loader.get_dataloaders()
    '''
    # VOC Dataset
    '''
    loader = VOCDetectionLoader(mini_batch_size,
                                num_workers=params["num_workers"],
                                shuffle=params["shuffle"],
                                target_width=params["input_width"],
                                target_height=params["input_height"],
                                mean=params["mean"],
                                std=params["std"],
                                download=params["download"])
    '''
    # COCO Dataset
    '''
    loader = COCODetectionLoader(batch_size=mini_batch_size,
                                 num_workers=params["num_workers"],
                                 shuffle=params["shuffle"],
                                 target_width=params["input_width"],
                                 target_height=params["input_height"],
                                 mean=params["mean"],
                                 std=params["std"],
                                 download=params["download"])
    '''
    # Weapon_Detection dataset from Roboflow
    # https://universe.roboflow.com/weapon-detection-swu02/weapon_detection-2ibmq
    data_dir = "./dataset/Weapon_Detection.v1i.voc"
    classes = ["Gun", "Knife", "Pistol"]
    loader = RoboFlowPascalVOCLoader(root=data_dir,
                                     classes=classes,
                                     batch_size=mini_batch_size,
                                     num_workers=params["num_workers"],
                                     shuffle=params["shuffle"],
                                     target_width=params["input_width"],
                                     target_height=params["input_height"],
                                     mean=params["mean"],
                                     std=params["std"]
    )

    train_loader, val_loader, class_names = loader.get_dataloaders()

    assert params["num_class"] == len(class_names), "Train set class num should be same with model class setting"
    num_class = params["num_class"]

    model = YOLOv3(num_class=num_class, num_anchor=len(params["anchors"][0]))
    loss_fn = YoloLoss(num_classes=num_class,
                       lambda_obj=params["loss_lambdas"][0],
                       lambda_noobj=params["loss_lambdas"][1],
                       lambda_bbox=params["loss_lambdas"][2],
                       lambda_class=params["loss_lambdas"][3],)

    trainer = YOLOv3Trainer(model, num_class, train_loader, val_loader, loss_fn, gt_to_yolo, params)

    # Train and validate
    #trainer.find_lr()
    trainer.train()
    #trainer.validate()

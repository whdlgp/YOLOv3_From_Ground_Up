from pathlib import Path
from tqdm import tqdm

# PyTorch
from torch import nn, optim
import torch

# My Custom Preprocessor and Dataloader
from preprocess import PreProcessor
from dataloaders import Caltech101Loader

# My Custom Darknet53 model
from backbone import Darknet53

class MyBackboneTrainer:
    def __init__(self):
        self.params = {
            # Default training params
            "batch_size": 16,               # Typical batch size
            "epoch": 100,                   # Darknet default max_batches / iterations
            #"epoch": 273,                   # Darknet default max_batches / iterations
            "learning_rate": 0.001,         # Initial learning rate

            # Input
            "input_width": 416,
            "input_height": 416,
            "input_chan": 3,
            "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
            "std": [0.229, 0.224, 0.225],   # ImageNet normalization

            ## SGD Optimizer
            "momentum": 0.949,              # YOLOv3-specific momentum
            "weight_decay": 0.0005,         # YOLOv3-specific weight decay
            "loss_function": nn.CrossEntropyLoss(),  # Replace if needed

            # DataLoader
            "num_workers": 4,
            "shuffle": True,

            # Learning Rate Scheduler
            "scheduler_step_size": 35,       # Reduce LR every 90 epochs
            "scheduler_gamma": 0.1,          # Multiply LR by 0.1

            # Device
            "device": "cuda" if torch.cuda.is_available() else "cpu",

            # Checkpoint paths
            "checkpoint_dir": Path("./checkpoints"),
            "last_checkpoint": "last.pth"
        }

        # Preprocessor
        self.preprocessor = PreProcessor(self.params["input_width"], self.params["input_height"], self.params["input_chan"], self.params["mean"], self.params["std"])
        # DataLoader
        self.dataloader = Caltech101Loader(self.preprocessor, self.params["batch_size"], self.params["num_workers"], self.params["shuffle"])
        self.train_loader, self.test_loader = self.dataloader.get_dataloaders()

        # Model
        self.model = Darknet53(3, True, self.dataloader.num_classes).to(self.params["device"])

        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.params["learning_rate"], momentum=self.params["momentum"], weight_decay=self.params["weight_decay"])
        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params["scheduler_step_size"], gamma=self.params["scheduler_gamma"])

        # Loss function
        self.loss_fn = self.params["loss_function"]

        # Ensure checkpoint directory exists
        self.params["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

        # Load checkpoint if exists
        last_checkpoint_path = self.params["checkpoint_dir"] / self.params["last_checkpoint"]
        if last_checkpoint_path.exists():
            self.load_checkpoint(last_checkpoint_path)
        else:
            print("No checkpoint found. Training will start from scratch.")

        # Check Nan
        torch.autograd.set_detect_anomaly(True)

    # Save checkpoint
    def save_checkpoint(self, epoch, filename):
        checkpoint_path = self.params["checkpoint_dir"] / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    # Load saved checkpoint
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.params["device"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        print(f"Loaded checkpoint from {checkpoint_path}. Starting from epoch {self.start_epoch}")

    # Training
    def train(self):
        print("Start Training")

        checkpoint_dir = self.params["checkpoint_dir"]
        results_file = checkpoint_dir / "train_results.txt"

        with results_file.open("w") as f:
            f.write("Epoch\tLoss (%)\n")
            f.write("=" * 30 + "\n")

        # Example training loop (simplified)
        self.start_epoch = getattr(self, "start_epoch", 0)
        for epoch in range(self.start_epoch, self.params["epoch"]):
            self.model.train()
            epoch_loss = 0

            # Progress bar for training
            train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.params['epoch']}")

            for images, labels in train_bar:
                images, labels = images.to(self.params["device"]), labels.to(self.params["device"])
                
                if torch.isnan(images).any() or torch.isinf(images).any():
                    print(f"NaN or Inf found in input images at epoch {epoch + 1}, batch {train_bar.n}!")
                    continue

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
                self.optimizer.step()
                epoch_loss += loss.item()

                # Update progress bar with loss
                train_bar.set_postfix(loss=f"{loss.item():.4f}")

            print(f"Epoch {epoch + 1}/{self.params['epoch']}, Loss: {epoch_loss:.4f}")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, f"checkpoint_epoch_{epoch + 1}.pth")

            # Save last checkpoint
            self.save_checkpoint(epoch + 1, self.params["last_checkpoint"])
            self.scheduler.step()

            # Save epoch accuracy to file
            with results_file.open("a") as f:
                f.write(f"{epoch + 1}\t{epoch_loss:.2f}\n")

            # Clear cache
            torch.cuda.empty_cache()

    # Test
    def test(self):
        print("Start Testing Checkpoints")

        checkpoint_dir = self.params["checkpoint_dir"]
        checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        results_file = checkpoint_dir / "test_results.txt"

        with results_file.open("w") as f:
            f.write("Checkpoint\tAccuracy (%)\n")
            f.write("=" * 30 + "\n")

        for checkpoint_path in tqdm(checkpoint_files, desc="Testing Checkpoints"):
            # Load checkpoint
            self.load_checkpoint(checkpoint_path)

            # Evaluate model on test set
            self.model.eval()
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.params["device"]), labels.to(self.params["device"])
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

            accuracy = total_correct / total_samples * 100
            print(f"Checkpoint {checkpoint_path.name}: Accuracy = {accuracy:.2f}%")

            # Append results to file
            with results_file.open("a") as f:
                f.write(f"{checkpoint_path.name}\t{accuracy:.2f}\n")

        print(f"Results saved to {results_file}")

    
if __name__ == "__main__":
    trainer = MyBackboneTrainer()

    # Train model
    trainer.train()

    # Test model
    trainer.test()
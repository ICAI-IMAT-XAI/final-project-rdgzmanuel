"""
Training pipeline for GTSRB models.
Includes training loops for baseline and pretrained models.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preprocessing import GTSRBConfig, GTSRBDataset, get_transforms, load_splits
from models import LogisticRegressionHOG, get_model


class Trainer:
    """Trainer class for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0

        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # TODO: Forward pass
            # Get predictions from model
            outputs: torch.Tensor = self.model(images)

            # TODO: Compute loss
            loss: torch.Tensor = self.criterion(outputs, labels)

            # TODO: Backward pass and optimization
            # Zero gradients, backward, optimizer step
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        avg_loss: float = total_loss / len(train_loader)
        accuracy: float = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)

                # TODO: Forward pass (no gradients needed)
                outputs: torch.Tensor = self.model(images)

                # TODO: Compute loss
                loss: torch.Tensor = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_path: Path | None = None,
    ) -> None:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_path: Path to save best model
        """
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")

    def plot_metrics(self, save_path: Path | None = None) -> None:
        """Plot training metrics."""
        _, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        axes[0].plot(self.train_losses, label="Train Loss")
        axes[0].plot(self.val_losses, label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Plot accuracy
        axes[1].plot(self.train_accuracies, label="Train Acc")
        axes[1].plot(self.val_accuracies, label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, device: str = "cuda"
) -> dict[str, float]:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device

    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images: torch.Tensor = images.to(device)
            outputs: torch.Tensor = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds: np.ndarray = np.array(all_preds)
    all_labels: np.ndarray = np.array(all_labels)

    # Compute metrics
    accuracy: float = accuracy_score(all_labels, all_preds)
    f1_macro: float = f1_score(all_labels, all_preds, average="macro")
    f1_weighted: float = f1_score(all_labels, all_preds, average="weighted")

    metrics = {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")

    return metrics


def train_logistic_regression(
    processed_dir: Path, save_path: Path
) -> LogisticRegressionHOG:
    """
    Train logistic regression on HOG features.

    Args:
        processed_dir: Directory with preprocessed data
        save_path: Path to save trained model

    Returns:
        Trained LogisticRegressionHOG model
    """
    print("Training Logistic Regression on HOG features...")

    # Load HOG features
    with open(processed_dir / "train_hog_features.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(processed_dir / "val_hog_features.pkl", "rb") as f:
        X_val = pickle.load(f)

    # Load labels
    splits = load_splits(processed_dir)
    _, y_train = splits["train"]
    _, y_val = splits["val"]

    # TODO: Initialize and train LogisticRegressionHOG
    model: LogisticRegressionHOG = LogisticRegressionHOG(
        num_classes=GTSRBConfig.NUM_CLASSES
    )
    model.train(X_train, y_train)

    # TODO: Evaluate on validation set
    y_val_pred: np.ndarray = model.predict(X_val)
    val_acc: float = accuracy_score(y_val, y_val_pred)  # Calculate accuracy
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Save model
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    return model


def train_pretrained_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    save_dir: Path,
    device: str = "cuda",
) -> None:
    """
    Train pretrained model with two-phase fine-tuning.

    Phase 1: Feature extractor mode (frozen backbone)
    Phase 2: Partial fine-tuning (unfreeze last blocks)

    Args:
        model: Pretrained model (MobileNetV2 or ResNet18)
        train_loader: Training data loader
        val_loader: Validation data loader
        model_name: Name of model ('mobilenet' or 'resnet18')
        save_dir: Directory to save checkpoints
        device: Device
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Feature extractor mode
    print("\n" + "=" * 60)
    print("PHASE 1: Feature Extractor Mode (Frozen Backbone)")
    print("=" * 60)

    # TODO: Freeze backbone
    model.freeze_backbone()

    # TODO: Create trainer with higher learning rate for phase 1
    trainer_phase1 = Trainer(
        model,
        device=device,
        learning_rate=1e-3,  # Higher LR for new classifier head
        weight_decay=1e-4,
    )

    # Train phase 1
    trainer_phase1.train(
        train_loader,
        val_loader,
        num_epochs=8,
        save_path=save_dir / f"{model_name}_phase1.pth",
    )

    # Plot phase 1 metrics
    trainer_phase1.plot_metrics(save_dir / f"{model_name}_phase1_metrics.png")

    # Phase 2: Partial fine-tuning
    print("PHASE 2: Partial Fine-tuning (Unfreeze Last Blocks)")

    # TODO: Unfreeze last blocks based on model type
    if model_name == "mobilenet":
        model.unfreeze_last_blocks(num_blocks=2)
    elif model_name == "resnet18":
        model.unfreeze_last_block()

    # TODO: Create new trainer with lower learning rate for phase 2
    trainer_phase2 = Trainer(
        model,
        device=device,
        learning_rate=1e-4,  # Lower LR for fine-tuning
        weight_decay=1e-4,
    )

    # Train phase 2
    trainer_phase2.train(
        train_loader,
        val_loader,
        num_epochs=12,
        save_path=save_dir / f"{model_name}_phase2.pth",
    )

    # Plot phase 2 metrics
    trainer_phase2.plot_metrics(save_dir / f"{model_name}_phase2_metrics.png")


def main() -> None:
    """Main training script."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create directories
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Load data splits
    splits = load_splits(GTSRBConfig.PROCESSED_DIR)

    # Create datasets and dataloaders
    train_dataset = GTSRBDataset(
        splits["train"][0], splits["train"][1], transform=get_transforms(augment=True)
    )
    val_dataset = GTSRBDataset(
        splits["val"][0], splits["val"][1], transform=get_transforms(augment=False)
    )
    test_dataset = GTSRBDataset(
        splits["test"][0], splits["test"][1], transform=get_transforms(augment=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 1. Train Logistic Regression
    print("\n" + "=" * 60)
    print("Training Logistic Regression")
    print("=" * 60)
    lr_model = train_logistic_regression(
        GTSRBConfig.PROCESSED_DIR, models_dir / "logistic_regression.pkl"
    )

    # 2. Train Shallow CNN
    print("\n" + "=" * 60)
    print("Training Shallow CNN")
    print("=" * 60)
    shallow_cnn: nn.Module = get_model("shallow_cnn", device=device)
    trainer_cnn = Trainer(shallow_cnn, device=device, learning_rate=1e-3)
    trainer_cnn.train(
        train_loader,
        val_loader,
        num_epochs=30,
        save_path=models_dir / "shallow_cnn.pth",
    )
    trainer_cnn.plot_metrics(models_dir / "shallow_cnn_metrics.png")

    # 3. Train MobileNetV2
    print("\n" + "=" * 60)
    print("Training MobileNetV2")
    print("=" * 60)
    mobilenet: nn.Module = get_model("mobilenet", pretrained=True, device=device)
    train_pretrained_model(
        mobilenet, train_loader, val_loader, "mobilenet", models_dir, device
    )

    # 4. Train ResNet18
    print("\n" + "=" * 60)
    print("Training ResNet18")
    print("=" * 60)
    resnet: nn.Module = get_model("resnet18", pretrained=True, device=device)
    train_pretrained_model(
        resnet, train_loader, val_loader, "resnet18", models_dir, device
    )

    # Evaluate all models
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # TODO: Load best models and evaluate
    # For each model, load weights and run evaluate_model()

    lr_metrics: dict[str, float] = evaluate_model(lr_model, test_loader, device=device)

    shallow_cnn.load_state_dict(
        torch.load(models_dir / "shallow_cnn.pth", map_location=device)
    )
    cnn_metrics: dict[str, float] = evaluate_model(
        shallow_cnn, test_loader, device=device
    )

    mobilenet.load_state_dict(
        torch.load(models_dir / "mobilenet_phase2.pth", map_location=device)
    )
    mobilenet_metrics: dict[str, float] = evaluate_model(
        mobilenet, test_loader, device=device
    )

    resnet.load_state_dict(
        torch.load(models_dir / "resnet18_phase2.pth", map_location=device)
    )
    resnet_metrics: dict[str, float] = evaluate_model(
        resnet, test_loader, device=device
    )

    # Summarize results
    print("\n" + "=" * 60)
    print("Summary of Test Results:")
    print(f"Logistic Regression: {lr_metrics}")
    print(f"Shallow CNN: {cnn_metrics}")
    print(f"MobileNetV2: {mobilenet_metrics}")
    print(f"ResNet18: {resnet_metrics}")

    return None


if __name__ == "__main__":
    main()

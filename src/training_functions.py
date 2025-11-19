"""
Training and evaluation functions for AKI prediction models.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("aki_prediction")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use

    Returns:
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (features, labels) in enumerate(tqdm(dataloader, desc="Training")):
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    auc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "auc": auc,
        "auprc": auprc,
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Validate model for one epoch.

    Args:
        model: PyTorch model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to use

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validating"):
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    auc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    # Classification metrics at 0.5 threshold
    pred_classes = (all_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, pred_classes)
    precision = precision_score(all_labels, pred_classes, zero_division=0)
    recall = recall_score(all_labels, pred_classes, zero_division=0)
    f1 = f1_score(all_labels, pred_classes, zero_division=0)

    return {
        "loss": avg_loss,
        "auc": auc,
        "auprc": auprc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: str = "cuda",
    early_stopping_patience: int = 10,
    save_path: str | None = None,
) -> dict[str, list[float]]:
    """
    Complete training loop with early stopping.

    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        device: Device to use
        early_stopping_patience: Epochs to wait before early stopping
        save_path: Path to save best model

    Returns:
        Dictionary with training history
    """
    logger.info("Starting training...")

    history = {
        "train_loss": [],
        "train_auc": [],
        "val_loss": [],
        "val_auc": [],
        "val_auprc": [],
    }

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_auc"].append(train_metrics["auc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_auprc"].append(val_metrics["auprc"])

        # Log metrics
        logger.info(
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train AUC: {train_metrics['auc']:.4f}"
        )
        logger.info(
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val AUC: {val_metrics['auc']:.4f}, "
            f"Val AUPRC: {val_metrics['auprc']:.4f}"
        )

        # Early stopping
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0

            # Save best model
            if save_path:
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved best model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{early_stopping_patience}"
            )

        if patience_counter >= early_stopping_patience:
            logger.info("Early stopping triggered!")
            break

    logger.info(f"\nTraining complete! Best Val AUC: {best_val_auc:.4f}")

    return history


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """
    Comprehensive model evaluation.

    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device to use

    Returns:
        Tuple of (metrics_dict, predictions, labels)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            features = features.to(device)
            outputs = model(features)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    # Calculate comprehensive metrics
    auc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)

    # Classification metrics at various thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_metrics = {}

    for thresh in thresholds:
        pred_classes = (all_preds > thresh).astype(int)
        threshold_metrics[f"threshold_{thresh}"] = {
            "accuracy": accuracy_score(all_labels, pred_classes),
            "precision": precision_score(all_labels, pred_classes, zero_division=0),
            "recall": recall_score(all_labels, pred_classes, zero_division=0),
            "f1": f1_score(all_labels, pred_classes, zero_division=0),
        }

    # Confusion matrix at 0.5 threshold
    pred_classes_05 = (all_preds > 0.5).astype(int)
    cm = confusion_matrix(all_labels, pred_classes_05)

    metrics = {
        "auc": auc,
        "auprc": auprc,
        "confusion_matrix": cm,
        "threshold_metrics": threshold_metrics,
    }

    logger.info("\n" + "=" * 60)
    logger.info("Test Set Evaluation")
    logger.info("=" * 60)
    logger.info(f"AUC-ROC: {auc:.4f}")
    logger.info(f"AUPRC: {auprc:.4f}")
    logger.info(f"\nConfusion Matrix (threshold=0.5):\n{cm}")
    logger.info("\nClassification Report (threshold=0.5):")
    logger.info(
        classification_report(
            all_labels, pred_classes_05, target_names=["No AKI", "AKI"]
        )
    )

    return metrics, all_preds, all_labels


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | None = None,
) -> None:
    """
    Plot training history.

    Args:
        history: Dictionary with training metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # AUC plot
    axes[1].plot(history["train_auc"], label="Train AUC")
    axes[1].plot(history["val_auc"], label="Val AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Training and Validation AUC")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved training plot to {save_path}")

    plt.show()


def plot_roc_curve(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str | None = None,
) -> None:
    """
    Plot ROC curve.

    Args:
        labels: True labels
        predictions: Predicted probabilities
        save_path: Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    auc = roc_auc_score(labels, predictions)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ROC curve to {save_path}")

    plt.show()


def plot_precision_recall_curve(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str | None = None,
) -> None:
    """
    Plot Precision-Recall curve.

    Args:
        labels: True labels
        predictions: Predicted probabilities
        save_path: Path to save figure
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    auprc = average_precision_score(labels, predictions)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUPRC = {auprc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PR curve to {save_path}")

    plt.show()


def get_class_weights(labels: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.

    Args:
        labels: Training labels
        device: Device to use

    Returns:
        Class weights tensor
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    # Weight inversely proportional to class frequency
    weights = total / (len(unique) * counts)

    # Convert to tensor
    weight_tensor = torch.FloatTensor([weights[0], weights[1]]).to(device)

    logger.info(f"Class weights: Negative={weights[0]:.2f}, Positive={weights[1]:.2f}")

    return weight_tensor


def save_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: str,
) -> None:
    """
    Save predictions to CSV file.

    Args:
        predictions: Model predictions
        labels: True labels
        save_path: Path to save CSV
    """
    import pandas as pd

    df = pd.DataFrame({
        "true_label": labels,
        "predicted_probability": predictions,
        "predicted_class": (predictions > 0.5).astype(int),
    })

    df.to_csv(save_path, index=False)
    logger.info(f"Saved predictions to {save_path}")

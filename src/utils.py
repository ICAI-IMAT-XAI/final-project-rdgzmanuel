"""
Utility functions for visualization and analysis.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def denormalize_image(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = (0.3403, 0.3121, 0.3214),
    std: tuple[float, float, float] = (0.2724, 0.2608, 0.2669),
) -> np.ndarray:
    """
    Denormalize an image tensor for visualization.

    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized image as numpy array (H, W, C)
    """
    img: torch.Tensor = tensor.clone()

    # Denormalize
    for t, m, s in zip(img, mean, std, strict=True):
        t.mul_(s).add_(m)

    # Convert to numpy and transpose to (H, W, C)
    img = img.numpy().transpose(1, 2, 0)

    img = np.clip(img, 0, 1)

    return img


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a heatmap on an image.

    Args:
        image: Original image (H, W, C) in range [0, 1]
        heatmap: Heatmap (H, W) in range [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap

    Returns:
        Overlayed image
    """
    # Resize heatmap to match image size
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), 
            interpolation=cv2.INTER_CUBIC)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = heatmap_colored.astype(np.float32) / 255.0

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Blend
    overlayed = alpha * heatmap_colored + (1 - alpha) * image
    overlayed = np.clip(overlayed, 0, 1)

    return overlayed


def plot_xai_comparison(
    image: np.ndarray,
    explanations: dict[str, np.ndarray],
    title: str = "XAI Method Comparison",
    save_path: Path | None = None,
) -> None:
    """
    Plot comparison of different XAI methods.

    Args:
        image: Original image
        explanations: Dictionary with method names as keys and heatmaps as values
        title: Plot title
        save_path: Optional path to save figure
    """
    num_methods = len(explanations)
    _, axes = plt.subplots(1, num_methods + 1, figsize=(4 * (num_methods + 1), 4))

    # original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # explanations
    for idx, (method_name, heatmap) in enumerate(explanations.items(), start=1):
        overlayed = overlay_heatmap(image, heatmap)
        axes[idx].imshow(overlayed)
        axes[idx].set_title(method_name)
        axes[idx].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_class_aggregated_cam(
    aggregated_heatmaps: dict[int, np.ndarray],
    class_names: dict[int, str] | None = None,
    num_classes_to_show: int = 10,
    save_path: Path | None = None,
) -> None:
    """
    Plot aggregated CAM heatmaps for multiple classes.

    Args:
        aggregated_heatmaps: Dictionary mapping class_id to aggregated heatmap
        class_names: Optional mapping of class_id to class name
        num_classes_to_show: Number of classes to display
        save_path: Optional path to save figure
    """
    classes_to_plot = list(aggregated_heatmaps.keys())[:num_classes_to_show]

    num_cols = 5
    num_rows = (len(classes_to_plot) + num_cols - 1) // num_cols

    _, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes

    for idx, class_id in enumerate(classes_to_plot):
        heatmap = aggregated_heatmaps[class_id]

        axes[idx].imshow(heatmap, cmap="jet")

        title = f"Class {class_id}"
        if class_names and class_id in class_names:
            title += f"\n{class_names[class_id]}"

        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis("off")

    # Hide unused subplots
    for idx in range(len(classes_to_plot), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Aggregated Grad-CAM by Class", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    normalize: bool = True,
    save_path: Path | None = None,
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        normalize: Whether to normalize
        save_path: Optional path to save figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names if class_names else range(len(cm)),
        yticklabels=class_names if class_names else range(len(cm)),
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def visualize_model_predictions(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    num_samples: int = 16,
    device: str = "cuda",
    class_names: dict[int, str] | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Visualize model predictions on sample images.

    Args:
        model: Trained model
        dataset: Dataset to sample from
        num_samples: Number of samples to show
        device: Device
        class_names: Optional class names
        save_path: Optional path to save figure
    """
    model.eval()

    # Sample random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    num_cols = 4
    num_rows = (num_samples + num_cols - 1) // num_cols

    _, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))
    axes = axes.flatten()

    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            image_tensor, true_label = dataset[sample_idx]

            # Get prediction
            image_batch = image_tensor.unsqueeze(0).to(device)
            output = model(image_batch)
            probabilities = torch.softmax(output, dim=1)
            pred_label = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, pred_label].item()

            # Denormalize image for visualization
            image = denormalize_image(image_tensor)

            # Plot
            axes[idx].imshow(image)

            true_name = class_names[true_label] if class_names else str(true_label)
            pred_name = class_names[pred_label] if class_names else str(pred_label)

            color = "green" if pred_label == true_label else "red"
            title = f"True: {true_name}\nPred: {pred_name} ({confidence:.2f})"

            axes[idx].set_title(title, color=color, fontsize=9)
            axes[idx].axis("off")

    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Model Predictions", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# German Traffic Sign class names (43 classes)
GTSRB_CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

"""
Model definitions for GTSRB classification.
Includes baseline models (Logistic Regression, Shallow CNN) and pretrained models.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    mobilenet_v2,
    resnet18,
)


class LogisticRegressionHOG:
    """Logistic Regression classifier for HOG features."""

    def __init__(
        self, num_classes: int = 43, max_iter: int = 1000, random_state: int = 42
    ) -> None:
        """
        Initialize Logistic Regression model.

        Args:
            num_classes: Number of output classes
            max_iter: Maximum iterations for optimization
            random_state: Random seed
        """
        self.model: LogisticRegression = LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state,
        )
        self.num_classes: int = num_classes

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the logistic regression model.

        Args:
            X_train: Training features (HOG vectors)
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for input features.

        Args:
            X: Input features

        Returns:
            Predicted class labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input features.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        return self.model.predict_proba(X)


class ShallowCNN(nn.Module):
    """
    Shallow CNN with two convolutional layers followed by a small MLP head.
    """

    def __init__(
        self,
        num_classes: int = 43,
        input_channels: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.features: nn.Module = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2),
        )

        self.pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier: nn.Module = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming initialization for conv layers and
        Xavier initialization for linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, input_channels, H, W)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class PretrainedMobileNetV2(nn.Module):
    """MobileNetV2 with custom classifier head."""

    def __init__(self, num_classes: int = 43, pretrained: bool = True) -> None:
        """
        Initialize MobileNetV2 with custom head.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to load pretrained ImageNet weights
        """
        super().__init__()

        weights: MobileNet_V2_Weights = (
            MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.model: nn.Module = mobilenet_v2(weights=weights)

        input_features: int = self.model.last_channel
        self.model.classifier[1] = nn.Linear(input_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def freeze_backbone(self) -> None:
        """Freeze all layers except classifier head."""
        for param in self.model.features.parameters():
            param.requires_grad_(False)

        for param in self.model.classifier.parameters():
            param.requires_grad_(True)

    def unfreeze_last_blocks(self, num_blocks: int = 2) -> None:
        """
        Unfreeze last N blocks for fine-tuning.

        Args:
            num_blocks: Number of inverted residual blocks to unfreeze
        """
        total_blocks: int = len(self.model.features)

        if num_blocks > total_blocks:
            num_blocks = total_blocks

        for block in self.model.features[total_blocks - num_blocks :]:
            for param in block.parameters():
                param.requires_grad_(True)


class PretrainedResNet18(nn.Module):
    """ResNet18 with custom classifier head."""

    def __init__(self, num_classes: int = 43, pretrained: bool = True) -> None:
        """
        Initialize ResNet18 with custom head.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to load pretrained ImageNet weights
        """
        super().__init__()

        weights: ResNet18_Weights = (
            ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.model: nn.Module = resnet18(weights=weights)

        input_features: int = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def freeze_backbone(self) -> None:
        """Freeze all layers except fc head."""
        for param in self.model.parameters():
            param.requires_grad_(False)

        for param in self.model.fc.parameters():
            param.requires_grad_(True)

    def unfreeze_last_block(self) -> None:
        """Unfreeze the last residual block (layer4) for fine-tuning."""
        for param in self.model.layer4.parameters():
            param.requires_grad_(True)

        for param in self.model.fc.parameters():
            param.requires_grad_(True)


def get_model(
    model_name: str,
    num_classes: int = 43,
    pretrained: bool = True,
    device: str = "cuda",
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_name: One of ['shallow_cnn', 'mobilenet', 'resnet18']
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for MobileNet/ResNet)
        device: Device to place model on

    Returns:
        Initialized model
    """
    if model_name == "shallow_cnn":
        model = ShallowCNN(num_classes=num_classes)
    elif model_name == "mobilenet":
        model = PretrainedMobileNetV2(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet18":
        model = PretrainedResNet18(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)

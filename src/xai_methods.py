"""
XAI methods for model interpretability.
Includes Grad-CAM, Integrated Gradients, and Occlusion Sensitivity.
"""

import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Initialize Grad-CAM.

        Args:
            model: The neural network model
            target_layer: The convolutional layer to extract gradients from
        """
        self.model: nn.Module = model
        self.target_layer: nn.Module = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""

        def forward_hook(
            module: nn.Module, input: torch.Tensor, output: torch.Tensor
        ) -> None:
            # TODO: Store the forward activations in self.activations
            # This captures the feature maps from the target layer
            self.activations = output.detach()

            return None

        def backward_hook(
            module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor
        ) -> None:
            # TODO: Store the gradients in self.gradients
            # This captures the gradients flowing back through the target layer
            self.gradients = grad_output[0].detach()

            return None

        # TODO: Register the hooks on self.target_layer
        # Use target_layer.register_forward_hook() and register_backward_hook()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an input image.

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W)
            target_class: Target class index (if None, use predicted class)

        Returns:
            Grad-CAM heatmap as numpy array of shape (H, W)
        """
        self.model.eval()

        # TODO: Forward pass
        # Get model output logits
        output: torch.Tensor = self.model(input_tensor)

        # TODO: Determine target class if not provided
        if target_class is None:
            target_class = int(torch.argmax(output, dim=1).item())

        # TODO: Zero gradients and compute gradients for target class
        # Use backward on output[0, target_class]
        self.model.zero_grad()
        output[0, target_class].backward()

        # TODO: Compute weights by global average pooling of gradients
        # weights shape: (num_channels,)
        # Use self.gradients and take mean over spatial dimensions
        # Compute alpha_k
        gradients: torch.Tensor = self.gradients  # (1, C, H', W')
        activations: torch.Tensor = self.activations  # (1, C, H', W')

        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations not captured. Check hooks.")

        weights: torch.Tensor = gradients.mean(dim=(2, 3))

        # TODO: Compute weighted combination of activation maps
        cam: torch.Tensor = torch.zeros(
            activations.shape[2:], device=activations.device
        )

        for k in range(weights.shape[1]):
            cam += weights[0, k] * activations[0, k]

        cam = torch.relu(cam)

        # TODO: Normalize CAM to [0, 1]
        # Subtract min, divide by max
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # TODO: Resize CAM to input image size
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
        cam = torch.nn.functional.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        cam: np.ndarray = cam.squeeze().cpu().numpy()

        return cam

    def generate_aggregated_cam(
        self,
        data_loader: torch.utils.data.DataLoader,
        class_id: int,
        num_samples: int = 100,
    ) -> np.ndarray:
        """
        Generate aggregated Grad-CAM heatmap for a specific class.

        Args:
            data_loader: DataLoader containing images
            class_id: Class to generate aggregated CAM for
            num_samples: Maximum number of samples to aggregate

        Returns:
            Aggregated heatmap averaged across samples
        """
        # TODO: Collect CAMs for num_samples images of class_id
        # Filter dataloader for images with label == class_id
        # Generate CAM for each and store in a list
        # Return the mean across all CAMs

        cams: list[np.ndarray] = []
        collected: int = 0

        for images, labels in data_loader:
            for i in range(images.size(0)):
                if labels[i].item() == class_id:
                    cam: np.ndarray = self.generate_cam(
                        images[i].unsqueeze(0), target_class=class_id
                    )
                    cams.append(cam)
                    collected += 1

                    if collected >= num_samples:
                        break

            if collected >= num_samples:
                break

        if not cams:
            raise ValueError(
                f"No samples found for class_id {class_id} in data_loader."
            )

        return np.mean(np.stack(cams, axis=0), axis=0)


class IntegratedGradients:
    """Integrated Gradients attribution method."""

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize Integrated Gradients.

        Args:
            model: The neural network model
        """
        self.model: nn.Module = model

    def generate_attribution(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        baseline: torch.Tensor | None = None,
        steps: int = 50,
    ) -> np.ndarray:
        """
        Generate Integrated Gradients attribution map.

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W)
            target_class: Target class index
            baseline: Baseline image (default: black image)
            steps: Number of integration steps

        Returns:
            Attribution map as numpy array
        """
        self.model.eval()

        # TODO: Create baseline if not provided (typically zeros)
        if baseline is None:
            baseline: torch.Tensor = torch.zeros_like(input_tensor)

        # TODO: Determine target class if not provided
        with torch.enable_grad():
            if target_class is None:
                output: torch.Tensor = self.model(input_tensor)
                target_class: int = int(torch.argmax(output, dim=1).item())

            # TODO: Generate interpolated images between baseline and input
            # Create steps+1 images: baseline + (i/steps) * (input - baseline)
            # Shape: (steps+1, C, H, W)
            alphas: torch.Tensor = torch.linspace(
                0.0, 1.0, steps + 1, device=input_tensor.device
            )

            total_gradients: torch.Tensor = torch.zeros_like(input_tensor)

            for alpha in alphas:
                interpolated: torch.Tensor = baseline + alpha * (
                    input_tensor - baseline
                )
                interpolated = interpolated.requires_grad_(True)

                output: torch.Tensor = self.model(interpolated)
                self.model.zero_grad()
                output[0, target_class].backward()

                total_gradients += interpolated.grad.detach()

            avg_gradients: torch.Tensor = total_gradients / (steps + 1)
            attribution: torch.Tensor = (input_tensor - baseline) * avg_gradients

        return attribution.squeeze(0).permute(1, 2, 0).cpu().numpy()


class OcclusionSensitivity:
    """Occlusion sensitivity analysis."""

    def __init__(
        self,
        model: nn.Module,
        occlusion_size: int = 8,
        occlusion_stride: int = 4,
        occlusion_value: float = 0.5,
    ) -> None:
        """
        Initialize Occlusion Sensitivity.

        Args:
            model: The neural network model
            occlusion_size: Size of occlusion window (square)
            occlusion_stride: Stride for sliding window
            occlusion_value: Value to fill occluded region
        """
        self.model: nn.Module = model
        self.occlusion_size: int = occlusion_size
        self.occlusion_stride: int = occlusion_stride
        self.occlusion_value: float = occlusion_value

    def generate_sensitivity_map(
        self, input_tensor: torch.Tensor, target_class: int
    ) -> np.ndarray:
        """
        Generate occlusion sensitivity map.

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W)
            target_class: Target class to measure sensitivity for

        Returns:
            Sensitivity map showing prediction change at each position
        """
        self.model.eval()

        # TODO: Get original prediction probability for target class
        with torch.no_grad():
            original_output: torch.Tensor = self.model(input_tensor)
            original_prob: float = float(
                torch.softmax(original_output, dim=1)[0, target_class].item()
            )

        # Get image dimensions
        _, _, height, width = input_tensor.shape

        # TODO: Initialize sensitivity map
        # Shape should match spatial dimensions of input
        sensitivity_map: np.ndarray = np.zeros((
            height,
            width,
        ))

        # TODO: Slide occlusion window across image
        for i in range(0, height - self.occlusion_size + 1, self.occlusion_stride):
            for j in range(0, width - self.occlusion_size + 1, self.occlusion_stride):
                occluded_image: torch.Tensor = input_tensor.clone()
                occluded_image[
                    :,
                    :,
                    i : i + self.occlusion_size,
                    j : j + self.occlusion_size,
                ].fill_(self.occlusion_value)

                with torch.no_grad():
                    occluded_output: torch.Tensor = self.model(occluded_image)
                    occluded_prob: float = float(
                        torch.softmax(occluded_output, dim=1)[0, target_class].item()
                    )

                score: float = original_prob - occluded_prob
                sensitivity_map[
                    i : i + self.occlusion_size,
                    j : j + self.occlusion_size,
                ] += score

        return sensitivity_map


class XAIEvaluator:
    """Evaluate and compare different XAI methods."""

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize XAI Evaluator.

        Args:
            model: The neural network model
        """
        self.model: nn.Module = model

    def compare_methods(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        gradcam: GradCAM,
        integrated_gradients: IntegratedGradients,
        occlusion: OcclusionSensitivity,
    ) -> dict[str, np.ndarray]:
        """
        Generate and compare multiple XAI methods on the same input.

        Args:
            input_tensor: Input image tensor
            target_class: Target class
            gradcam: GradCAM instance
            integrated_gradients: IntegratedGradients instance
            occlusion: OcclusionSensitivity instance

        Returns:
            Dictionary with method names as keys and heatmaps as values
        """
        # TODO: Generate explanations using all three methods
        # Return dictionary with keys: 'gradcam', 'integrated_gradients', 'occlusion'

        results = {}

        # Generate Grad-CAM
        results["gradcam"] = gradcam.generate_cam(input_tensor, target_class)

        # Generate Integrated Gradients
        results["integrated_gradients"] = integrated_gradients.generate_attribution(
            input_tensor, target_class
        )

        # Generate Occlusion Sensitivity
        results["occlusion"] = occlusion.generate_sensitivity_map(
            input_tensor, target_class
        )

        return results

    def top_k_overlap(
        self, map1: np.ndarray, map2: np.ndarray, k_percent: float = 0.1
    ) -> float:
        """
        Compute overlap of top-k% most important pixels.

        Args:
            map1: First attribution map
            map2: Second attribution map
            k_percent: Percentage of top pixels to consider

        Returns:
            Overlap ratio (Jaccard index)
        """
        # TODO: For each map, identify top k% pixels
        # Compute intersection and union
        # Return intersection / union (Jaccard index)
        num_pixels: int = map1.size
        k: int = max(1, int(num_pixels * k_percent))

        topk_indices1: np.ndarray = np.argsort(map1.flatten())[-k:]
        topk_indices2: np.ndarray = np.argsort(map2.flatten())[-k:]

        set1: set = set(topk_indices1)
        set2: set = set(topk_indices2)

        intersection: int = len(set1.intersection(set2))
        union: int = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the appropriate target layer for Grad-CAM.

    Args:
        model: The model
        model_name: Name of the model architecture

    Returns:
        Target convolutional layer
    """
    # TODO: Return appropriate layer based on model_name
    # For 'shallow_cnn': return model.conv2
    # For 'mobilenet': return model.model.features[-1]
    # For 'resnet18': return model.model.layer4[-1]

    if model_name == "shallow_cnn":
        return model.conv2
    elif model_name == "mobilenet":
        return model.model.features[-1]
    elif model_name == "resnet18":
        return model.model.layer4[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

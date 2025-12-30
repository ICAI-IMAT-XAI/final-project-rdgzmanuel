"""
GTSRB Data Preprocessing Pipeline
Handles data loading, preprocessing, HOG feature extraction, and train/val/test splits.
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class GTSRBConfig:
    """Configuration for GTSRB dataset preprocessing."""

    # Paths
    DATA_DIR: Path = Path("data/raw")
    PROCESSED_DIR: Path = Path("data/processed")

    # Image preprocessing
    IMG_SIZE: tuple[int, int] = (64, 64)
    NORMALIZE_MEAN: tuple[float, float, float] = (0.3403, 0.3121, 0.3214)
    NORMALIZE_STD: tuple[float, float, float] = (0.2724, 0.2608, 0.2669)

    # HOG parameters
    HOG_ORIENTATIONS: int = 9
    HOG_PIXELS_PER_CELL: tuple[int, int] = (8, 8)
    HOG_CELLS_PER_BLOCK: tuple[int, int] = (2, 2)

    # Dataset splits
    VAL_SIZE: float = 0.15
    TEST_SIZE: float = 0.15
    RANDOM_STATE: int = 42

    # Number of classes
    NUM_CLASSES: int = 43


class GTSRBDataset(Dataset):
    """PyTorch Dataset for GTSRB images."""

    def __init__(
        self,
        image_paths: np.ndarray,
        labels: np.ndarray,
        transform: transforms.Compose | None = None,
    ) -> None:
        """
        Args:
            image_paths: Array of paths to images
            labels: Array of corresponding labels
            transform: Optional torchvision transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path: str = str(self.image_paths[idx]).strip()
        label: int = int(self.labels[idx])

        image: torch.Tensor = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(augment: bool = False) -> transforms.Compose:
    """
    Get image transformations for training or validation/test.

    Args:
        augment: Whether to apply data augmentation

    Returns:
        Composed transforms
    """
    if augment:
        return transforms.Compose([
            transforms.Resize(GTSRBConfig.IMG_SIZE),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=GTSRBConfig.NORMALIZE_MEAN, std=GTSRBConfig.NORMALIZE_STD
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(GTSRBConfig.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=GTSRBConfig.NORMALIZE_MEAN, std=GTSRBConfig.NORMALIZE_STD
            ),
        ])


def load_gtsrb_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load GTSRB dataset from directory structure."""
    image_paths: list[str] = []
    labels: list[int] = []

    # Convert to absolute path at the start
    data_dir = data_dir.absolute()
    train_dir: Path = data_dir / "Train"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found at {train_dir}. "
            "Please extract the GTSRB dataset to data/raw/"
        )

    print("Loading training data...")
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]

    for class_dir in tqdm(sorted(class_dirs)):
        class_id = int(class_dir.name)
        for img_file in class_dir.glob("*.png"):
            image_paths.append(str(img_file.absolute()))  # Make absolute
            labels.append(class_id)

    # Load test data if available
    test_csv: Path = data_dir / "Test.csv"
    if test_csv.exists():
        print("Loading test data...")
        test_df = pd.read_csv(test_csv)

        print("\nTest.csv sample paths:")
        print(test_df["Path"].head())

        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            path_str = str(row["Path"])

            if path_str.startswith("Test/") or "/" in path_str or "\\" in path_str:
                img_path = data_dir / path_str
            else:
                img_path = data_dir / "Test" / path_str

            if img_path.exists():
                image_paths.append(str(img_path.absolute()))  # Make absolute
                labels.append(int(row["ClassId"]))
            else:
                print(f"Warning: Image not found at {img_path}")
                alt_path = data_dir / "Test" / Path(path_str).name
                if alt_path.exists():
                    image_paths.append(str(alt_path.absolute()))  # Make absolute
                    labels.append(int(row["ClassId"]))
                    print(f"  Found at alternative path: {alt_path}")

    image_paths: np.ndarray = np.array(image_paths)
    labels: np.ndarray = np.array(labels, dtype=np.int32)

    print(f"\nLoaded {len(image_paths)} images across {len(np.unique(labels))} classes")
    print(f"Class distribution:\n{np.bincount(labels)}")

    return image_paths, labels


def extract_hog_features(image_path: str) -> np.ndarray:
    """
    Extract HOG features from an image.

    Args:
        image_path: Path to the image

    Returns:
        HOG feature vector
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to standard size
    img = cv2.resize(img, GTSRBConfig.IMG_SIZE)

    # Extract HOG features
    features = hog(
        img,
        orientations=GTSRBConfig.HOG_ORIENTATIONS,
        pixels_per_cell=GTSRBConfig.HOG_PIXELS_PER_CELL,
        cells_per_block=GTSRBConfig.HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )

    return features


def extract_all_hog_features(
    image_paths: np.ndarray, save_path: Path | None = None
) -> np.ndarray:
    """
    Extract HOG features for all images.

    Args:
        image_paths: Array of image paths
        save_path: Optional path to save features

    Returns:
        Array of HOG features
    """
    print(f"Extracting HOG features for {len(image_paths)} images...")

    features_list = []
    for img_path in tqdm(image_paths):
        features = extract_hog_features(img_path)
        features_list.append(features)

    features_array = np.array(features_list)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(features_array, f)
        print(f"HOG features saved to {save_path}")

    print(f"HOG feature shape: {features_array.shape}")
    return features_array


def create_data_splits(
    image_paths: np.ndarray, labels: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Create train/validation/test splits with stratification.

    Args:
        image_paths: Array of image paths
        labels: Array of labels

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing (paths, labels) tuples
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths,
        labels,
        test_size=GTSRBConfig.TEST_SIZE,
        stratify=labels,
        random_state=GTSRBConfig.RANDOM_STATE,
    )

    # Second split: separate validation from training
    val_size_adjusted = GTSRBConfig.VAL_SIZE / (1 - GTSRBConfig.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=GTSRBConfig.RANDOM_STATE,
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    print("\nDataset splits:")
    for split_name, (paths, lbls) in splits.items():
        print(
            f"{split_name}: {len(paths)} samples ({len(paths) / len(image_paths) * 100:.1f}%)"
        )
        print(f"  Class distribution: {np.bincount(lbls)}")

    return splits


def save_splits(
    splits: dict[str, tuple[np.ndarray, np.ndarray]], output_dir: Path
) -> None:
    """
    Save data splits to disk.

    Args:
        splits: Dictionary with train/val/test splits
        output_dir: Directory to save splits
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, (paths, labels) in splits.items():
        split_data = {"image_paths": paths, "labels": labels}

        save_path = output_dir / f"{split_name}_split.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(split_data, f)

        print(f"Saved {split_name} split to {save_path}")


def load_splits(processed_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load preprocessed data splits.

    Args:
        processed_dir: Directory containing split files

    Returns:
        Dictionary with train/val/test splits
    """
    splits = {}

    for split_name in ["train", "val", "test"]:
        split_path = processed_dir / f"{split_name}_split.pkl"

        with open(split_path, "rb") as f:
            split_data = pickle.load(f)

        splits[split_name] = (split_data["image_paths"], split_data["labels"])

    return splits


def compute_dataset_statistics(
    image_paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for dataset normalization.

    Args:
        image_paths: Array of image paths (typically from training set)

    Returns:
        Tuple of (mean, std) for RGB channels
    """
    print("Computing dataset statistics for normalization...")

    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    pixel_count = 0

    # Sample subset for efficiency (use first 5000 images)
    sample_paths = image_paths[: min(5000, len(image_paths))]

    for img_path in tqdm(sample_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, GTSRBConfig.IMG_SIZE)
        img = img.astype(np.float32) / 255.0

        pixel_sum += img.sum(axis=(0, 1))
        pixel_sq_sum += (img**2).sum(axis=(0, 1))
        pixel_count += img.shape[0] * img.shape[1]

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean**2)

    print(f"Mean: {mean}")
    print(f"Std: {std}")

    return mean, std


def main() -> None:
    """Main preprocessing pipeline."""

    print("=" * 60)
    print("GTSRB Data Preprocessing Pipeline")
    print("=" * 60)

    # Load raw data
    image_paths, labels = load_gtsrb_data(GTSRBConfig.DATA_DIR)

    # Create splits
    splits = create_data_splits(image_paths, labels)

    # Save splits
    save_splits(splits, GTSRBConfig.PROCESSED_DIR)

    # Extract HOG features for all splits
    for split_name, (paths, _) in splits.items():
        hog_save_path = GTSRBConfig.PROCESSED_DIR / f"{split_name}_hog_features.pkl"
        extract_all_hog_features(paths, save_path=hog_save_path)

    # Optionally compute dataset statistics (if you want to verify normalization values)
    print("\nComputing dataset statistics...")
    mean, std = compute_dataset_statistics(splits["train"][0])
    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Processed data saved to: {GTSRBConfig.PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

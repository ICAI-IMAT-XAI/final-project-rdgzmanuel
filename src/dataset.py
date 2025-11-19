"""
PyTorch Dataset classes for AKI prediction.
"""

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.config import Config

logger = logging.getLogger("aki_prediction")


class AKIDataset(Dataset):
    """
    PyTorch Dataset for AKI prediction with tabular features.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "label_aki_48h",
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ):
        """
        Initialize AKI dataset.

        Args:
            data: DataFrame with features and labels
            feature_cols: List of feature column names
            label_col: Name of label column
            scaler: Pre-fitted scaler (for val/test sets)
            fit_scaler: Whether to fit scaler on this data
        """
        self.data = data.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.label_col = label_col

        # Extract features and labels
        self.X = self.data[feature_cols].values.astype(np.float32)
        self.y = self.data[label_col].values.astype(np.float32)

        # Standardize features
        if fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        elif scaler is not None:
            self.scaler = scaler
            self.X = self.scaler.transform(self.X)
        else:
            self.scaler = None

        # Convert to tensors
        self.X_tensor = torch.from_numpy(self.X)
        self.y_tensor = torch.from_numpy(self.y)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Tuple of (features, label)
        """
        return self.X_tensor[idx], self.y_tensor[idx]

    def get_feature_names(self) -> list[str]:
        """Return feature names."""
        return self.feature_cols

    def get_scaler(self) -> StandardScaler | None:
        """Return fitted scaler."""
        return self.scaler


class AKISequenceDataset(Dataset):
    """
    PyTorch Dataset for AKI prediction with sequential features (for LSTM).
    Each sample is a time series of observations for a patient.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "label_aki_48h",
        sequence_length: int = 24,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ):
        """
        Initialize sequential AKI dataset.

        Args:
            data: DataFrame with features and labels
            feature_cols: List of feature column names
            label_col: Name of label column
            sequence_length: Length of sequence (hours)
            scaler: Pre-fitted scaler
            fit_scaler: Whether to fit scaler
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.sequence_length = sequence_length

        # Group by stay_id to create sequences
        self.sequences = []
        self.labels = []

        for stay_id in data["stay_id"].unique():
            stay_data = data[data["stay_id"] == stay_id].sort_values("hour")

            # Create sliding windows
            for i in range(len(stay_data) - sequence_length + 1):
                window = stay_data.iloc[i : i + sequence_length]

                # Features: sequence of observations
                features = window[feature_cols].values.astype(np.float32)

                # Label: prediction at end of sequence
                label = window[label_col].iloc[-1]

                self.sequences.append(features)
                self.labels.append(label)

        # Convert to numpy
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels).astype(np.float32)

        # Standardize features
        # Reshape to (samples * time_steps, features) for scaling
        n_samples, n_timesteps, n_features = self.sequences.shape
        sequences_reshaped = self.sequences.reshape(-1, n_features)

        if fit_scaler:
            self.scaler = StandardScaler()
            sequences_scaled = self.scaler.fit_transform(sequences_reshaped)
        elif scaler is not None:
            self.scaler = scaler
            sequences_scaled = self.scaler.transform(sequences_reshaped)
        else:
            self.scaler = None
            sequences_scaled = sequences_reshaped

        # Reshape back to (samples, time_steps, features)
        self.sequences = sequences_scaled.reshape(n_samples, n_timesteps, n_features)

        # Convert to tensors
        self.X_tensor = torch.from_numpy(self.sequences)
        self.y_tensor = torch.from_numpy(self.labels)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Tuple of (sequence_features, label)
        """
        return self.X_tensor[idx], self.y_tensor[idx]

    def get_scaler(self) -> StandardScaler | None:
        """Return fitted scaler."""
        return self.scaler


def prepare_datasets(
    df: pd.DataFrame,
    config: Config,
    sequence_length: int = 24,
) -> dict[str, Dataset]:
    """
    Prepare train/val/test datasets.

    Args:
        df: Final dataset DataFrame
        config: Configuration object
        sequence_length: Sequence length for LSTM (if used)

    Returns:
        Dictionary with 'train', 'val', 'test' datasets
    """
    logger.info("Preparing train/val/test datasets...")

    # Get feature columns (exclude metadata and labels)
    exclude_cols = [
        "stay_id",
        "hour",
        "subject_id",
        "hadm_id",
        "intime",
        "outtime",
        "aki_stage",
        "aki_stage_creat",
        "aki_stage_uo",
        "label_aki_48h",
        "died_in_hospital",
        "los_hours",
        "gender",
        "admission_type",
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"Using {len(feature_cols)} features")

    # Split by stay_id to avoid data leakage
    stay_ids = df["stay_id"].unique()

    # First split: train and temp (val+test)
    train_ids, temp_ids = train_test_split(
        stay_ids,
        test_size=(config.model.val_ratio + config.model.test_ratio),
        random_state=config.model.random_seed,
    )

    # Second split: val and test
    val_size = config.model.val_ratio / (
        config.model.val_ratio + config.model.test_ratio
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=(1 - val_size), random_state=config.model.random_seed
    )

    # Create data splits
    train_df = df[df["stay_id"].isin(train_ids)].copy()
    val_df = df[df["stay_id"].isin(val_ids)].copy()
    test_df = df[df["stay_id"].isin(test_ids)].copy()

    logger.info(f"Train: {len(train_df)} samples ({len(train_ids)} stays)")
    logger.info(f"Val: {len(val_df)} samples ({len(val_ids)} stays)")
    logger.info(f"Test: {len(test_df)} samples ({len(test_ids)} stays)")

    # Check class balance
    logger.info("\nClass distribution:")
    logger.info(
        f"Train - Positive: {train_df['label_aki_48h'].sum()} "
        f"({100 * train_df['label_aki_48h'].mean():.2f}%)"
    )
    logger.info(
        f"Val - Positive: {val_df['label_aki_48h'].sum()} "
        f"({100 * val_df['label_aki_48h'].mean():.2f}%)"
    )
    logger.info(
        f"Test - Positive: {test_df['label_aki_48h'].sum()} "
        f"({100 * test_df['label_aki_48h'].mean():.2f}%)"
    )

    # Create tabular datasets
    train_dataset = AKIDataset(train_df, feature_cols, fit_scaler=True)
    val_dataset = AKIDataset(val_df, feature_cols, scaler=train_dataset.get_scaler())
    test_dataset = AKIDataset(test_df, feature_cols, scaler=train_dataset.get_scaler())

    # Create sequence datasets (for LSTM)
    train_seq_dataset = AKISequenceDataset(
        train_df, feature_cols, sequence_length=sequence_length, fit_scaler=True
    )
    val_seq_dataset = AKISequenceDataset(
        val_df,
        feature_cols,
        sequence_length=sequence_length,
        scaler=train_seq_dataset.get_scaler(),
    )
    test_seq_dataset = AKISequenceDataset(
        test_df,
        feature_cols,
        sequence_length=sequence_length,
        scaler=train_seq_dataset.get_scaler(),
    )

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
        "train_seq": train_seq_dataset,
        "val_seq": val_seq_dataset,
        "test_seq": test_seq_dataset,
        "feature_cols": feature_cols,
    }


def get_dataloaders(
    datasets: dict[str, Dataset],
    config: Config,
) -> dict[str, DataLoader]:
    """
    Create DataLoaders from datasets.

    Args:
        datasets: Dictionary of datasets
        config: Configuration object

    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.model.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=config.model.batch_size,
            shuffle=False,
            num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.model.batch_size,
            shuffle=False,
            num_workers=0,
        ),
    }

    # Add sequence dataloaders if available
    if "train_seq" in datasets:
        dataloaders["train_seq"] = DataLoader(
            datasets["train_seq"],
            batch_size=config.model.batch_size,
            shuffle=True,
            num_workers=0,
        )
        dataloaders["val_seq"] = DataLoader(
            datasets["val_seq"],
            batch_size=config.model.batch_size,
            shuffle=False,
            num_workers=0,
        )
        dataloaders["test_seq"] = DataLoader(
            datasets["test_seq"],
            batch_size=config.model.batch_size,
            shuffle=False,
            num_workers=0,
        )

    return dataloaders

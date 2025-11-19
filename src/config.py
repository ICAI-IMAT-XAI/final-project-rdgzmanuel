"""
Configuration file for AKI Prediction Project
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""

    # Paths
    mimic_root: Path = Path("data/raw/mimic-iv-3.1")
    processed_dir: Path = Path("data/processed")

    # Core tables
    hosp_dir: Path = mimic_root / "hosp"
    icu_dir: Path = mimic_root / "icu"

    # Specific files
    admissions_file: Path = hosp_dir / "admissions.csv.gz"
    patients_file: Path = hosp_dir / "patients.csv.gz"
    icustays_file: Path = icu_dir / "icustays.csv.gz"
    chartevents_file: Path = icu_dir / "chartevents.csv.gz"
    labevents_file: Path = hosp_dir / "labevents.csv.gz"
    outputevents_file: Path = icu_dir / "outputevents.csv.gz"
    inputevents_file: Path = icu_dir / "inputevents.csv.gz"
    d_items_file: Path = icu_dir / "d_items.csv.gz"
    d_labitems_file: Path = hosp_dir / "d_labitems.csv.gz"

    # Processing parameters
    chunk_size: int = 100_000  # For reading large files
    max_icu_stays: int | None = 30_000  # Limit to 30K stays for manageable dataset


@dataclass
class AKIConfig:
    """AKI labeling configuration based on KDIGO criteria."""

    # Prediction window
    prediction_window_hours: int = 48  # Predict AKI within next 48 hours
    min_hours_from_admission: int = 6  # Exclude first 6 hours (stabilization)

    # KDIGO Creatinine criteria (mg/dL)
    creat_increase_absolute: float = 0.3  # Stage 1: ≥0.3 mg/dL increase in 48h
    creat_increase_relative_stage1: float = 1.5  # Stage 1: 1.5x baseline in 7d
    creat_increase_relative_stage2: float = 2.0  # Stage 2: 2.0x baseline in 7d
    creat_increase_relative_stage3: float = 3.0  # Stage 3: 3.0x baseline in 7d
    creat_absolute_stage3: float = 4.0  # Stage 3: ≥4.0 mg/dL

    # KDIGO Urine output criteria (mL/kg/h)
    uo_threshold_stage1: float = 0.5  # Stage 1: <0.5 mL/kg/h for 6h
    uo_duration_stage1_hours: int = 6
    uo_threshold_stage2: float = 0.5  # Stage 2: <0.5 mL/kg/h for 12h
    uo_duration_stage2_hours: int = 12
    uo_threshold_stage3: float = 0.3  # Stage 3: <0.3 mL/kg/h for 24h
    uo_duration_stage3_hours: int = 24
    uo_threshold_anuria: float = 0.0  # Anuria for 12h
    uo_anuria_duration_hours: int = 12

    # Target AKI stages to predict (we'll focus on Stage 2+)
    target_stages: list[int] = None  # [2, 3] for moderate-severe AKI

    def __post_init__(self):
        if self.target_stages is None:
            self.target_stages = [2, 3]


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    # Vital signs itemids (from d_items.csv)
    vital_itemids: dict = None

    # Lab test itemids (from d_labitems.csv)
    lab_itemids: dict = None

    # Time-based features
    time_windows_hours: list[int] = None  # [6, 12, 24] for rolling statistics

    def __post_init__(self):
        if self.vital_itemids is None:
            # REDUCED: Only most critical vital signs
            self.vital_itemids = {
                "heart_rate": [220045],
                "sbp": [220050, 220179],
                "mbp": [220052, 220181],  # Mean BP (most important)
                "resp_rate": [220210],
                "spo2": [220277],
            }

        if self.lab_itemids is None:
            # REDUCED: Only essential labs for AKI
            self.lab_itemids = {
                "creatinine": [50912],
                "bun": [51006],
                "potassium": [50971],
                "sodium": [50983],
                "hemoglobin": [51222],
            }

        if self.time_windows_hours is None:
            # REDUCED: Only 24h window instead of [6, 12, 24]
            self.time_windows_hours = [24]


@dataclass
class ModelConfig:
    """Model training configuration."""

    # Model types to train
    model_types: list[str] = None  # ['logistic', 'rf', 'xgboost', 'lstm']

    # Training parameters
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 50
    early_stopping_patience: int = 10

    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Class imbalance handling
    use_class_weights: bool = True
    oversample_minority: bool = False

    # Random seed
    random_seed: int = 42

    # Device
    device: str = "cuda"  # "cuda" or "cpu"

    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ["logistic", "rf", "xgboost", "lstm"]


@dataclass
class Config:
    """Main configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    aki: AKIConfig = field(default_factory=AKIConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        self.data.processed_dir.mkdir(parents=True, exist_ok=True)


config = Config()

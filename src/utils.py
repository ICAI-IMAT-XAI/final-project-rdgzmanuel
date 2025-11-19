"""
Utility functions for AKI prediction project.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_file: Optional path to log file

    Returns:
        Configured logger
    """
    logger = logging.getLogger("aki_prediction")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def read_mimic_csv(
    filepath: Path,
    columns: list[str] | None = None,
    chunk_size: int | None = None,
    filters: dict | None = None,
) -> pd.DataFrame:
    """
    Read MIMIC CSV file (compressed or not) with optional filtering.

    Args:
        filepath: Path to CSV file (.csv or .csv.gz)
        columns: Specific columns to read (None = all)
        chunk_size: Read in chunks to save memory
        filters: Dictionary of {column: values} to filter rows

    Returns:
        DataFrame with loaded data
    """
    logger = logging.getLogger("aki_prediction")
    logger.info(f"Reading {filepath.name}...")

    if not chunk_size:
        df = pd.read_csv(filepath, usecols=columns)
        if filters:
            for col, values in filters.items():
                df = df[df[col].isin(values)]
        return df

    # Read in chunks
    chunks = []
    for chunk in pd.read_csv(filepath, usecols=columns, chunksize=chunk_size):
        if filters:
            for col, values in filters.items():
                chunk = chunk[chunk[col].isin(values)]
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def save_processed_data(
    df: pd.DataFrame,
    filename: str,
    output_dir: Path,
) -> Path:
    """
    Save processed DataFrame to parquet format.

    Args:
        df: DataFrame to save
        filename: Output filename (without extension)
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filename}.parquet"
    df.to_parquet(output_path, index=False, compression="gzip")

    logger = logging.getLogger("aki_prediction")
    logger.info(f"Saved {len(df)} rows to {output_path}")

    return output_path


def load_processed_data(filepath: Path) -> pd.DataFrame:
    """
    Load processed data from parquet format.

    Args:
        filepath: Path to parquet file

    Returns:
        Loaded DataFrame
    """
    logger = logging.getLogger("aki_prediction")
    logger.info(f"Loading {filepath.name}...")
    return pd.read_parquet(filepath)


def convert_time_to_hours(
    time_col: pd.Series,
    reference_time: pd.Series,
) -> pd.Series:
    """
    Convert datetime difference to hours since reference time.

    Args:
        time_col: Time column to convert
        reference_time: Reference time (e.g., ICU admission time)

    Returns:
        Hours since reference time
    """
    time_diff = pd.to_datetime(time_col) - pd.to_datetime(reference_time)
    return time_diff.dt.total_seconds() / 3600


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get human-readable memory usage of DataFrame.

    Args:
        df: DataFrame to check

    Returns:
        Memory usage string
    """
    memory_bytes = df.memory_usage(deep=True).sum()

    for unit in ["B", "KB", "MB", "GB"]:
        if memory_bytes < 1024:
            return f"{memory_bytes:.2f} {unit}"
        memory_bytes /= 1024

    return f"{memory_bytes:.2f} TB"


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric types.

    Args:
        df: DataFrame to optimize
        verbose: Print memory reduction info

    Returns:
        Memory-optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            # Machine limits for numeric types.
            if pd.api.types.is_integer_dtype(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:  # float
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        logger = logging.getLogger("aki_prediction")
        reduction = 100 * (start_mem - end_mem) / start_mem
        logger.info(
            f"Memory reduced from {start_mem:.2f}MB to {end_mem:.2f}MB "
            f"({reduction:.1f}% reduction)"
        )

    return df


def calculate_age(birth_date: pd.Series, reference_date: pd.Series) -> pd.Series:
    """
    Calculate age in years from birth date and reference date.

    Args:
        birth_date: Birth dates
        reference_date: Reference dates (e.g., admission date)

    Returns:
        Age in years
    """
    birth = pd.to_datetime(birth_date)
    reference = pd.to_datetime(reference_date)
    age = (reference - birth).dt.days / 365.25
    return age.round(1)


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print useful information about a DataFrame.

    Args:
        df: DataFrame to inspect
        name: Name to display
    """
    logger = logging.getLogger("aki_prediction")
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{name} Info:")
    logger.info(f"{'=' * 60}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory: {get_memory_usage(df)}")
    logger.info(f"\nColumns: {list(df.columns)}")
    logger.info(f"\nMissing values:\n{df.isnull().sum()}")
    logger.info(f"\nData types:\n{df.dtypes}")
    logger.info(f"{'=' * 60}\n")

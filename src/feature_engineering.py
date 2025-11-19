"""
Feature engineering module.
Creates features from time series data for AKI prediction.
"""

import logging

import pandas as pd

from src.config import Config
from src.utils import reduce_mem_usage, save_processed_data

logger = logging.getLogger("aki_prediction")


def aggregate_vitals_by_hour(
    vitals: pd.DataFrame,
    time_windows: list[int] = [6, 12, 24],
) -> pd.DataFrame:
    """
    Aggregate vital signs into hourly features with rolling statistics.

    For each vital sign, calculates:
    - Most recent value
    - Mean over time windows
    - Min/Max over time windows
    - Trend (slope of linear regression)

    Args:
        vitals: Processed vitals DataFrame
        time_windows: Time windows in hours for rolling statistics

    Returns:
        DataFrame with aggregated vital features per stay-hour
    """
    logger.info("Aggregating vital signs by hour...")

    # Bin into hourly intervals
    vitals["hour"] = vitals["hours_since_intime"].astype(int)

    # Get most recent value per hour
    recent_vitals = (
        vitals.sort_values("hours_since_intime")
        .groupby(["stay_id", "hour", "vital_name"])["valuenum"]
        .last()
        .reset_index()
    )

    # Pivot to wide format
    vitals_wide = recent_vitals.pivot_table(
        index=["stay_id", "hour"],
        columns="vital_name",
        values="valuenum",
        aggfunc="last",
    ).reset_index()

    # Rename columns
    vitals_wide.columns = [
        f"vital_{col}" if col not in ["stay_id", "hour"] else col
        for col in vitals_wide.columns
    ]

    # Calculate rolling statistics
    for window in time_windows:
        logger.info(f"Calculating {window}h rolling statistics...")

        for vital in vitals["vital_name"].unique():
            col_name = f"vital_{vital}"

            if col_name not in vitals_wide.columns:
                continue

            # Mean
            vitals_wide[f"{col_name}_mean_{window}h"] = vitals_wide.groupby("stay_id")[
                col_name
            ].transform(lambda x: x.rolling(window, min_periods=1).mean())

            # Min
            vitals_wide[f"{col_name}_min_{window}h"] = vitals_wide.groupby("stay_id")[
                col_name
            ].transform(lambda x: x.rolling(window, min_periods=1).min())

            # Max
            vitals_wide[f"{col_name}_max_{window}h"] = vitals_wide.groupby("stay_id")[
                col_name
            ].transform(lambda x: x.rolling(window, min_periods=1).max())

            # Standard deviation
            vitals_wide[f"{col_name}_std_{window}h"] = vitals_wide.groupby("stay_id")[
                col_name
            ].transform(lambda x: x.rolling(window, min_periods=2).std())

    logger.info(f"Created {len(vitals_wide.columns)} vital sign features")

    return reduce_mem_usage(vitals_wide)


def aggregate_labs_by_hour(
    labs: pd.DataFrame,
    time_windows: list[int] = [6, 12, 24],
) -> pd.DataFrame:
    """
    Aggregate lab results into hourly features.

    Args:
        labs: Processed labs DataFrame
        time_windows: Time windows for rolling statistics

    Returns:
        DataFrame with aggregated lab features per stay-hour
    """
    logger.info("Aggregating lab results by hour...")

    # Bin into hourly intervals
    labs["hour"] = labs["hours_since_intime"].astype(int)

    # Get most recent value per hour
    recent_labs = (
        labs.sort_values("hours_since_intime")
        .groupby(["stay_id", "hour", "lab_name"])["valuenum"]
        .last()
        .reset_index()
    )

    # Pivot to wide format
    labs_wide = recent_labs.pivot_table(
        index=["stay_id", "hour"], columns="lab_name", values="valuenum", aggfunc="last"
    ).reset_index()

    # Rename columns
    labs_wide.columns = [
        f"lab_{col}" if col not in ["stay_id", "hour"] else col
        for col in labs_wide.columns
    ]

    # Calculate rolling statistics
    for window in time_windows:
        logger.info(f"Calculating {window}h rolling lab statistics...")

        for lab in labs["lab_name"].unique():
            col_name = f"lab_{lab}"

            if col_name not in labs_wide.columns:
                continue

            # Mean
            labs_wide[f"{col_name}_mean_{window}h"] = labs_wide.groupby("stay_id")[
                col_name
            ].transform(lambda x: x.rolling(window, min_periods=1).mean())

            # Trend (change over window)
            labs_wide[f"{col_name}_change_{window}h"] = labs_wide.groupby("stay_id")[
                col_name
            ].transform(lambda x: x - x.shift(window))

    logger.info(f"Created {len(labs_wide.columns)} lab features")

    return reduce_mem_usage(labs_wide)


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived clinical features from vitals and labs.

    Includes:
    - Shock Index (HR / SBP)
    - Mean Arterial Pressure
    - BUN/Creatinine ratio
    - Missing data indicators

    Args:
        df: DataFrame with vital and lab features

    Returns:
        DataFrame with additional derived features
    """
    logger.info("Creating derived clinical features...")

    # Shock Index (HR / SBP) - indicator of shock
    if "vital_heart_rate" in df.columns and "vital_sbp" in df.columns:
        df["shock_index"] = df["vital_heart_rate"] / df["vital_sbp"]
        df["shock_index"] = df["shock_index"].clip(0, 5)  # Reasonable range

    # Pulse pressure (SBP - DBP)
    if "vital_sbp" in df.columns and "vital_dbp" in df.columns:
        df["pulse_pressure"] = df["vital_sbp"] - df["vital_dbp"]

    # BUN/Creatinine ratio - indicator of prerenal azotemia
    if "lab_bun" in df.columns and "lab_creatinine" in df.columns:
        df["bun_creat_ratio"] = df["lab_bun"] / df["lab_creatinine"]
        df["bun_creat_ratio"] = df["bun_creat_ratio"].clip(0, 100)

    # Missing data indicators (important for clinical data)
    feature_cols = [col for col in df.columns if col.startswith(("vital_", "lab_"))]

    # Create all missing indicators at once using pd.concat
    missing_indicators = pd.DataFrame(
        {f"{col}_missing": df[col].isna().astype(int) for col in feature_cols},
        index=df.index,
    )
    df = pd.concat([df, missing_indicators], axis=1)

    logger.info(f"Created {3 + len(feature_cols)} derived features")

    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features.

    Args:
        df: DataFrame with hour column

    Returns:
        DataFrame with temporal features
    """
    logger.info("Creating temporal features...")

    # Time since ICU admission (already have 'hour')
    df["hour_of_day"] = df["hour"] % 24

    # Day/night indicator (night = 22:00 to 06:00)
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] < 6)).astype(int)

    # Days since admission
    df["days_since_admission"] = df["hour"] // 24

    logger.info("Created temporal features")

    return df


def merge_all_features(
    aki_labels: pd.DataFrame,
    vitals_features: pd.DataFrame,
    labs_features: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Merge all features into final dataset.

    Args:
        aki_labels: AKI labels DataFrame
        vitals_features: Vital sign features
        labs_features: Lab features
        config: Configuration object

    Returns:
        Final merged dataset ready for modeling
    """
    logger.info("Merging all features...")

    # Start with AKI labels
    final_df = aki_labels.copy()

    # Merge vitals
    final_df = final_df.merge(vitals_features, on=["stay_id", "hour"], how="left")

    # Merge labs
    final_df = final_df.merge(labs_features, on=["stay_id", "hour"], how="left")

    # Create derived features
    # final_df = create_derived_features(final_df)

    # Create temporal features
    final_df = create_temporal_features(final_df)

    # Forward fill missing values within each stay (carry forward last observation)
    feature_cols = [
        col
        for col in final_df.columns
        if col.startswith(("vital_", "lab_", "shock_", "pulse_", "bun_"))
    ]

    final_df[feature_cols] = final_df.groupby("stay_id")[feature_cols].fillna(
        method="ffill"
    )

    # Fill remaining NaNs with 0 (or median - you can adjust)
    final_df[feature_cols] = final_df[feature_cols].fillna(0)

    logger.info(f"Final dataset shape: {final_df.shape}")
    logger.info(
        f"Total features: {len([col for col in final_df.columns if col.startswith(('vital_', 'lab_', 'shock_', 'pulse_', 'bun_', 'hour_', 'is_', 'days_'))])}"
    )

    return reduce_mem_usage(final_df)


def run_feature_engineering(
    aki_labels: pd.DataFrame,
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Run complete feature engineering pipeline.

    Args:
        aki_labels: AKI labels DataFrame
        vitals: Processed vitals DataFrame
        labs: Processed labs DataFrame
        config: Configuration object

    Returns:
        Final dataset with all features
    """
    logger.info("=" * 60)
    logger.info("Starting feature engineering")
    logger.info("=" * 60)

    # Aggregate vitals
    vitals_features = aggregate_vitals_by_hour(
        vitals, config.features.time_windows_hours
    )

    # Aggregate labs
    labs_features = aggregate_labs_by_hour(labs, config.features.time_windows_hours)

    # Merge all features
    final_dataset = merge_all_features(
        aki_labels, vitals_features, labs_features, config
    )

    # Save final dataset
    logger.info("\nSaving final dataset...")
    save_processed_data(final_dataset, "final_dataset", config.data.processed_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Feature engineering complete!")
    logger.info("=" * 60)

    return final_dataset

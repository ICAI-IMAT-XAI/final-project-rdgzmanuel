"""
AKI label generation based on KDIGO criteria.
Implements KDIGO 2012 clinical practice guidelines for AKI.
"""

import logging

import pandas as pd

from src.config import Config
from src.utils import reduce_mem_usage, save_processed_data

logger = logging.getLogger("aki_prediction")


def get_baseline_creatinine(
    labs: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate baseline creatinine for each ICU stay.

    KDIGO defines baseline as:
    - Lowest creatinine in 7 days before ICU admission, or
    - Lowest creatinine in first 48h of ICU stay

    Args:
        labs: Processed labs DataFrame
        cohort: Cohort DataFrame

    Returns:
        DataFrame with baseline creatinine per stay
    """
    logger.info("Calculating baseline creatinine...")

    # Get creatinine measurements only
    creat = labs[labs["lab_name"] == "creatinine"].copy()

    # Get measurements in baseline window (-7 days to +48 hours)
    baseline_window = creat[
        (creat["hours_since_intime"] >= -168)  # -7 days
        & (creat["hours_since_intime"] <= 48)
    ].copy()

    # Get lowest creatinine per stay
    baseline = baseline_window.groupby("stay_id").agg({"valuenum": "min"}).reset_index()

    baseline.rename(columns={"valuenum": "baseline_creatinine"}, inplace=True)

    logger.info(f"Calculated baseline creatinine for {len(baseline)} stays")
    logger.info(f"Mean baseline: {baseline['baseline_creatinine'].mean():.2f} mg/dL")

    return baseline


def calculate_aki_creatinine_stage(
    labs: pd.DataFrame,
    baseline: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Calculate AKI stage based on creatinine criteria.

    KDIGO creatinine criteria:
    - Stage 1: ≥0.3 mg/dL increase within 48h OR 1.5-1.9x baseline within 7 days
    - Stage 2: 2.0-2.9x baseline
    - Stage 3: ≥3.0x baseline OR ≥4.0 mg/dL OR RRT

    Args:
        labs: Processed labs DataFrame
        baseline: Baseline creatinine DataFrame
        config: Configuration object

    Returns:
        DataFrame with hourly AKI stages based on creatinine
    """
    logger.info("Calculating AKI stages from creatinine...")

    # Get creatinine measurements
    creat = labs[labs["lab_name"] == "creatinine"].copy()

    # Merge with baseline
    creat = creat.merge(baseline, on="stay_id", how="left")

    # Calculate relative increase
    creat["creat_ratio"] = creat["valuenum"] / creat["baseline_creatinine"]

    # Sort by stay and time
    creat = creat.sort_values(["stay_id", "hours_since_intime"])

    # Calculate 48-hour rolling change
    creat["creat_48h_change"] = creat.groupby("stay_id")["valuenum"].transform(
        lambda x: x - x.shift(1).rolling(window=2, min_periods=1).min()
    )

    # Initialize stage
    creat["aki_stage_creat"] = 0

    # Stage 1: ≥0.3 increase in 48h or 1.5x baseline
    stage1_mask = (creat["creat_48h_change"] >= config.aki.creat_increase_absolute) | (
        creat["creat_ratio"] >= config.aki.creat_increase_relative_stage1
    )
    creat.loc[stage1_mask, "aki_stage_creat"] = 1

    # Stage 2: 2.0x baseline
    stage2_mask = creat["creat_ratio"] >= config.aki.creat_increase_relative_stage2
    creat.loc[stage2_mask, "aki_stage_creat"] = 2

    # Stage 3: 3.0x baseline or ≥4.0 mg/dL
    stage3_mask = (
        creat["creat_ratio"] >= config.aki.creat_increase_relative_stage3
    ) | (creat["valuenum"] >= config.aki.creat_absolute_stage3)
    creat.loc[stage3_mask, "aki_stage_creat"] = 3

    # Create hourly grid for each stay
    hourly_stages = []

    for stay_id in creat["stay_id"].unique():
        stay_creat = creat[creat["stay_id"] == stay_id].copy()

        # Get time range
        min_hour = int(stay_creat["hours_since_intime"].min())
        max_hour = int(stay_creat["hours_since_intime"].max())

        # Create hourly grid
        hours = range(min_hour, max_hour + 1)

        # Forward fill AKI stage
        for hour in hours:
            # Get most recent stage up to this hour
            recent_stages = stay_creat[stay_creat["hours_since_intime"] <= hour][
                "aki_stage_creat"
            ]

            if len(recent_stages) > 0:
                stage = recent_stages.iloc[-1]
            else:
                stage = 0

            hourly_stages.append({
                "stay_id": stay_id,
                "hour": hour,
                "aki_stage_creat": stage,
            })

    result = pd.DataFrame(hourly_stages)

    logger.info(
        f"Calculated creatinine-based AKI for {len(result)} hourly observations"
    )
    logger.info(
        f"AKI stages: {result['aki_stage_creat'].value_counts().sort_index().to_dict()}"
    )

    return reduce_mem_usage(result)


def calculate_aki_urine_stage(
    uo_hourly: pd.DataFrame,
    cohort: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Calculate AKI stage based on urine output criteria.

    KDIGO urine output criteria (requires patient weight):
    - Stage 1: <0.5 mL/kg/h for 6-12 hours
    - Stage 2: <0.5 mL/kg/h for ≥12 hours
    - Stage 3: <0.3 mL/kg/h for ≥24 hours OR anuria for ≥12 hours

    Args:
        uo_hourly: Hourly urine output DataFrame
        cohort: Cohort DataFrame with patient weights
        config: Configuration object

    Returns:
        DataFrame with hourly AKI stages based on UO
    """
    logger.info("Calculating AKI stages from urine output...")

    # For this implementation, we'll use an estimated weight
    # In production, you should extract actual weight from chartevents
    # Typical adult weight: 70-80 kg
    ESTIMATED_WEIGHT_KG = 75

    # Calculate UO rate (mL/kg/h)
    uo_hourly["uo_rate"] = uo_hourly["uo_ml"] / ESTIMATED_WEIGHT_KG

    # Sort by stay and hour
    uo_hourly = uo_hourly.sort_values(["stay_id", "hour_bin"])

    # Calculate rolling windows for each stay
    def calculate_uo_stage(group):
        """Calculate AKI stage for a single stay based on UO."""
        group = group.sort_values("hour_bin").reset_index(drop=True)
        group["aki_stage_uo"] = 0

        for idx in range(len(group)):
            hour = group.loc[idx, "hour_bin"]

            # Check 6-hour window (Stage 1)
            if idx >= 5:
                window_6h = group.loc[idx - 5 : idx, "uo_rate"]
                if (window_6h < config.aki.uo_threshold_stage1).all():
                    group.loc[idx, "aki_stage_uo"] = 1

            # Check 12-hour window (Stage 2)
            if idx >= 11:
                window_12h = group.loc[idx - 11 : idx, "uo_rate"]
                if (window_12h < config.aki.uo_threshold_stage2).all():
                    group.loc[idx, "aki_stage_uo"] = 2

                # Check for anuria (Stage 3)
                if (window_12h == config.aki.uo_threshold_anuria).all():
                    group.loc[idx, "aki_stage_uo"] = 3

            # Check 24-hour window (Stage 3)
            if idx >= 23:
                window_24h = group.loc[idx - 23 : idx, "uo_rate"]
                if (window_24h < config.aki.uo_threshold_stage3).all():
                    group.loc[idx, "aki_stage_uo"] = 3

        return group[["stay_id", "hour_bin", "aki_stage_uo"]]

    result = (
        uo_hourly.groupby("stay_id").apply(calculate_uo_stage).reset_index(drop=True)
    )
    result.rename(columns={"hour_bin": "hour"}, inplace=True)

    logger.info(f"Calculated UO-based AKI for {len(result)} hourly observations")
    logger.info(
        f"AKI stages: {result['aki_stage_uo'].value_counts().sort_index().to_dict()}"
    )

    return reduce_mem_usage(result)


def combine_aki_stages(
    aki_creat: pd.DataFrame,
    aki_uo: pd.DataFrame,
    cohort: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Combine creatinine and UO stages to get overall AKI stage.

    Overall stage is the maximum of creatinine and UO stages.

    Args:
        aki_creat: Creatinine-based AKI stages
        aki_uo: UO-based AKI stages
        cohort: Cohort DataFrame
        config: Configuration object

    Returns:
        DataFrame with combined AKI stages and prediction labels
    """
    logger.info("Combining AKI stages and creating labels...")

    # Merge creatinine and UO stages
    aki_combined = aki_creat.merge(aki_uo, on=["stay_id", "hour"], how="outer").fillna(
        0
    )

    # Overall stage is maximum
    aki_combined["aki_stage"] = aki_combined[["aki_stage_creat", "aki_stage_uo"]].max(
        axis=1
    )

    # Filter to prediction time window
    aki_combined = aki_combined[
        aki_combined["hour"] >= config.aki.min_hours_from_admission
    ].copy()

    # Create labels: predict if AKI stage 2+ will occur in next 48 hours
    def create_prediction_labels(group):
        """Create prediction labels for a single stay."""
        group = group.sort_values("hour").reset_index(drop=True)
        group["label_aki_48h"] = 0

        for idx in range(len(group)):
            # Look ahead 48 hours
            future_window = group.loc[idx:, "aki_stage"]
            future_hours = group.loc[idx:, "hour"] - group.loc[idx, "hour"]

            # Check if stage 2+ occurs within next 48 hours
            future_in_window = future_window[
                future_hours <= config.aki.prediction_window_hours
            ]

            if len(future_in_window) > 0:
                max_future_stage = future_in_window.max()
                if max_future_stage in config.aki.target_stages:
                    group.loc[idx, "label_aki_48h"] = 1

        return group

    aki_labeled = (
        aki_combined.groupby("stay_id")
        .apply(create_prediction_labels)
        .reset_index(drop=True)
    )

    # Add cohort information
    aki_labeled = aki_labeled.merge(
        cohort[
            [
                "stay_id",
                "age",
                "gender_encoded",
                "admission_type_encoded",
                "died_in_hospital",
                "los_hours",
            ]
        ],
        on="stay_id",
        how="left",
    )

    logger.info(f"\nFinal AKI-labeled dataset: {len(aki_labeled)} hourly observations")
    logger.info(
        f"Overall AKI stages: {aki_labeled['aki_stage'].value_counts().sort_index().to_dict()}"
    )
    logger.info(
        f"Prediction labels: {aki_labeled['label_aki_48h'].value_counts().to_dict()}"
    )
    logger.info(f"Positive rate: {100 * aki_labeled['label_aki_48h'].mean():.2f}%")

    return reduce_mem_usage(aki_labeled)


def run_label_generation(
    cohort: pd.DataFrame,
    labs: pd.DataFrame,
    uo_hourly: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Run complete AKI label generation pipeline.

    Args:
        cohort: Cohort DataFrame
        labs: Processed labs DataFrame
        uo_hourly: Hourly urine output DataFrame
        config: Configuration object

    Returns:
        DataFrame with AKI labels
    """
    logger.info("=" * 60)
    logger.info("Starting AKI label generation (KDIGO criteria)")
    logger.info("=" * 60)

    # Calculate baseline creatinine
    baseline = get_baseline_creatinine(labs, cohort)

    # Calculate creatinine-based stages
    aki_creat = calculate_aki_creatinine_stage(labs, baseline, config)

    # Calculate UO-based stages
    aki_uo = calculate_aki_urine_stage(uo_hourly, cohort, config)

    # Combine stages and create labels
    aki_labels = combine_aki_stages(aki_creat, aki_uo, cohort, config)

    # Save labeled data
    logger.info("\nSaving AKI labels...")
    save_processed_data(aki_labels, "aki_labels", config.data.processed_dir)
    save_processed_data(baseline, "baseline_creatinine", config.data.processed_dir)

    logger.info("\n" + "=" * 60)
    logger.info("AKI label generation complete!")
    logger.info("=" * 60)

    return aki_labels

"""
Data preprocessing module.
Cleans and merges extracted data.
"""

import logging

import pandas as pd

from src.config import Config
from src.utils import reduce_mem_usage, save_processed_data

logger = logging.getLogger("aki_prediction")


def create_cohort_base(
    icustays: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create base cohort with demographics and admission info.

    Args:
        icustays: ICU stays DataFrame
        patients: Patients DataFrame
        admissions: Admissions DataFrame

    Returns:
        Merged cohort DataFrame
    """
    logger.info("Creating base cohort...")

    # Merge ICU stays with admissions
    cohort = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="left")

    # Merge with patients
    cohort = cohort.merge(patients, on="subject_id", how="left")

    # Calculate age at admission
    cohort["age"] = cohort["anchor_age"] + (
        cohort["admittime"].dt.year - cohort["anchor_year"]
    )

    # Calculate if patient died in hospital
    cohort["died_in_hospital"] = cohort["hospital_expire_flag"] == 1

    # Calculate if patient died in ICU
    cohort["died_in_icu"] = (
        cohort["deathtime"].notna()
        & (cohort["deathtime"] >= cohort["intime"])
        & (cohort["deathtime"] <= cohort["outtime"])
    )

    # Encode categorical variables
    cohort["gender_encoded"] = (cohort["gender"] == "M").astype(int)

    # Admission type encoding
    admission_type_map = {
        "AMBULATORY OBSERVATION": 0,
        "DIRECT EMER.": 1,
        "DIRECT OBSERVATION": 2,
        "ELECTIVE": 3,
        "EU OBSERVATION": 4,
        "EW EMER.": 5,
        "OBSERVATION ADMIT": 6,
        "SURGICAL SAME DAY ADMISSION": 7,
        "URGENT": 8,
    }
    cohort["admission_type_encoded"] = cohort["admission_type"].map(admission_type_map)

    logger.info(f"Base cohort created: {len(cohort)} ICU stays")
    logger.info(f"Age range: {cohort['age'].min():.0f} - {cohort['age'].max():.0f}")
    logger.info(f"Gender: {cohort['gender'].value_counts().to_dict()}")
    logger.info(
        f"In-hospital mortality: {cohort['died_in_hospital'].sum()} "
        f"({100 * cohort['died_in_hospital'].mean():.1f}%)"
    )

    return cohort


def process_vitals_timeseries(
    vitals: pd.DataFrame,
    cohort: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Process vital signs into time series format.

    Args:
        vitals: Raw vitals DataFrame
        cohort: Cohort DataFrame
        config: Configuration object

    Returns:
        Processed vitals DataFrame
    """
    logger.info("Processing vital signs time series...")

    # Merge with cohort to get admission times
    vitals_merged = vitals.merge(
        cohort[["stay_id", "intime", "outtime"]], on="stay_id", how="inner"
    )

    # Calculate hours since ICU admission
    vitals_merged["hours_since_intime"] = (
        vitals_merged["charttime"] - vitals_merged["intime"]
    ).dt.total_seconds() / 3600

    # Filter to valid time range (during ICU stay)
    vitals_merged = vitals_merged[
        (vitals_merged["hours_since_intime"] >= 0)
        & (vitals_merged["charttime"] <= vitals_merged["outtime"])
    ].copy()

    # Map itemid to vital sign name
    itemid_to_name = {}
    for name, itemids in config.features.vital_itemids.items():
        for itemid in itemids:
            itemid_to_name[itemid] = name

    vitals_merged["vital_name"] = vitals_merged["itemid"].map(itemid_to_name)

    # Remove outliers (basic filtering)
    outlier_ranges = {
        "heart_rate": (0, 300),
        "sbp": (0, 300),
        "dbp": (0, 200),
        "mbp": (0, 250),
        "resp_rate": (0, 70),
        "temperature": (70, 115),  # Fahrenheit
        "spo2": (0, 100),
        "glucose": (0, 1000),
    }

    for vital, (min_val, max_val) in outlier_ranges.items():
        mask = vitals_merged["vital_name"] == vital
        vitals_merged.loc[mask, "valuenum"] = vitals_merged.loc[mask, "valuenum"].clip(
            min_val, max_val
        )

    # Keep only necessary columns
    vitals_processed = vitals_merged[
        ["stay_id", "charttime", "hours_since_intime", "vital_name", "valuenum"]
    ].copy()

    logger.info(f"Processed {len(vitals_processed):,} vital sign measurements")
    logger.info(
        f"Vital types: {vitals_processed['vital_name'].value_counts().to_dict()}"
    )

    return reduce_mem_usage(vitals_processed)


def process_labs_timeseries(
    labs: pd.DataFrame,
    cohort: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Process laboratory results into time series format.

    Args:
        labs: Raw labs DataFrame
        cohort: Cohort DataFrame
        config: Configuration object

    Returns:
        Processed labs DataFrame
    """
    logger.info("Processing laboratory results time series...")

    # Merge with cohort to get admission times
    labs_merged = labs.merge(
        cohort[["stay_id", "hadm_id", "intime", "outtime"]], on=["hadm_id"], how="inner"
    )

    # Calculate hours since ICU admission
    labs_merged["hours_since_intime"] = (
        labs_merged["charttime"] - labs_merged["intime"]
    ).dt.total_seconds() / 3600

    # Filter to relevant time range (within 48h before to end of ICU stay)
    labs_merged = labs_merged[
        (labs_merged["hours_since_intime"] >= -48)
        & (labs_merged["charttime"] <= labs_merged["outtime"])
    ].copy()

    # Map itemid to lab test name
    itemid_to_name = {}
    for name, itemids in config.features.lab_itemids.items():
        for itemid in itemids:
            itemid_to_name[itemid] = name

    labs_merged["lab_name"] = labs_merged["itemid"].map(itemid_to_name)

    # Remove outliers (basic filtering)
    outlier_ranges = {
        "creatinine": (0, 30),
        "bun": (0, 300),
        "potassium": (0, 15),
        "sodium": (100, 180),
        "chloride": (50, 150),
        "bicarbonate": (0, 60),
        "hemoglobin": (0, 25),
        "wbc": (0, 500),
        "platelet": (0, 2000),
    }

    for lab, (min_val, max_val) in outlier_ranges.items():
        mask = labs_merged["lab_name"] == lab
        labs_merged.loc[mask, "valuenum"] = labs_merged.loc[mask, "valuenum"].clip(
            min_val, max_val
        )

    # Keep only necessary columns
    labs_processed = labs_merged[
        ["stay_id", "charttime", "hours_since_intime", "lab_name", "valuenum"]
    ].copy()

    logger.info(f"Processed {len(labs_processed):,} lab measurements")
    logger.info(f"Lab types: {labs_processed['lab_name'].value_counts().to_dict()}")

    return reduce_mem_usage(labs_processed)


def process_urine_output_timeseries(
    urine_output: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process urine output into hourly measurements.

    Args:
        urine_output: Raw urine output DataFrame
        cohort: Cohort DataFrame

    Returns:
        Processed urine output DataFrame with hourly totals
    """
    logger.info("Processing urine output time series...")

    # Merge with cohort to get admission times
    uo_merged = urine_output.merge(
        cohort[["stay_id", "intime", "outtime"]], on="stay_id", how="inner"
    )

    # Calculate hours since ICU admission
    uo_merged["hours_since_intime"] = (
        uo_merged["charttime"] - uo_merged["intime"]
    ).dt.total_seconds() / 3600

    # Filter to valid time range
    uo_merged = uo_merged[
        (uo_merged["hours_since_intime"] >= 0)
        & (uo_merged["charttime"] <= uo_merged["outtime"])
    ].copy()

    # Bin into hourly intervals
    uo_merged["hour_bin"] = uo_merged["hours_since_intime"].astype(int)

    # Sum urine output per hour per stay
    uo_hourly = uo_merged.groupby(["stay_id", "hour_bin"])["value"].sum().reset_index()
    uo_hourly.rename(columns={"value": "uo_ml"}, inplace=True)

    logger.info(f"Processed urine output into {len(uo_hourly)} hourly measurements")
    logger.info(f"Mean hourly UO: {uo_hourly['uo_ml'].mean():.1f} mL")

    return reduce_mem_usage(uo_hourly)


def run_preprocessing(
    icustays: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    urine_output: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run complete preprocessing pipeline.

    Args:
        icustays: ICU stays DataFrame
        patients: Patients DataFrame
        admissions: Admissions DataFrame
        vitals: Vitals DataFrame
        labs: Labs DataFrame
        urine_output: Urine output DataFrame
        config: Configuration object

    Returns:
        Tuple of (cohort, vitals_processed, labs_processed, uo_processed)
    """
    logger.info("=" * 60)
    logger.info("Starting data preprocessing")
    logger.info("=" * 60)

    # Create base cohort
    cohort = create_cohort_base(icustays, patients, admissions)

    # Process time series data
    vitals_processed = process_vitals_timeseries(vitals, cohort, config)
    labs_processed = process_labs_timeseries(labs, cohort, config)
    uo_processed = process_urine_output_timeseries(urine_output, cohort)

    # Save preprocessed data
    logger.info("\nSaving preprocessed data...")
    save_processed_data(cohort, "cohort", config.data.processed_dir)
    save_processed_data(vitals_processed, "vitals_processed", config.data.processed_dir)
    save_processed_data(labs_processed, "labs_processed", config.data.processed_dir)
    save_processed_data(uo_processed, "uo_processed", config.data.processed_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Data preprocessing complete!")
    logger.info("=" * 60)

    return cohort, vitals_processed, labs_processed, uo_processed

"""
Data extraction module for MIMIC-IV dataset.
Extracts relevant tables for AKI prediction.
"""

import logging

import pandas as pd

from src.config import Config
from src.utils import (
    read_mimic_csv,
    reduce_mem_usage,
    save_processed_data,
)

logger = logging.getLogger("aki_prediction")


def extract_icu_stays(config: Config) -> pd.DataFrame:
    """
    Extract ICU stay information.

    Args:
        config: Configuration object

    Returns:
        DataFrame with ICU stay information
    """
    logger.info("Extracting ICU stays...")

    icustays = read_mimic_csv(
        config.data.icustays_file,
        columns=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "los",
            "first_careunit",
            "last_careunit",
        ],
    )

    if config.data.max_icu_stays:
        icustays = icustays.head(config.data.max_icu_stays)
        logger.info(f"Limited to {config.data.max_icu_stays} ICU stays for testing")

    # Convert times to datetime
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

    # Calculate length of stay in hours
    icustays["los_hours"] = (
        icustays["outtime"] - icustays["intime"]
    ).dt.total_seconds() / 3600

    # Filter out very short stays (less than 12 hours)
    icustays = icustays[icustays["los_hours"] >= 12].copy()

    logger.info(f"Extracted {len(icustays)} ICU stays (≥12 hours)")

    return reduce_mem_usage(icustays)


def extract_patients(config: Config, subject_ids: list[int]) -> pd.DataFrame:
    """
    Extract patient demographics.

    Args:
        config: Configuration object
        subject_ids: List of subject IDs to extract

    Returns:
        DataFrame with patient demographics
    """
    logger.info("Extracting patient demographics...")

    patients = read_mimic_csv(
        config.data.patients_file,
        columns=[
            "subject_id",
            "gender",
            "anchor_age",
            "anchor_year",
            "anchor_year_group",
            "dod",
        ],
    )

    # Filter to relevant patients
    patients = patients[patients["subject_id"].isin(subject_ids)].copy()

    # Convert death date to datetime
    patients["dod"] = pd.to_datetime(patients["dod"])

    logger.info(f"Extracted {len(patients)} patients")

    return reduce_mem_usage(patients)


def extract_admissions(config: Config, hadm_ids: list[int]) -> pd.DataFrame:
    """
    Extract hospital admission information.

    Args:
        config: Configuration object
        hadm_ids: List of hospital admission IDs to extract

    Returns:
        DataFrame with admission information
    """
    logger.info("Extracting hospital admissions...")

    admissions = read_mimic_csv(
        config.data.admissions_file,
        columns=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "hospital_expire_flag",
        ],
    )

    # Filter to relevant admissions
    admissions = admissions[admissions["hadm_id"].isin(hadm_ids)].copy()

    # Convert times to datetime
    time_cols = ["admittime", "dischtime", "deathtime"]
    for col in time_cols:
        admissions[col] = pd.to_datetime(admissions[col])

    logger.info(f"Extracted {len(admissions)} hospital admissions")

    return reduce_mem_usage(admissions)


def extract_vitals(
    config: Config, stay_ids: list[int], chunk_size: int = 1_000_000
) -> pd.DataFrame:
    """
    Extract vital signs from chartevents.

    Args:
        config: Configuration object
        stay_ids: List of ICU stay IDs to extract
        chunk_size: Chunk size for reading large file

    Returns:
        DataFrame with vital signs
    """
    logger.info("Extracting vital signs (this may take a while)...")

    # Get all vital sign itemids
    vital_itemids = []
    for itemid_list in config.features.vital_itemids.values():
        vital_itemids.extend(itemid_list)

    logger.info(f"Extracting {len(vital_itemids)} vital sign types...")

    # Read in chunks and filter
    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(
        config.data.chartevents_file,
        usecols=["stay_id", "charttime", "itemid", "value", "valuenum", "valueuom"],
        chunksize=chunk_size,
    ):
        # Filter to relevant stays and itemids
        chunk = chunk[
            (chunk["stay_id"].isin(stay_ids)) & (chunk["itemid"].isin(vital_itemids))
        ].copy()

        if len(chunk) > 0:
            chunks.append(chunk)
            total_rows += len(chunk)
            logger.info(f"Processed chunk: {total_rows:,} vital signs extracted so far")

    if not chunks:
        logger.warning("No vital signs found!")
        return pd.DataFrame()

    vitals = pd.concat(chunks, ignore_index=True)
    vitals["charttime"] = pd.to_datetime(vitals["charttime"])

    # Only keep numeric values
    vitals = vitals[vitals["valuenum"].notna()].copy()

    logger.info(f"Extracted {len(vitals):,} vital sign measurements")

    return reduce_mem_usage(vitals)


def extract_labs(
    config: Config, subject_ids: list[int], chunk_size: int = 1_000_000
) -> pd.DataFrame:
    """
    Extract laboratory results.

    Args:
        config: Configuration object
        subject_ids: List of subject IDs to extract
        chunk_size: Chunk size for reading large file

    Returns:
        DataFrame with lab results
    """
    logger.info("Extracting laboratory results (this may take a while)...")

    # Get all lab itemids
    lab_itemids = []
    for itemid_list in config.features.lab_itemids.values():
        lab_itemids.extend(itemid_list)

    logger.info(f"Extracting {len(lab_itemids)} lab test types...")

    # Read in chunks and filter
    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(
        config.data.labevents_file,
        usecols=[
            "subject_id",
            "hadm_id",
            "charttime",
            "itemid",
            "value",
            "valuenum",
            "valueuom",
            "flag",
        ],
        chunksize=chunk_size,
    ):
        # Filter to relevant subjects and itemids
        chunk = chunk[
            (chunk["subject_id"].isin(subject_ids))
            & (chunk["itemid"].isin(lab_itemids))
        ].copy()

        if len(chunk) > 0:
            chunks.append(chunk)
            total_rows += len(chunk)
            logger.info(f"Processed chunk: {total_rows:,} lab results extracted so far")

    if not chunks:
        logger.warning("No lab results found!")
        return pd.DataFrame()

    labs = pd.concat(chunks, ignore_index=True)
    labs["charttime"] = pd.to_datetime(labs["charttime"])

    # Only keep numeric values
    labs = labs[labs["valuenum"].notna()].copy()

    logger.info(f"Extracted {len(labs):,} lab measurements")

    return reduce_mem_usage(labs)


def extract_urine_output(
    config: Config, stay_ids: list[int], chunk_size: int = 500_000
) -> pd.DataFrame:
    """
    Extract urine output measurements.

    Args:
        config: Configuration object
        stay_ids: List of ICU stay IDs to extract
        chunk_size: Chunk size for reading

    Returns:
        DataFrame with urine output
    """
    logger.info("Extracting urine output...")

    # Read output events
    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(
        config.data.outputevents_file,
        usecols=["stay_id", "charttime", "itemid", "value"],
        chunksize=chunk_size,
    ):
        # Filter to relevant stays
        chunk = chunk[chunk["stay_id"].isin(stay_ids)].copy()

        if len(chunk) > 0:
            chunks.append(chunk)
            total_rows += len(chunk)
            logger.info(f"Processed chunk: {total_rows:,} output events extracted")

    if not chunks:
        logger.warning("No urine output found!")
        return pd.DataFrame()

    output = pd.concat(chunks, ignore_index=True)
    output["charttime"] = pd.to_datetime(output["charttime"])

    # Filter for urine output only (most common itemids)
    # itemids for urine: 226559 (Foley), 226560 (Void), 226561 (Condom Cath), etc.
    urine_itemids = [
        226559,
        226560,
        226561,
        226564,
        226565,
        226566,
        226567,
        226584,
        226627,
        226631,
    ]
    output = output[output["itemid"].isin(urine_itemids)].copy()

    logger.info(f"Extracted {len(output):,} urine output measurements")

    return reduce_mem_usage(output)


def run_extraction(config: Config) -> tuple[pd.DataFrame, ...]:
    """
    Run complete data extraction pipeline.

    Args:
        config: Configuration object

    Returns:
        Tuple of extracted DataFrames:
        (icustays, patients, admissions, vitals, labs, urine_output)
    """
    logger.info("=" * 60)
    logger.info("Starting data extraction from MIMIC-IV")
    logger.info("=" * 60)

    # Extract ICU stays first (this defines our cohort)
    icustays = extract_icu_stays(config)
    stay_ids = icustays["stay_id"].unique().tolist()
    subject_ids = icustays["subject_id"].unique().tolist()
    hadm_ids = icustays["hadm_id"].unique().tolist()

    logger.info(
        f"\nCohort: {len(stay_ids)} ICU stays, "
        f"{len(subject_ids)} patients, "
        f"{len(hadm_ids)} admissions"
    )

    # Extract patient demographics
    patients = extract_patients(config, subject_ids)

    # Extract admission information
    admissions = extract_admissions(config, hadm_ids)

    # Extract vitals (large file - may take time)
    vitals = extract_vitals(config, stay_ids)

    # Extract labs (large file - may take time)
    labs = extract_labs(config, subject_ids)

    # Extract urine output
    urine_output = extract_urine_output(config, stay_ids)

    # Save extracted data
    logger.info("\nSaving extracted data...")
    save_processed_data(icustays, "icustays_extracted", config.data.processed_dir)
    save_processed_data(patients, "patients_extracted", config.data.processed_dir)
    save_processed_data(admissions, "admissions_extracted", config.data.processed_dir)
    save_processed_data(vitals, "vitals_extracted", config.data.processed_dir)
    save_processed_data(labs, "labs_extracted", config.data.processed_dir)
    save_processed_data(
        urine_output, "urine_output_extracted", config.data.processed_dir
    )

    logger.info("\n" + "=" * 60)
    logger.info("Data extraction complete!")
    logger.info("=" * 60)

    return icustays, patients, admissions, vitals, labs, urine_output

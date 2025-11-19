"""
Complete data extraction and preprocessing pipeline.
Run this script first to prepare data before training.

Usage:
    python run_data_pipeline.py

This will:
1. Extract relevant data from MIMIC-IV CSVs
2. Preprocess and clean the data
3. Generate AKI labels using KDIGO criteria
4. Create features for modeling
5. Save everything to data/processed/
"""

from datetime import datetime
from pathlib import Path

from src.config import Config
from src.data_extraction import run_extraction
from src.data_preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.label_generation import run_label_generation
from src.utils import setup_logging


def main() -> None:
    """Run complete data pipeline."""

    # Setup
    config = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"logs/pipeline_{timestamp}.log")
    log_file.parent.mkdir(exist_ok=True)

    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("AKI Prediction - Data Pipeline")
    logger.info("=" * 60)
    logger.info(f"MIMIC-IV root: {config.data.mimic_root}")
    logger.info(f"Output directory: {config.data.processed_dir}")
    logger.info(f"Prediction window: {config.aki.prediction_window_hours} hours")
    logger.info(f"Target AKI stages: {config.aki.target_stages}")
    logger.info("=" * 60)

    # Check if MIMIC data exists
    if not config.data.mimic_root.exists():
        logger.error(f"MIMIC-IV directory not found: {config.data.mimic_root}")
        logger.error("Please update the mimic_root path in config.py")
        return

    # ========================================================================
    # Step 1: Data Extraction
    # ========================================================================
    logger.info("\n\nSTEP 1: DATA EXTRACTION")
    logger.info("=" * 60)

    try:
        icustays, patients, admissions, vitals, labs, urine_output = run_extraction(
            config
        )
    except Exception as e:
        logger.error(f"Error during data extraction: {e}", exc_info=True)
        return

    # ========================================================================
    # Step 2: Data Preprocessing
    # ========================================================================
    logger.info("\n\nSTEP 2: DATA PREPROCESSING")
    logger.info("=" * 60)

    try:
        cohort, vitals_processed, labs_processed, uo_processed = run_preprocessing(
            icustays, patients, admissions, vitals, labs, urine_output, config
        )
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        return

    # ========================================================================
    # Step 3: AKI Label Generation
    # ========================================================================
    logger.info("\n\nSTEP 3: AKI LABEL GENERATION")
    logger.info("=" * 60)

    try:
        aki_labels = run_label_generation(cohort, labs_processed, uo_processed, config)
    except Exception as e:
        logger.error(f"Error during label generation: {e}", exc_info=True)
        return

    # ========================================================================
    # Step 4: Feature Engineering
    # ========================================================================
    logger.info("\n\nSTEP 4: FEATURE ENGINEERING")
    logger.info("=" * 60)

    try:
        final_dataset = run_feature_engineering(
            aki_labels, vitals_processed, labs_processed, config
        )
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}", exc_info=True)
        return

    # ========================================================================
    # Pipeline Complete
    # ========================================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("DATA PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Final dataset shape: {final_dataset.shape}")
    logger.info(f"Total ICU stays: {final_dataset['stay_id'].nunique()}")
    logger.info(f"Total observations: {len(final_dataset)}")
    logger.info(f"Positive samples: {final_dataset['label_aki_48h'].sum()}")
    logger.info(f"Positive rate: {100 * final_dataset['label_aki_48h'].mean():.2f}%")
    logger.info(f"\nProcessed data saved to: {config.data.processed_dir}")
    logger.info("\nYou can now run: python main.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

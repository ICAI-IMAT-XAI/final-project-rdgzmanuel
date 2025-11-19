"""
Main training script for AKI prediction.
Run this after data extraction and preprocessing.
"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from src.config import Config
from src.dataset import get_dataloaders, prepare_datasets
from src.models import get_model
from src.training_functions import (
    evaluate_model,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_training_history,
    save_predictions,
    train_model,
)
from src.utils import load_processed_data, setup_logging


def main():
    """Main training pipeline."""

    # Setup
    config = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"logs/training_{timestamp}.log")
    log_file.parent.mkdir(exist_ok=True)

    logger = setup_logging(log_file)
    logger.info("=" * 60)
    logger.info("AKI Prediction - Training Pipeline")
    logger.info("=" * 60)

    # Set device
    device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.model.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.model.random_seed)

    # ========================================================================
    # Load processed data
    # ========================================================================
    logger.info("\nLoading processed data...")
    final_dataset = load_processed_data(
        config.data.processed_dir / "final_dataset.parquet"
    )

    logger.info(f"Dataset shape: {final_dataset.shape}")
    logger.info(
        f"Positive class rate: {100 * final_dataset['label_aki_48h'].mean():.2f}%"
    )

    # ========================================================================
    # Prepare datasets
    # ========================================================================
    datasets = prepare_datasets(final_dataset, config, sequence_length=24)
    dataloaders = get_dataloaders(datasets, config)

    # Get feature dimension
    input_dim = len(datasets["feature_cols"])
    logger.info(f"Input dimension: {input_dim} features")

    # ========================================================================
    # Train models
    # ========================================================================
    results_dir = Path("results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Calculate class weights for imbalanced data
    train_labels = datasets["train"].y
    pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
    logger.info(f"Positive class weight: {pos_weight:.2f}")

    # ========================================================================
    # 1. Logistic Regression (Baseline)
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training Logistic Regression")
    logger.info("=" * 60)

    # TODO: Implement sklearn model training
    # from models import LogisticRegressionModel
    # lr_model = LogisticRegressionModel(C=1.0, max_iter=1000, class_weight='balanced')
    # lr_model.fit(datasets['train'].X, datasets['train'].y)
    # lr_preds = lr_model.predict_proba(datasets['test'].X)
    # ... evaluate and save results

    # ========================================================================
    # 2. Random Forest (Baseline)
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training Random Forest")
    logger.info("=" * 60)

    # TODO: Implement RF training

    # ========================================================================
    # 3. XGBoost (Strong Baseline)
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training XGBoost")
    logger.info("=" * 60)

    # TODO: Implement XGBoost training
    # This will likely be your best baseline!

    # ========================================================================
    # 4. MLP (Deep Learning)
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training MLP")
    logger.info("=" * 60)

    # Initialize model
    mlp_model = get_model(
        "mlp", input_dim=input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3
    ).to(device)

    logger.info(f"Model architecture:\n{mlp_model}")

    # Loss function with class weights
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = optim.Adam(mlp_model.parameters(), lr=config.model.learning_rate)

    # Train
    mlp_save_path = results_dir / "mlp_best_model.pth"
    mlp_history = train_model(
        mlp_model,
        dataloaders["train"],
        dataloaders["val"],
        criterion,
        optimizer,
        num_epochs=config.model.num_epochs,
        device=device,
        early_stopping_patience=config.model.early_stopping_patience,
        save_path=str(mlp_save_path),
    )

    # Plot training history
    plot_training_history(
        mlp_history, save_path=str(results_dir / "mlp_training_history.png")
    )

    # Load best model and evaluate
    mlp_model.load_state_dict(torch.load(mlp_save_path))
    mlp_metrics, mlp_preds, mlp_labels = evaluate_model(
        mlp_model, dataloaders["test"], device
    )

    # Plot ROC and PR curves
    plot_roc_curve(
        mlp_labels, mlp_preds, save_path=str(results_dir / "mlp_roc_curve.png")
    )
    plot_precision_recall_curve(
        mlp_labels, mlp_preds, save_path=str(results_dir / "mlp_pr_curve.png")
    )

    # Save predictions
    save_predictions(
        mlp_preds, mlp_labels, save_path=str(results_dir / "mlp_predictions.csv")
    )

    # ========================================================================
    # 5. LSTM (Sequential Model)
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training LSTM")
    logger.info("=" * 60)

    # Initialize model
    lstm_model = get_model(
        "lstm", input_dim=input_dim, hidden_dim=128, num_layers=2, dropout_rate=0.3
    ).to(device)

    # Loss and optimizer
    criterion_lstm = nn.BCELoss()
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=config.model.learning_rate)

    # Train
    lstm_save_path = results_dir / "lstm_best_model.pth"
    lstm_history = train_model(
        lstm_model,
        dataloaders["train_seq"],
        dataloaders["val_seq"],
        criterion_lstm,
        optimizer_lstm,
        num_epochs=config.model.num_epochs,
        device=device,
        early_stopping_patience=config.model.early_stopping_patience,
        save_path=str(lstm_save_path),
    )

    # Plot and evaluate
    plot_training_history(
        lstm_history, save_path=str(results_dir / "lstm_training_history.png")
    )

    lstm_model.load_state_dict(torch.load(lstm_save_path))
    lstm_metrics, lstm_preds, lstm_labels = evaluate_model(
        lstm_model, dataloaders["test_seq"], device
    )

    plot_roc_curve(
        lstm_labels, lstm_preds, save_path=str(results_dir / "lstm_roc_curve.png")
    )

    save_predictions(
        lstm_preds, lstm_labels, save_path=str(results_dir / "lstm_predictions.csv")
    )

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete! Summary:")
    logger.info("=" * 60)
    logger.info(f"MLP Test AUC: {mlp_metrics['auc']:.4f}")
    logger.info(f"LSTM Test AUC: {lstm_metrics['auc']:.4f}")
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

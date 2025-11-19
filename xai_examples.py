"""
Explainability analysis examples for AKI prediction models.
Run this after training models to understand predictions.

Usage:
    python explainability_examples.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_tabular import LimeTabularExplainer


def analyze_feature_importance_xgboost(
    model,
    feature_names: list,
    top_k: int = 20,
    save_path: str = "feature_importance.png",
):
    """
    Analyze and plot feature importance from XGBoost model.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_k: Number of top features to show
        save_path: Path to save plot
    """
    # Get feature importance
    importance = model.feature_importances_

    # Create dataframe
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    # Plot top K features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_k)
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_k} Most Important Features (XGBoost)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nTop {top_k} Features:")
    print(importance_df.head(top_k))

    return importance_df


def shap_global_analysis(
    model, X_test: np.ndarray, feature_names: list, save_dir: str = "shap_plots"
):
    """
    Global SHAP analysis - understand overall feature contributions.

    Args:
        model: Trained model (XGBoost or sklearn)
        X_test: Test features
        feature_names: Feature names
        save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(exist_ok=True)

    print("Computing SHAP values (this may take a few minutes)...")

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot - shows feature importance and impact direction
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Bar plot - mean absolute SHAP values
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_importance.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nSHAP plots saved to {save_dir}/")

    return shap_values


def shap_individual_prediction(
    explainer,
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    sample_idx: int = 0,
    save_path: str = "shap_force_plot.png",
):
    """
    Explain individual prediction using SHAP force plot.

    Args:
        explainer: SHAP explainer
        shap_values: Computed SHAP values
        X_test: Test features
        feature_names: Feature names
        sample_idx: Index of sample to explain
        save_path: Path to save plot
    """
    # Force plot
    shap.force_plot(
        explainer.expected_value,
        shap_values[sample_idx],
        X_test[sample_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Waterfall plot (better for single predictions)
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value,
            data=X_test[sample_idx],
            feature_names=feature_names,
        ),
        show=False,
    )
    plt.savefig(
        save_path.replace(".png", "_waterfall.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    print(f"\nExplanation for sample {sample_idx}:")
    print(f"Predicted probability: {X_test[sample_idx].mean():.3f}")  # Placeholder


def lime_explanation(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    sample_idx: int = 0,
    save_path: str = "lime_explanation.png",
):
    """
    Explain individual prediction using LIME.

    Args:
        model: Trained model with predict_proba method
        X_train: Training features (for LIME reference)
        X_test: Test features
        feature_names: Feature names
        sample_idx: Index of sample to explain
        save_path: Path to save explanation
    """
    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["No AKI", "AKI"],
        mode="classification",
    )

    # Explain instance
    exp = explainer.explain_instance(
        X_test[sample_idx], model.predict_proba, num_features=20
    )

    # Plot
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nLIME Explanation for sample {sample_idx}:")
    print("Top contributing features:")
    for feature, weight in exp.as_list()[:10]:
        print(f"  {feature}: {weight:+.3f}")

    return exp


def analyze_attention_weights(
    model,
    dataloader,
    device: str = "cuda",
    num_samples: int = 5,
    save_dir: str = "attention_plots",
):
    """
    Analyze attention weights from Attention LSTM model.
    Shows which time steps (hours) are most important for prediction.

    Args:
        model: Trained Attention LSTM model
        dataloader: Test dataloader
        device: Device
        num_samples: Number of samples to visualize
        save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(exist_ok=True)

    model.eval()
    samples_analyzed = 0

    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            if samples_analyzed >= num_samples:
                break

            sequences = sequences.to(device)

            # Forward pass to get attention weights
            predictions, attention_weights = model(sequences)

            # Visualize attention for each sample in batch
            for i in range(min(len(sequences), num_samples - samples_analyzed)):
                attention = attention_weights[i].cpu().numpy()

                plt.figure(figsize=(12, 4))
                plt.plot(attention, marker="o")
                plt.xlabel("Hours Before Prediction")
                plt.ylabel("Attention Weight")
                plt.title(
                    f"Attention Weights - Sample {samples_analyzed} "
                    f"(Pred: {predictions[i].item():.3f}, "
                    f"True: {labels[i].item():.0f})"
                )
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(
                    f"{save_dir}/attention_sample_{samples_analyzed}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                samples_analyzed += 1

    print(f"\nAttention visualizations saved to {save_dir}/")


def feature_correlation_analysis(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    top_k: int = 20,
    save_path: str = "feature_correlations.png",
):
    """
    Analyze correlation between features and target.

    Args:
        X: Features
        y: Labels
        feature_names: Feature names
        top_k: Number of top features to show
        save_path: Path to save plot
    """
    # Calculate correlations
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    correlations = df.corr()["target"].drop("target").abs().sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(10, 8))
    top_corr = correlations.head(top_k)
    plt.barh(top_corr.index, top_corr.values)
    plt.xlabel("Absolute Correlation with AKI")
    plt.title(f"Top {top_k} Features by Correlation")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nTop {top_k} Correlated Features:")
    print(correlations.head(top_k))

    return correlations


def partial_dependence_analysis(
    model,
    X: np.ndarray,
    feature_names: list,
    features_to_plot: list,
    save_dir: str = "pdp_plots",
):
    """
    Partial Dependence Plots - show how predictions change with feature values.

    Args:
        model: Trained model
        X: Features
        feature_names: Feature names
        features_to_plot: List of feature names to plot
        save_dir: Directory to save plots
    """
    from sklearn.inspection import PartialDependenceDisplay

    Path(save_dir).mkdir(exist_ok=True)

    # Get feature indices
    feature_indices = [
        feature_names.index(f) for f in features_to_plot if f in feature_names
    ]

    # Plot
    fig, ax = plt.subplots(figsize=(15, 10))
    PartialDependenceDisplay.from_estimator(
        model, X, feature_indices, feature_names=feature_names, ax=ax, n_cols=3
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/partial_dependence.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nPartial dependence plots saved to {save_dir}/")


def main():
    """Run explainability analysis examples."""

    print("=" * 60)
    print("AKI Prediction - Explainability Analysis")
    print("=" * 60)

    # TODO: Load your trained models and data
    # This is a template showing what analyses you can do

    print("\nAvailable Explainability Methods:")
    print("1. Feature Importance (XGBoost/RF)")
    print("2. SHAP Values (Global)")
    print("3. SHAP Values (Individual Predictions)")
    print("4. LIME (Local Explanations)")
    print("5. Attention Weights (LSTM)")
    print("6. Feature Correlations")
    print("7. Partial Dependence Plots")

    print("\nTo use these methods:")
    print("1. Train your models using main.py")
    print("2. Load the saved models")
    print("3. Call the analysis functions above")
    print("\nExample code is provided in each function!")


if __name__ == "__main__":
    main()

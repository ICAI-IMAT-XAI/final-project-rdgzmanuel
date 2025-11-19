# AKI Prediction from MIMIC-IV using Explainable AI

Predict **Acute Kidney Injury (AKI)** 48 hours in advance using ICU patient data from MIMIC-IV. This project implements KDIGO clinical criteria for AKI labeling and trains both baseline and deep learning models with a focus on **explainability**.

## 🎯 Project Overview

**Goal**: Predict if a patient will develop moderate-to-severe AKI (KDIGO Stage 2+) within the next 48 hours based on:
- Demographics (age, gender)
- Vital signs (HR, BP, temperature, SpO2, etc.)
- Laboratory values (creatinine, BUN, electrolytes, etc.)
- Urine output

**Why AKI?**
- Affects 10-15% of hospitalized patients
- Early detection enables intervention (adjust fluids, stop nephrotoxic drugs)
- Clear clinical criteria (KDIGO guidelines)
- Highly interpretable features

## 📁 Project Structure

```
aki_prediction_project/
├── data/
│   ├── raw/                    # Your MIMIC-IV CSV files go here
│   │   └── mimic-iv-3.1/
│   │       ├── hosp/
│   │       └── icu/
│   └── processed/              # Generated processed data
├── src/
│   ├── config.py               # Configuration
│   ├── utils.py                # Utility functions
│   ├── data_extraction.py      # Extract from MIMIC-IV
│   ├── data_preprocessing.py   # Clean and merge data
│   ├── label_generation.py     # KDIGO AKI labeling
│   ├── feature_engineering.py  # Create features
│   ├── dataset.py              # PyTorch datasets
│   ├── models.py               # Model architectures (templates)
│   └── training_functions.py   # Train/val/test functions
├── run_data_pipeline.py        # Run full data pipeline
├── main.py                     # Main training script
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (using uv package)
uv pip install -r requirements.txt
```

### 2. Configure Paths

Edit `config.py` to point to your MIMIC-IV data:

```python
mimic_root: Path = Path("data/raw/mimic-iv-3.1")
```

### 3. Run Data Pipeline

This extracts and processes all data (may take 30-60 minutes):

```bash
python run_data_pipeline.py
```

**What it does:**
1. Extracts ICU stays, patients, vitals, labs, urine output
2. Cleans and preprocesses time series data
3. Generates AKI labels using KDIGO criteria
4. Engineers features (rolling statistics, derived features)
5. Saves processed data to `data/processed/`

### 4. Train Models

```bash
python main.py
```

**Models trained:**
- Logistic Regression (baseline)
- Random Forest (baseline)
- XGBoost (strong baseline)
- MLP (Multi-Layer Perceptron)
- LSTM (sequential model)
- Attention LSTM (explainable sequential model)

Results saved to `results/[timestamp]/`

## 📊 Data Pipeline Details

### KDIGO AKI Criteria

AKI is defined by **creatinine** and **urine output** criteria:

**Creatinine:**
- Stage 1: ≥0.3 mg/dL increase in 48h OR 1.5-1.9× baseline
- Stage 2: 2.0-2.9× baseline
- Stage 3: ≥3.0× baseline OR ≥4.0 mg/dL

**Urine Output:**
- Stage 1: <0.5 mL/kg/h for 6-12 hours
- Stage 2: <0.5 mL/kg/h for ≥12 hours
- Stage 3: <0.3 mL/kg/h for ≥24 hours OR anuria ≥12h

Overall AKI stage = max(creatinine_stage, urine_stage)

### Features

**Static Features:**
- Age, gender, admission type

**Vital Signs** (rolling 6h, 12h, 24h statistics):
- Heart rate, blood pressure, respiratory rate
- Temperature, SpO2, glucose

**Laboratory Values** (rolling statistics + trends):
- Creatinine, BUN, electrolytes
- Hemoglobin, WBC, platelets

**Derived Features:**
- Shock index (HR/SBP)
- BUN/Creatinine ratio
- Pulse pressure
- Missing data indicators

**Temporal Features:**
- Hour since admission
- Hour of day, day/night
- Days since admission

## 🎓 Model Implementation Guide

The `models.py` file contains **templates** with docstrings. You should implement:

### Baseline Models (sklearn)

```python
# LogisticRegressionModel
# - Initialize with class_weight='balanced'
# - Good baseline, fast training

# RandomForestModel  
# - n_estimators=100, max_depth=10
# - Returns feature importances

# XGBoostModel
# - Best baseline model
# - Use scale_pos_weight for imbalance
# - Early stopping on validation set
```

### Deep Learning Models (PyTorch)

```python
# MLPModel
# - Feedforward network: Input -> [256,128,64] -> Output
# - BatchNorm + Dropout for regularization

# LSTMModel
# - Captures temporal patterns
# - Input: (batch, seq_length=24, features)

# AttentionLSTM
# - LSTM + attention mechanism
# - Returns attention weights for explainability!
```

## 📈 Evaluation Metrics

Focus on these metrics (in order of importance):

1. **AUPRC** (Area Under Precision-Recall Curve)
   - More important than AUC-ROC for imbalanced data
   - Emphasizes performance on positive class

2. **AUC-ROC** (Area Under ROC Curve)
   - Overall discriminative ability

3. **Recall @ High Precision**
   - Clinical priority: catch AKI cases (high recall)
   - Without too many false alarms (precision)

4. **Calibration**
   - Are predicted probabilities accurate?

## 🔍 Explainability Analysis

After training, analyze your models:

### 1. Feature Importance (XGBoost/RF)

```python
importance = xgb_model.get_feature_importance()
# Plot top 20 features
```

### 2. SHAP Values

```python
import shap

# Global importance
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# Individual prediction
shap.force_plot(explainer.expected_value, shap_values[i], X_test[i])
```

### 3. Attention Weights (for Attention LSTM)

```python
# Extract attention weights during forward pass
predictions, attention_weights = model(x)

# Visualize which hours were most important
plt.plot(attention_weights[0].detach().cpu())
```

## 🎯 Project Extensions

Once you have a working model:

1. **Temporal Analysis**: How does prediction accuracy change with prediction horizon (24h vs 48h)?

2. **Subgroup Analysis**: Does model perform differently for:
   - Different age groups
   - Medical vs surgical patients
   - Different baseline kidney function

3. **Clinical Validation**: Compare to clinical scores (SOFA, APACHE)

4. **Intervention Analysis**: What interventions could have prevented AKI?

5. **Real-time Dashboard**: Build web app for real-time risk monitoring

## ⚠️ Important Notes

### Memory Management

MIMIC-IV is large! The code uses:
- Chunked reading for large files
- Memory reduction via dtype optimization
- Parquet format for efficient storage

If you run out of memory:
- Reduce `max_icu_stays` in config.py for testing
- Increase `chunk_size` in extraction functions
- Use a subset of features

### Data Leakage Prevention

- Split by `stay_id`, not by individual observations
- No future data in features
- Forward fill only (never backward fill)

### Class Imbalance

AKI Stage 2+ is rare (~5-15% of cases). Handle with:
- `class_weight='balanced'` in sklearn models
- `scale_pos_weight` in XGBoost
- Weighted loss functions in PyTorch
- Focus on AUPRC over accuracy

## 📚 References

1. **KDIGO Clinical Practice Guideline for AKI**  
   Kidney Disease: Improving Global Outcomes (2012)

2. **MIMIC-IV Documentation**  
   https://mimic.mit.edu/docs/iv/

3. **pyAKI Implementation**  
   https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0315325

## 🤝 Contributing

This is your project! Customize:
- Prediction windows (24h, 36h, 72h)
- Target AKI stages (Stage 1 vs 2+)
- Feature engineering approaches
- Model architectures
- Explainability methods

## 📝 License

Follow MIMIC-IV data use agreement and citation requirements.

## 🆘 Troubleshooting

**Issue**: "File not found" errors  
**Fix**: Check paths in `config.py` match your MIMIC-IV structure

**Issue**: Out of memory  
**Fix**: Set `max_icu_stays=1000` in config.py for testing

**Issue**: Slow data extraction  
**Fix**: This is normal! chartevents is huge. First run takes time.

**Issue**: Low AUC scores  
**Fix**: Check class balance, try different models, tune hyperparameters

## ✅ Success Criteria

A successful project should achieve:
- [ ] AUPRC > 0.25 (significantly better than baseline ~0.10)
- [ ] AUC-ROC > 0.75
- [ ] Clear explainability analysis
- [ ] Comparison of multiple models
- [ ] Well-documented code and results

Good luck with your project! 🎓
# GTSRB Traffic Sign Classification - XAI Case Study

**Explainable AI Final Project**  
German Traffic Sign Recognition Benchmark (GTSRB)

## Project Overview

This project implements an end-to-end Explainable AI case study for traffic sign classification using the GTSRB dataset. I train four models (two baselines and two pretrained CNNs) and apply various XAI techniques to understand model decisions and improve performance.

<div align="center">
  <a href="https://github.com/ICAI-IMAT-XAI/final-project-rdgzmanuel/raw/main/report.pdf" target="_blank">
    <img src="maxresdefault.png" alt="Project Report" width="450"/>
  </a>
  <p>Click the image to read the project report.</p>
</div>

### Key Features
- **Multiple Model Architectures**: Logistic Regression (HOG), Shallow CNN, MobileNetV2, ResNet18
- **XAI Techniques**: Grad-CAM, Integrated Gradients, Occlusion Sensitivity
- **Two-Phase Fine-tuning**: Careful fine-tuning strategy for pretrained models
- **Comprehensive Evaluation**: Model performance metrics and XAI quality assessment

## Project Structure

```
gtsrb-xai-project/
│
├── data/
│   ├── raw/                          # Raw GTSRB dataset (from Kaggle)
│   │   ├── Train/
│   │   │   ├── 0/
│   │   │   ├── 1/
│   │   │   └── ...
│   │   ├── Test.csv
│   │   └── ...
│   └── processed/                    # Preprocessed data
│       ├── train_split.pkl
│       ├── val_split.pkl
│       ├── test_split.pkl
│       ├── train_hog_features.pkl
│       ├── val_hog_features.pkl
│       └── test_hog_features.pkl
│
├── models/                           # Saved model checkpoints
│   ├── logistic_regression.pkl
│   ├── shallow_cnn.pth
│   ├── mobilenet_phase1.pth
│   ├── mobilenet_phase2.pth
│   ├── resnet18_phase1.pth
│   └── resnet18_phase2.pth
│
├── notebooks/                        # Jupyter notebooks for analysis
│   ├── test_implementation.ipynb
│   └── xai_explanations.ipynb
│
├── images/                          # Outputs and visualizations
│
├── src/                              # Source code
│   ├── data_preprocessing.py         # Data loading and preprocessing
│   ├── models.py                     # Model definitions
│   ├── train.py                      # Training pipelines
│   ├── xai_methods.py                # XAI implementations
│   └── utils.py                      # Utility functions
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── report.pdf                        # Final project report
```

## Getting Started

### Prerequisites

- Python 3.11
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd gtsrb-xai-project
```

2. **Create virtual environment (using uv package)**
```bash
uv venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

3. **Install dependencies (using uv package)**
```bash
uv pip install -r requirements.txt
```

4. **Download GTSRB dataset**
```bash
# Install Kaggle CLI
uv pip install kaggle

# Download dataset (requires Kaggle API credentials)
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

# Create data directory and extract to data/raw/
mkdir -p data/raw && unzip -o gtsrb-german-traffic-sign.zip -d data/raw/

# remove the zip file
rm gtsrb-german-traffic-sign.zip
```

### Data Preprocessing

Run the preprocessing pipeline to prepare the data:

```bash
python -m src.data_preprocessing
```

This will:
- Load the raw GTSRB images
- Create train/validation/test splits (70%/15%/15%)
- Extract HOG features for logistic regression
- Save processed data to `data/processed/`

**Expected Output:**
- Train/val/test splits saved as pickle files
- HOG features extracted and saved
- Dataset statistics printed to console

## Training Models

### Train All Models

Run the complete training pipeline:

```bash
python -m src.train
```

This trains:
1. **Logistic Regression** on HOG features (~10 min)
2. **Shallow CNN** from scratch (~30 min)
3. **MobileNetV2** with two-phase fine-tuning (~1 hour)
4. **ResNet18** with two-phase fine-tuning (~1 hour)

### Training Phases (Pretrained Models)

**Phase 1: Feature Extractor Mode (5-10 epochs)**
- Freeze all convolutional layers
- Train only the classifier head
- Higher learning rate (1e-3)

**Phase 2: Partial Fine-tuning (10-15 epochs)**
- Unfreeze last 1-2 blocks (MobileNetV2) or last residual block (ResNet18)
- Lower learning rate (1e-4)
- Fine-tune with careful regularization

### Model Checkpoints

Models are saved to `models/` directory:
- Best validation accuracy checkpoint
- Phase 1 and Phase 2 checkpoints for pretrained models

Images are saved to the `images/` directory:
- Training curves saved as PNG files

## XAI Analysis

### XAI Methods Implemented

#### 1. **Grad-CAM** (Global & Local)
- **Global**: Aggregated heatmaps across samples of the same class
- **Local**: Individual prediction explanations
- Target layer: Last convolutional layer

#### 2. **Integrated Gradients**
- Attribution to input features
- 50 integration steps
- Black image baseline

#### 3. **Occlusion Sensitivity**
- 8x8 pixel occlusion window
- 4-pixel stride
- Measures prediction change

### Running XAI Analysis

Use the provided notebooks:

```bash
jupyter notebook notebooks/xai_explanations.ipynb
```


### XAI Insights

- **Global explanations**: Identify common visual patterns per sign class
- **Local explanations**: Understand individual prediction decisions
- **Method comparison**: Spatial correlation between Grad-CAM, IG, and Occlusion

## Requirements

See `requirements.txt` for full dependencies. Key libraries:

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
opencv-python>=4.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
jupyter>=1.0.0
```

## Academic Integrity

- All external code and resources are properly cited
- Open-source libraries and pretrained models are used with attribution
- Original analysis and explanations written by project author
- Understanding of all implementation steps

## References

- GTSRB Dataset: [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- Integrated Gradients: Sundararajan et al., "Axiomatic Attribution for Deep Networks"
- MobileNetV2: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
- ResNet: He et al., "Deep Residual Learning for Image Recognition"

## Contact

For questions or issues, please contact manuel.rodriguezvillegas09@gmail.com

---

**Last Updated**: December 2025

**Course**: Ética y Explicabilidad de la Inteligencia Artificial (Ethics and Explainability of Artificial Intelligence)

**Academic Year**: 2025/2026
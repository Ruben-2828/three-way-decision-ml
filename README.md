# Three-Way Decision Analysis

This project implements three-way decision (3WD) analysis on binary classification datasets using machine learning models. The three-way decision approach allows models to make three types of decisions:
- **Accept** (classify as positive with high confidence)
- **Reject** (classify as negative with high confidence) 
- **Defer** (abstain from making a decision when uncertain)

## Table of Contents

- [Three-Way Decision Analysis](#three-way-decision-analysis)
- [Datasets Used](#datasets-used)
- [Key Features](#key-features)
   - [Three-Way Decision Parameters](#three-way-decision-parameters)
   - [Models Evaluated](#models-evaluated)
   - [Analysis Components](#analysis-components)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## Datasets Used
Two publicly available datasets are used for analysis to test different scenarios of class imbalance and feature types.

### 1. Bank Marketing Dataset
- **Source**: [UCI Machine Learning Repository (ID: 222)](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Size**: 45,211 samples
- **Features**: 16 original features (38 after one-hot encoding)
- **Target**: Binary classification (client subscription to term deposit)
- **Imbalance**: Highly imbalanced (~88% negative, ~12% positive)

### 2. Adult Census Income Dataset
- **Source**: [UCI Machine Learning Repository (Adult)](https://archive.ics.uci.edu/ml/datasets/adult)
- **Size**: 32,561 original samples → **14,000 balanced samples**
- **Features**: 14 original features (99 after one-hot encoding)
- **Target**: Binary classification (income >50K vs ≤50K)
- **Balancing**: Automatically creates a balanced subset with 7,000 samples per class

## Key Features

### Three-Way Decision Parameters
- **Alpha (α)**: List of thresholds for accepting positive classification (default: `[0.70, 0.80, 0.90]`)
- **Beta (β)**: List of thresholds for accepting negative classification (default: `[0.30, 0.20, 0.10]`)
- **Deferral Zone**: For each (α, β) pair, predictions with probability between β and α are deferred (i.e., `β < probability < α`)

### Models Evaluated
1. **Random Forest**: 200 estimators, optimized for performance
2. **Multi-Layer Perceptron (MLP)**: Neural network with hidden layers (128, 64)

### Analysis Components
1. **Binary Classification Results**: Standard accuracy, confusion matrices, classification reports
2. **Three-Way Decision Results**: Coverage, accuracy on certain cases, deferral rates
3. **ROC Curves**: With operating points for different α values
4. **Precision-Recall Curves**: For both positive and negative classes
5. **Calibration Analysis**: Probability calibration with Brier scores
6. **Coverage-Accuracy Trade-off**: Systematic sweep of α and β parameters
7. **Probability Histograms**: Distribution of predicted probabilities with decision thresholds

## Installation 

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/three-way-decision-analysis.git
   cd three-way-decision-analysis
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not present, install the main dependencies manually:
   ```bash
   pip install scikit-learn numpy pandas matplotlib ucimlrepo
   ```

## Usage

   **Run the main analysis script**
   ```bash
   python main.py
   ```

   This will:
   - Download and preprocess the datasets
   - Train Random Forest and MLP models
   - Perform three-way decision analysis
   - Generate all plots and CSV reports in the `outputs/` directory
   - All output files (plots, confusion matrices, calibration diagrams, etc.) will be saved under `outputs/` in subfolders for each dataset.


## Project Structure 

```
three-way-decision-analysis/
├── utils/
│   └── functions.py          # Helper functions
├── .gitignore                # Git ignore file
├── LICENSE                   # MIT License file
├── main.py                   # Main script to start the execution of the analysis
├── README.md                 # This file
└── requirements.txt          # Python requirements for the project
```

## License

This project is open source and available under the MIT License.

## Contact
Ruben Tenderini — GitHub: [Ruben-2828](https://github.com/Ruben-2828) — Email: [rubentenderini@gmail.com](mailto:rubentenderini@gmail.com)
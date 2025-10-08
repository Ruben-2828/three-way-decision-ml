# Three-Way Decision Analysis

This project implements three-way decision (3WD) analysis on binary classification datasets using machine learning models. The three-way decision approach allows models to make three types of decisions:
- **Accept** (classify as positive with high confidence)
- **Reject** (classify as negative with high confidence) 
- **Defer** (abstain from making a decision when uncertain)

## Datasets Supported

### 1. Bank Marketing Dataset
- **Source**: UCI Machine Learning Repository (ID: 222)
- **Size**: 45,211 samples
- **Features**: 16 original features (38 after one-hot encoding)
- **Target**: Binary classification (client subscription to term deposit)
- **Imbalance**: Highly imbalanced (~88% negative, ~12% positive)

### 2. Adult Census Income Dataset
- **Source**: UCI Machine Learning Repository
- **Size**: 32,561 original samples → **12,000 balanced samples**
- **Features**: 14 original features (99 after one-hot encoding)
- **Target**: Binary classification (income >50K vs ≤50K)
- **Balancing**: Automatically creates a balanced subset with 6,000 samples per class

## Key Features

### Three-Way Decision Parameters
- **Alpha (α)**: 0.8 - Threshold for accepting positive classification
- **Beta (β)**: 0.2 - Threshold for accepting negative classification
- **Deferral Zone**: 0.2 < probability < 0.8

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

## Usage

### Quick Start
```bash
# Run analysis on Bank Marketing dataset
python run_analysis.py bank

# Run analysis on Adult Census Income dataset  
python run_analysis.py adult
```

### Manual Usage
```bash
# Edit main.py and change DATASET_CHOICE variable
# Then run:
python main.py
```

## Output Structure

All outputs are saved in dataset-specific directories under `outputs/`:

```
outputs/
├── run_bank_marketing/
│   ├── coverage_accuracy_rf_mlp.png
│   ├── rf_calibration.png
│   ├── rf_confusion_matrices.png
│   ├── rf_prob_hist.png
│   ├── mlp_calibration.png
│   ├── mlp_confusion_matrices.png
│   ├── mlp_prob_hist.png
│   ├── pr_3wd/
│   │   ├── rf_pr_multi_3wd.png
│   │   ├── rf_pr_negative_beta.png
│   │   ├── mlp_pr_multi_3wd.png
│   │   └── mlp_pr_negative_beta.png
│   └── roc_3wd/
│       ├── rf_roc_positive.png
│       ├── rf_roc_negative.png
│       ├── mlp_roc_positive.png
│       └── mlp_roc_negative.png
└── run_adult_census_income/
    └── (same structure as above)
```

## Key Results Interpretation

### Coverage vs Accuracy Trade-off
- **Coverage**: Percentage of samples that receive certain decisions (not deferred)
- **Higher Coverage**: More decisions made, potentially lower accuracy
- **Lower Coverage**: Fewer but more confident decisions, higher accuracy on certain cases

### Deferral Analysis
- Samples in the deferral zone (0.2 < p < 0.8) are not classified
- This allows for human review or additional data collection
- Particularly useful in high-stakes decisions where errors are costly

### Model Comparison
- Random Forest typically shows more conservative deferral patterns
- MLP often achieves higher accuracy on certain cases but may defer more samples
- Both models show improved accuracy when only considering non-deferred cases

## Requirements

```
scikit-learn>=1.7.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.5.0
ucimlrepo>=0.0.7
```

## Installation

```bash
pip install scikit-learn pandas numpy matplotlib ucimlrepo
```

## Dataset Details

### Adult Census Income Balancing Process
1. Load original dataset (32,561 samples)
2. Remove samples with missing values
3. Identify majority class (≤50K) and minority class (>50K)
4. Randomly sample 6,000 samples from each class
5. Combine and shuffle to create balanced 12,000 sample dataset
6. Apply one-hot encoding to categorical features

### Feature Engineering
- **One-hot encoding** applied to all categorical variables
- **Standard scaling** applied to neural network inputs
- **Stratified splitting** ensures balanced train/test sets

## Research Applications

This implementation is suitable for:
- **Medical diagnosis**: Where false negatives are critical
- **Financial risk assessment**: Where high-confidence decisions are required
- **Quality control**: Where uncertain cases need human review
- **Fraud detection**: Where deferring uncertain cases is preferable to errors

## Citation

If you use this code in your research, please cite the original datasets:

- **Bank Marketing**: Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31.
- **Adult Census Income**: Kohavi, R. (1996). Scaling up the accuracy of naive-bayes classifiers: A decision-tree hybrid. In Proceedings of the second international conference on knowledge discovery and data mining (pp. 202-207).

## License

This project is open source and available under the MIT License.

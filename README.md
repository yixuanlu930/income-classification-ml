# Income Classification ML

An end-to-end supervised machine learning project for predicting whether an individual's income belongs to the **`<=50K` or `>50K`** category using demographic, occupational, and economic information.

The project covers the complete machine learning workflow:

- Data cleaning
- Missing-value imputation
- Outlier treatment
- One-Hot Encoding
- Class-imbalance analysis
- SMOTE oversampling
- Feature selection with RFECV
- Feature scaling
- Multi-model comparison
- Hyperparameter optimization
- Cross-validation
- Model interpretation
- Final prediction generation

Several supervised-learning algorithms are compared, with **Random Forest** achieving the strongest validation F1-score and being selected as the final model.

---

## Overview

The objective is to solve a binary classification problem:

```text
<=50K
>50K
```

using attributes related to:

- Age
- Employment
- Education
- Marital status
- Occupation
- Family position
- Ethnicity
- Sex
- Investment gains
- Investment losses
- Weekly working hours
- Country of origin

The complete workflow is divided across three notebooks:

```text
preprocess.ipynb
        ↓
supervised.ipynb
        ↓
best_model.ipynb
```

---

# Dataset

The project uses two main datasets.

## Training Dataset

```text
data/salario.csv
```

contains:

```text
27,998 observations
12 predictive variables
1 target variable
```

The target column is:

```text
salario
```

with the distribution:

```text
<=50K    21,290
>50K      6,708
```

This corresponds approximately to:

```text
76% majority class
24% minority class
```

which motivates the use of imbalance-handling techniques.

---

## External Test Dataset

```text
data/salario_test.csv
```

contains:

```text
4,563 observations
```

and does not contain the target during the prediction stage.

Each sample has an:

```text
ID
```

used to generate the final submission.

---

# Project Pipeline

The complete workflow can be summarized as:

```text
Raw data
   │
   ▼
Cleaning
   │
   ▼
Missing-value imputation
   │
   ▼
Train / Validation split
   │
   ▼
Winsorization
   │
   ▼
One-Hot Encoding
   │
   ▼
Class imbalance analysis
   │
   ▼
SMOTE
   │
   ▼
Feature Selection
RFECV + Logistic Regression
   │
   ▼
Multiple Model Training
   │
   ▼
GridSearchCV
   │
   ▼
Validation comparison
   │
   ▼
Random Forest
   │
   ▼
Final refit
   │
   ▼
External predictions
```

---

# 1. Data Cleaning

The preprocessing stage is implemented in:

```text
preprocess.ipynb
```

Missing categorical values are identified and replaced using the mode of the corresponding training feature.

The target labels are converted from:

```text
<=50K
>50K
```

to numerical values:

```text
0
1
```

using:

```python
LabelEncoder
```

---

# Train / Validation Split

The labeled dataset is divided using a stratified split:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This preserves the original class distribution.

An important design decision is that preprocessing transformations are fitted **only on the training data**.

This avoids:

```text
data leakage
```

from the validation set.

---

# Outlier Treatment

Several numerical features contain highly skewed distributions and extreme values.

Instead of removing observations, the project applies **Winsorization** using approximately the:

```text
5th percentile
95th percentile
```

Values outside these limits are capped.

This reduces the effect of extreme values while preserving all observations.

The Winsorization thresholds are calculated using training data only and then applied to validation and external test data.

---

# Categorical Encoding

Categorical variables are converted using:

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

Using:

```text
handle_unknown="ignore"
```

allows the pipeline to safely process categories that may appear in the external test dataset but were not observed during training.

---

# Class Imbalance

The training dataset contains approximately:

```text
76% <=50K
24% >50K
```

Since the positive class is underrepresented, the project evaluates oversampling techniques.

The final pipeline uses:

```python
SMOTE
```

to create synthetic minority-class examples.

SMOTE is applied **only to training data**, never to validation or external test data.

This prevents artificially modifying the evaluation distribution.

---

# Feature Selection

A broad comparison of feature-selection approaches is performed.

Methods explored include:

- Variance Threshold
- Chi-square
- ANOVA F-test
- Mutual Information
- RFE
- RFECV
- L1 Logistic Regression
- Linear SVC
- Random Forest importance
- Gradient Boosting importance

The selected strategy is:

```text
RFECV
+
Logistic Regression
```

using:

```text
F1-score
```

as the optimization metric.

---

## Why RFECV?

RFECV recursively removes features and evaluates the resulting subsets using cross-validation.

Conceptually:

```text
All features
     │
     ▼
Train model
     │
     ▼
Remove least useful features
     │
     ▼
Cross-validation
     │
     ▼
Repeat
     │
     ▼
Best feature subset
```

This is particularly useful after One-Hot Encoding, where the number of features becomes substantially larger.

---

# Feature Scaling

Different model families require different preprocessing.

The project generates several versions of the data:

```text
Unscaled
StandardScaler
MinMaxScaler
```

Tree-based models do not require feature scaling.

Distance-based and gradient-based models benefit significantly from standardized or normalized inputs.

---

# 2. Supervised Model Comparison

The main model comparison is implemented in:

```text
supervised.ipynb
```

Several algorithms are evaluated.

---

## Linear SVM

Implemented using:

```python
SGDClassifier(loss="hinge")
```

---

## Logistic Regression

A classical linear probabilistic classifier.

The project evaluates different hyperparameters and imbalance-handling configurations.

---

## SVM with RBF Kernel

```python
SVC(kernel="rbf")
```

allows nonlinear decision boundaries.

---

## Polynomial Logistic Regression

Polynomial features are generated for selected numerical variables before applying Logistic Regression.

This allows the linear model to capture nonlinear interactions.

---

## K-Nearest Neighbors

```python
KNeighborsClassifier
```

is evaluated on scaled data.

Its performance is negatively affected by the high-dimensional feature space created by One-Hot Encoding.

---

## Decision Tree

```python
DecisionTreeClassifier
```

provides nonlinear classification and direct interpretability.

---

## Random Forest

```python
RandomForestClassifier
```

combines multiple decision trees to improve robustness and generalization.

It ultimately produces the strongest F1-score.

---

## Neural Network — Scikit-learn

The project includes:

```python
MLPClassifier
```

with hyperparameter tuning and early stopping.

---

## Neural Network — Keras

A second neural-network implementation is built with:

```text
TensorFlow
Keras
SciKeras
```

SciKeras allows the Keras network to participate in the same Scikit-learn model-selection infrastructure.

The neural network includes:

- Fully connected layers
- Dropout
- L2 regularization

and is optimized through `GridSearchCV`.

---

# Grid Search and Cross-Validation

Models are placed inside `imblearn` pipelines.

For example:

```text
Preprocessing
     │
     ▼
SMOTE
     │
     ▼
Classifier
```

Hyperparameters are optimized using:

```python
GridSearchCV
```

with stratified cross-validation.

During the large multi-model comparison:

```text
CV = 3
```

is used to reduce computational cost.

The final Random Forest optimization uses:

```text
CV = 5
```

for a more robust estimate.

---

# Evaluation Metrics

Because the dataset is imbalanced, model comparison is not based only on accuracy.

The project calculates:

- F1-score
- Precision
- Recall
- Accuracy
- Matthews Correlation Coefficient
- ROC-AUC
- PR-AUC

The primary metric is:

```text
F1-score
```

because it balances:

```text
Precision
   +
Recall
```

for the minority class.

---

# Model Comparison

Representative validation F1-scores from the multi-model experiment are:

| Model | Validation F1 |
|---|---:|
| Random Forest | **0.7014** |
| SVM RBF | 0.6937 |
| Polynomial Logistic Regression | 0.6935 |
| MLP | ~0.6896 |
| Keras MLP | ~0.6869 |
| Logistic Regression | ~0.6751 |
| Decision Tree | ~0.6734 |
| Linear SVM | ~0.6600 |
| KNN | ~0.6358 |

Random Forest achieves the strongest balance between precision and recall.

---

# 3. Final Random Forest

The final implementation is contained in:

```text
best_model.ipynb
```

The optimized classifier uses:

```text
Random Forest
+
RFECV feature subset
+
SMOTE
```

A saved configuration included:

```text
n_estimators = 100
max_depth = 16
min_samples_leaf = 5
class_weight = {0: 1.25, 1: 1.0}
SMOTE k_neighbors = 3
```

---

# Final Validation Metrics

The saved final model reports approximately:

| Metric | Score |
|---|---:|
| F1 | **0.699** |
| Accuracy | **0.824** |
| Precision | **0.593** |
| Recall | **0.852** |
| ROC-AUC | **0.914** |
| PR-AUC | **0.776** |

The high recall is especially useful because the goal is to correctly identify as many individuals from the minority `>50K` class as possible while maintaining acceptable precision.

---

# External Competition Result

After selecting the best configuration, the Random Forest is retrained using the available training and validation information.

Predictions are generated for:

```text
salario_test.csv
```

and exported to:

```text
test_labels.csv
```

The project report records an official public competition score of approximately:

```text
F1 = 0.832
```

on the external evaluation set.

---

# Feature Importance

Random Forest feature importance is used to interpret the final classifier.

The analysis indicates that several sociodemographic variables are particularly influential, including features associated with:

- Marital status
- Family position
- Occupation
- Education
- Age
- Working hours

Feature importance should be interpreted as predictive relevance rather than causal influence.

---

# Confusion Matrix

The final model is also evaluated through a confusion matrix.

This allows direct inspection of:

```text
True Negatives
False Positives
False Negatives
True Positives
```

The selected Random Forest favors a relatively strong recall for the minority class while maintaining sufficient precision to maximize F1.

---

# Saved Model

The final estimator is stored in:

```text
best_model/best_model.pkl
```

Additional metadata is available in:

```text
best_model/best_model_info.json
```

and the corresponding search results are stored in:

```text
best_model/model_search_results.csv
```

This allows the final experiment to be inspected without rerunning the entire model comparison.

---

# Processed Dataset Export

The preprocessing notebook generates:

```text
processed_dataset.csv.zip
```

containing several versions of the dataset, including:

```text
OHE datasets
RFECV-selected datasets
Standard-scaled datasets
MinMax-scaled datasets
```

This separates the expensive preprocessing stage from subsequent model experiments.

---

# Project Structure

```text
income-classification-ml/
│
├── data/
│   ├── salario.csv
│   └── salario_test.csv
│
├── best_model/
│   ├── best_model.pkl
│   ├── best_model_info.json
│   └── model_search_results.csv
│
├── preprocess.ipynb
├── supervised.ipynb
├── best_model.ipynb
│
├── processed_dataset.csv.zip
├── test_labels.csv
├── memoria.pdf
├── README.md
└── .gitignore
```

---

# Notebook Workflow

## `preprocess.ipynb`

Contains:

- Data loading
- Cleaning
- Missing-value handling
- Target encoding
- Stratified splitting
- Winsorization
- One-Hot Encoding
- Class-imbalance analysis
- SMOTE
- Feature-selection comparison
- RFECV
- Scaling
- Processed dataset export

---

## `supervised.ipynb`

Contains:

- Import of processed datasets
- Model pipelines
- Grid Search
- Cross-validation
- Multi-model comparison
- Neural-network experiments
- Confusion matrices
- ROC curves
- Precision–Recall curves
- Feature importance
- Model ranking

---

## `best_model.ipynb`

Contains:

- Final preprocessing reconstruction
- RFECV feature selection
- Random Forest optimization
- Five-fold cross-validation
- Final model training
- Validation analysis
- External prediction
- Model serialization

---

# Installation

A Python environment with Jupyter is recommended.

Install the main dependencies:

```bash
pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    imbalanced-learn \
    tensorflow \
    keras \
    scikeras \
    jupyter \
    joblib
```

---

# Running the Project

Run the notebooks in the following order:

```text
1. preprocess.ipynb
2. supervised.ipynb
3. best_model.ipynb
```

Start Jupyter with:

```bash
jupyter notebook
```

---

# Technologies

## Machine Learning

- Scikit-learn
- Imbalanced-learn
- TensorFlow
- Keras
- SciKeras

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

## Development

- Python
- Jupyter Notebook
- Joblib

---

# Key Concepts

This project demonstrates:

- Supervised Learning
- Binary Classification
- Imbalanced Classification
- Data Cleaning
- Missing-Value Imputation
- Winsorization
- One-Hot Encoding
- Feature Scaling
- SMOTE
- Feature Selection
- RFE
- RFECV
- Cross-Validation
- Grid Search
- Logistic Regression
- Support Vector Machines
- KNN
- Decision Trees
- Random Forests
- Neural Networks
- F1-score
- ROC-AUC
- PR-AUC
- Feature Importance
- Model Serialization

---

# Main Findings

The experiments highlight several important observations.

### F1 is more informative than accuracy

Because approximately three quarters of the observations belong to the majority class, high accuracy alone does not necessarily imply good minority-class detection.

### SMOTE improves minority-class learning

Synthetic oversampling allows the classifiers to learn from a more balanced training distribution without discarding majority-class information.

### Feature selection helps control dimensionality

One-Hot Encoding creates a relatively large feature space.

RFECV identifies a smaller subset optimized for F1-score.

### Random Forest provides the best overall balance

The final model captures nonlinear interactions in the tabular data while remaining robust to feature scale and achieving the strongest validation F1-score.

### KNN struggles with the high-dimensional representation

The distance-based nature of KNN makes it less effective after extensive One-Hot Encoding.

---

# Academic Context

This project was developed for the **Machine Learning I** course as a supervised-learning classification assignment.

Authors:

- Yixuan Lu Guo
- Jennifer Juez Yanguas

The project report was completed in November 2025.

---

# Disclaimer

This project is intended for educational and experimental purposes.

The predicted income category represents a statistical classification based on the available dataset and should not be interpreted as a causal assessment of individual socioeconomic outcomes.

---

# License

See the repository license for applicable terms.

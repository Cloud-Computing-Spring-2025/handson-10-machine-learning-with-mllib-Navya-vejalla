# handson-10-MachineLearning-with-MLlib.

# Customer Churn Prediction with MLlib

This project uses Apache Spark MLlib to predict customer churn based on structured customer data. You will preprocess data, train classification models, perform feature selection, and tune hyperparameters using cross-validation.

---

Build and compare machine learning models using PySpark to predict whether a customer will churn based on their service usage and subscription features.

---

## Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

---

## Tasks

### Task 1: Data Preprocessing and Feature Engineering

**Objective:**  
Clean the dataset and prepare features for ML algorithms.

**Steps:**
1. Fill missing values in `TotalCharges` with 0.
2. Encode categorical features using `StringIndexer` and `OneHotEncoder`.
3. Assemble numeric and encoded features into a single feature vector with `VectorAssembler`.
4. Save first few rows of processed features into `outputs/task1_features.txt`.

**Output:**
```bash
+--------------------+-----------+
|features            |label      |
+--------------------+-----------+
|[0.0,21.0,94.49,2120.34,1.0,1.0,1.0,0.0]|0.0        |
|[1.0,3.0,83.8,225.05,1.0,1.0,0.0,1.0]|0.0        |
|[0.0,2.0,73.21,149.8,1.0,0.0,0.0,1.0]|1.0        |
|[1.0,21.0,42.56,885.5,1.0,0.0,0.0,1.0]|1.0        |
|[1.0,4.0,82.21,335.05,0.0,0.0,0.0,1.0]|1.0        |
+--------------------+-----------+
```
### Task 2: Train and Evaluate Logistic Regression Model

**Objective:**  
Train a logistic regression model and evaluate it using AUC (Area Under ROC Curve).

**Steps:**
1. Split dataset into training and test sets (80/20).
2. Train a logistic regression model.
3. Use `BinaryClassificationEvaluator` to evaluate.
4. Save evaluation results to `outputs/task2_logistic_auc.txt` and predictions to `outputs/task2_predictions.txt`.

**Output:**
```bash
Logistic Regression Model Accuracy (AUC): 0.77
```
### Task 3: Feature Selection using Chi-Square Test

**Objective:**  
Select the top 5 most important features using Chi-Square feature selection.

**Steps:**
1. Use `ChiSqSelector` to rank and select top 5 features.
2. Save selected features and label sample to `outputs/task3_selected_features.txt`.

**Output:**
```bash
+--------------------+-----------+
|selectedFeatures    |label      |
+--------------------+-----------+
|[0.0,21.0,2120.34,1.0,0.0]|0.0        |
|[1.0,3.0,225.05,0.0,1.0]|0.0        |
|[0.0,2.0,149.8,0.0,1.0]|1.0        |
|[1.0,21.0,885.5,0.0,1.0]|1.0        |
|[1.0,4.0,335.05,0.0,1.0]|1.0        |
+--------------------+-----------+
```
### Task 4: Hyperparameter Tuning and Model Comparison

**Objective:**  
Use CrossValidator to tune models and compare their AUC performance.

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosted Trees (GBT)

**Steps:**
1. Define models and parameter grids.
2. Use `CrossValidator` for 5-fold cross-validation.
3. Evaluate and print best model results.
4. Save results to `outputs/task4_model_comparison.txt` and model-specific predictions in respective files.


**Output:**
```bash
Tuning LogisticRegression...
LogisticRegression Best Model Accuracy (AUC): 0.76

Tuning DecisionTree...
DecisionTree Best Model Accuracy (AUC): 0.81

```
---

## Execution Instructions

### 1. Prerequisites
- Apache Spark installed
- Python environment with `pyspark` installed
- `customer_churn.csv` placed in the project directory

### 2. Run the Project

Run the main pipeline using:
```bash
spark-submit customer_churn_mllib.py
```

### 3. Output Files
All task results will be saved in the `outputs/` folder for easy access:
- `outputs/task1_features.txt`
- `outputs/task2_logistic_auc.txt`
- `outputs/task2_predictions.txt`
- `outputs/task3_selected_features.txt`
- `outputs/task4_model_comparison.txt`
- `outputs/task4_RandomForest_predictions.txt`

# Parkinson-Disease-Prediction-using-Machine-Learning

# 🧠 Parkinson's Disease Classification using Machine Learning

This project implements a complete machine learning pipeline to classify Parkinson’s Disease based on biomedical voice measurements. It includes preprocessing, correlation filtering, feature selection, class imbalance handling, and model training using Logistic Regression, XGBoost, and SVM.

---

## 📁 Dataset

- **Filename:** `parkinson_disease.csv`
- **Columns:**
  - `id`: Unique identifier for each subject
  - `class`: Target label (1 = Parkinson’s, 0 = Healthy)
  - Several voice-related features (e.g., jitter, shimmer, NHR, etc.)

> Each patient may have multiple voice recordings. The data is aggregated per patient by taking the average of their recordings.

---

## 📌 Pipeline Overview

### 🔹 Step 1: Data Loading & Preprocessing
- Loads the dataset using pandas
- Aggregates multiple voice recordings for the same patient by computing the mean
- Drops the `id` column

### 🔹 Step 2: Feature Correlation Filtering
- Removes highly correlated features (Pearson correlation > 0.7) to reduce redundancy and multicollinearity

### 🔹 Step 3: Feature Selection
- Scales features using `MinMaxScaler`
- Applies Chi-Square (`chi2`) test via `SelectKBest` to choose the top 30 features most relevant to the target variable

### 🔹 Step 4: Handling Class Imbalance
- Applies `RandomOverSampler` to balance the training data for better generalization on minority classes

### 🔹 Step 5: Model Training
Three machine learning models are trained and evaluated:
- **Logistic Regression** (with `class_weight='balanced'`)
- **XGBoost Classifier** (with `eval_metric='logloss'`)
- **Support Vector Classifier (SVC)** (with RBF kernel)

### 🔹 Step 6: Evaluation
- Evaluates all models using ROC AUC score on both training and validation sets
- Displays a confusion matrix for Logistic Regression
- Shows a classification report (Precision, Recall, F1 Score)

---

## 📊 Visualizations

- 🥧 Class distribution pie chart (before training)
- 📉 Confusion Matrix (Logistic Regression)

---

## 🧪 Models Used

| Model                | Description |
|----------------------|-------------|
| **Logistic Regression**  | Linear model with class weight balancing |
| **XGBoost Classifier**   | Gradient boosting model with strong performance |
| **Support Vector Classifier** | Non-linear classifier using RBF kernel |

---

## 📦 Requirements

Install required dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost

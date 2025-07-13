import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('parkinson_disease.csv')
pd.set_option('display.max_columns', 10)
print(df.sample(5))

df = df.groupby('id').mean().reset_index()
df.drop('id', axis=1, inplace=True)


columns = list(df.columns)
columns.remove('class')
filtered_columns = ['class']

for col in columns:
    keep = True
    for kept_col in filtered_columns:
        if kept_col == 'class':
            continue
        if abs(df[col].corr(df[kept_col])) > 0.7:
            keep = False
            break
    if keep:
        filtered_columns.append(col)

df = df[filtered_columns]
print("Shape after removing correlated features:", df.shape)


X = df.drop('class', axis=1)
y = df['class']
X_norm = MinMaxScaler().fit_transform(X)

selector = SelectKBest(chi2, k=30)
X_selected = selector.fit_transform(X_norm, y)
selected_columns = X.columns[selector.get_support()]

df_selected = pd.DataFrame(X_selected, columns=selected_columns)
df_selected['class'] = y.values
df = df_selected

print("Shape after feature selection:", df.shape)


x = df['class'].value_counts()
plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
plt.title('Class Distribution')
plt.show()

features = df.drop('class', axis=1)
target = df['class']
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=10)


ros = RandomOverSampler(sampling_strategy=1.0, random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
print("Resampled training set shape:", X_resampled.shape)
print("Resampled class distribution:\n", pd.Series(y_resampled).value_counts())


models = [
    LogisticRegression(class_weight='balanced', random_state=0),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0),
    SVC(kernel='rbf', probability=True, random_state=0)
]

for model in models:
    model.fit(X_resampled, y_resampled)
    print(f'{model.__class__.__name__}:')
    train_preds = model.predict(X_resampled)
    print('Training ROC AUC Score:', roc_auc_score(y_resampled, train_preds))
    val_preds = model.predict(X_val)
    print('Validation ROC AUC Score:', roc_auc_score(y_val, val_preds))
    print()


ConfusionMatrixDisplay.from_estimator(models[0], X_val, y_val)
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()

# Classification report
print("Classification Report (Logistic Regression):")
print(classification_report(y_val, models[0].predict(X_val)))

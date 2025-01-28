import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC

# Load data
data = pd.read_csv('GBV_survey.csv')

# Data Cleaning
data = data.dropna()
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Descriptive Statistics
summary = data.describe(include='all')
print("Data Summary:")
print(summary)

# Visualize Distribution of Key Variables
plt.figure(figsize=(12, 6))
for i, col in enumerate(data.select_dtypes(include=['int64', 'float64']).columns[:6]):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data[col], kde=True, bins=20, color='blue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Correlation Matrix
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = data[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Classification: Binary Questions
binary_questions = [col for col in data.columns if data[col].nunique() == 2]
classification_results = {}
for col in binary_questions:
    X = data.drop(binary_questions, axis=1)
    y = data[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print(f"Classification Report for {col} (Logistic Regression):")
    print(classification_report(y_test, y_pred))

    # Save results
    classification_results[col] = {
        'logistic_regression': classification_report(y_test, y_pred, output_dict=True)
    }

# Random Forest Classification for Multiclass Targets
for col in data.columns:
    if data[col].nunique() > 2 and data[col].nunique() <= 10:
        X = data.drop(col, axis=1)
        y = data[col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print(f"Classification Report for {col} (Random Forest):")
        print(classification_report(y_test, y_pred))

        classification_results[col] = {
            'random_forest': classification_report(y_test, y_pred, output_dict=True)
        }

# Gradient Boosting Classifier
for col in binary_questions:
    X = data.drop(binary_questions, axis=1)
    y = data[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    print(f"Classification Report for {col} (Gradient Boosting):")
    print(classification_report(y_test, y_pred))

    classification_results[col]['gradient_boosting'] = classification_report(y_test, y_pred, output_dict=True)

# SVM for Multiclass Classification
for col in data.columns:
    if data[col].nunique() > 2 and data[col].nunique() <= 5:
        X = data.drop(col, axis=1)
        y = data[col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        svm = SVC(probability=True, random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        print(f"Classification Report for {col} (SVM):")
        print(classification_report(y_test, y_pred))

        classification_results[col]['svm'] = classification_report(y_test, y_pred, output_dict=True)

# Recommendations
print("Recommendations:")
print("1. Focus on binary questions where logistic regression and gradient boosting show high accuracy.")
print("2. Leverage Random Forest and SVM for multiclass targets to understand key patterns.")
print("3. Use correlation matrix insights to identify and address multicollinearity in features.")

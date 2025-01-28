import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import pandas as pd


# Load the dataset
data_path = "pupil_data_cleaned.csv"
data = pd.read_csv(data_path)


# Convert categorical variables for visualization
categorical_vars = ["parental_education_level", "access_to_resources", "school"]
for var in categorical_vars:
    data[var] = data[var].astype('category')

# Function to create visualizations
def plot_correlation_matrix(df, title):
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.show()

# Correlation Analysis
plot_correlation_matrix(data, "Correlation Between Variables")

# Distribution of Exam Results by School
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="school", y="exam_results", palette="Set3")
plt.title("Distribution of Exam Results by School")
plt.show()

# Behavior vs Exam Results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="behaviour", y="exam_results", hue="school", palette="Set1")
plt.title("Behaviour vs Exam Results")
plt.show()

# Teacher Recommendation vs Exam Results
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="teacher_recommendation", y="exam_results", palette="coolwarm")
plt.title("Teacher Recommendation vs Exam Results")
plt.show()

# Linear Regression: Exam Results vs Behaviour
X = data[["behaviour"]]
y = data["exam_results"]

lin_reg = LinearRegression()
lin_reg.fit(X, y)

y_pred = lin_reg.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(data["behaviour"], data["exam_results"], alpha=0.5, label="Actual Data")
plt.plot(data["behaviour"], y_pred, color='red', label="Linear Fit")
plt.title("Linear Regression: Behaviour vs Exam Results")
plt.xlabel("Behaviour")
plt.ylabel("Exam Results")
plt.legend()
plt.show()

print(f"Linear Regression Coefficient: {lin_reg.coef_[0]:.2f}")
print(f"Linear Regression Intercept: {lin_reg.intercept_:.2f}")

# Logistic Regression for Teacher Recommendation
X = data[["exam_results", "behaviour", "income"]]
y = data["teacher_recommendation"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: Logistic Regression")
plt.show()

# K-Nearest Neighbors for Teacher Recommendation
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("K-Nearest Neighbors Report:")
print(classification_report(y_test, y_pred_knn))

# Random Forest for Teacher Recommendation
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# Visualizing Feature Importances from Random Forest
importances = rf.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importance from Random Forest")
plt.show()

# Decision Tree for Teacher Recommendation
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("Decision Tree Report:")
print(classification_report(y_test, y_pred_dt))

# Correlation of Income with Average House Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="income", y="average_house_price", hue="school", palette="coolwarm")
plt.title("Income vs Average House Price by School")
plt.show()

correlation, _ = pearsonr(data["income"], data["average_house_price"])
print(f"Correlation between Income and Average House Price: {correlation:.2f}")

# Behavior by Socioeconomic Resources
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="access_to_resources", y="behaviour", palette="Set2")
plt.title("Behavior by Access to Resources")
plt.show()

# Exam Results by Parental Education Level
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="parental_education_level", y="exam_results", palette="Set1")
plt.title("Exam Results by Parental Education Level")
plt.show()

# Exam Results and Behavior by Teacher Recommendation
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x="exam_results", y="behaviour", hue="teacher_recommendation", style="school", palette="Set2")
plt.title("Exam Results and Behavior by Teacher Recommendation")
plt.show()

# Distribution of Income by School
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="school", y="income", palette="coolwarm")
plt.title("Distribution of Income by School")
plt.show()

# Analyze the relationship between disciplinary record and teacher recommendation
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="teacher_recommendation", y="disciplinary_record", palette="coolwarm")
plt.title("Disciplinary Record vs Teacher Recommendation")
plt.show()

# Regression Analysis: Exam Results vs Average House Price
X = data[["average_house_price"]]
y = data["exam_results"]

lin_reg_hp = LinearRegression()
lin_reg_hp.fit(X, y)

y_pred_hp = lin_reg_hp.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(data["average_house_price"], data["exam_results"], alpha=0.5, label="Actual Data")
plt.plot(data["average_house_price"], y_pred_hp, color='red', label="Linear Fit")
plt.title("Linear Regression: Average House Price vs Exam Results")
plt.xlabel("Average House Price")
plt.ylabel("Exam Results")
plt.legend()
plt.show()

print(f"Linear Regression Coefficient: {lin_reg_hp.coef_[0]:.2f}")
print(f"Linear Regression Intercept: {lin_reg_hp.intercept_:.2f}")

from sklearn.ensemble import RandomForestRegressor

# Feature Importance for Exam Results using Random Forest (Regressor)
X_features = data[["income", "behaviour", "access_to_resources", "parental_education_level"]]
y_results = data["exam_results"]

# Use RandomForestRegressor for continuous target
rf_exam = RandomForestRegressor(random_state=42)
rf_exam.fit(X_features, y_results)

# Get feature importances
feature_importances_exam = rf_exam.feature_importances_
feature_names_exam = X_features.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_exam, y=feature_names_exam, palette="viridis")
plt.title("Feature Importance for Exam Results")
plt.show()


# Analyze progression targets vs exam results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="progression_targets", y="exam_results", hue="teacher_recommendation", palette="coolwarm")
plt.title("Progression Targets vs Exam Results")
plt.show()

# Clustering Analysis for Exam Results and Behavior
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[["exam_results", "behaviour"]])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="exam_results", y="behaviour", hue="Cluster", palette="viridis")
plt.title("Clustering: Exam Results and Behavior")
plt.show()

# Summarize the clustering results
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Teacher Recommendation by School
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x="school", hue="teacher_recommendation", palette="Set2")
plt.title("Teacher Recommendation Distribution by School")
plt.show()

# Export enriched dataset with clusters
export_path = "pupil_data_enriched_with_clusters.csv"
data.to_csv(export_path, index=False)
print(f"Enriched dataset with clustering saved to: {export_path}")

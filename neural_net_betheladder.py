#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:03:27 2024

@author: u1984485
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Load the cleaned dataset
data_path = "pupil_data_cleaned.csv"
data = pd.read_csv(data_path)

# Create a socioeconomic status index
# SES index = (Average house price / max house price) - (Crime rate / max crime rate) + (Income / max income)
data['socioeconomic_status_index'] = (
    (data['average_house_price'] / data['average_house_price'].max()) -
    (data['area_crime_rate'] / data['area_crime_rate'].max()) +
    (data['income'] / data['income'].max())
)

# Define the criteria for underperforming and underprivileged
underperforming_threshold = data['exam_results'].quantile(0.1)
underprivileged_criteria = (
    (data['income'] < data['income'].quantile(0.1)) |  # Low income
    (data['single_parent'] == 1) |  # Single parent household
    (data['parent_employment_status'] == 'Unemployed') |  # Unemployed parent(s)
    (data['access_to_resources'] == 0) |  # Limited access to resources
    (data['socioeconomic_status_index'] < data['socioeconomic_status_index'].quantile(0.1))  # Low SES index
)

# Create target variable for intervention
data['intervention_needed'] = (
    (data['exam_results'] < underperforming_threshold) & underprivileged_criteria
).astype(int)

# Factors considered for intervention:
# - Exam results (underperforming if in the bottom 25%)
# - Household income (underprivileged if below the 25th percentile)
# - Single parent household status
# - Parental employment status (unemployed)
# - Access to resources (limited)
# - Socioeconomic status index (below the 25th percentile)

# Prepare features and target variable
features = [
    'exam_results',
    'income',
    'single_parent',
    'parent_employment_status',
    'access_to_resources',
    'socioeconomic_status_index',
    'behaviour',
    'disciplinary_record'
]
X = pd.get_dummies(data[features], drop_first=True)
y = data['intervention_needed']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the neural network
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Make predictions
predictions = model.predict(X_scaled)
data['intervention_prediction'] = (predictions > 0.5).astype(int).flatten()

# Ensure at least 15 students are selected across all schools
if data['intervention_prediction'].sum() < 15:
    additional_needed = 15 - data['intervention_prediction'].sum()
    top_candidates = data[data['intervention_prediction'] == 0].nlargest(additional_needed, 'exam_results')
    data.loc[top_candidates.index, 'intervention_prediction'] = 1

# Ensure at least 5 additional students per school
additional_students = []
for school in data['school'].unique():
    school_data = data[data['school'] == school]
    current_count = school_data['intervention_prediction'].sum()
    if current_count < 5:
        additional_needed = 5 - current_count
        top_candidates = school_data[school_data['intervention_prediction'] == 0].nlargest(additional_needed, 'exam_results')
        data.loc[top_candidates.index, 'intervention_prediction'] = 1
        additional_students.append(top_candidates)

# Save students identified for intervention
intervention_students = data[data['intervention_prediction'] == 1]
intervention_students_path = "intervention_students.csv"
intervention_students.to_csv(intervention_students_path, index=False)

# Save additional students separately
if additional_students:
    additional_students_combined = pd.concat(additional_students)
    additional_students_path = "additional_students.csv"
    additional_students_combined.to_csv(additional_students_path, index=False)
    print(f"Additional student selections saved to: {additional_students_path}")

print(f"Intervention students saved to: {intervention_students_path}")

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load Data
data = pd.read_csv('houndsandhers.csv')

# Data Preprocessing
data['event_occurred'] = data['event_occurred'].astype(int)
data = data.dropna()

# Exploratory Data Analysis
print("Data Summary:")
print(data.describe())
print("Event Distribution:")
print(data['event_occurred'].value_counts())

# Kaplan-Meier Survival Curve
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))
kmf.fit(data['time'], event_observed=data['event_occurred'])
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()

# Survival Analysis - Cox Proportional Hazards
cox_data = data[['time', 'event_occurred'] + [col for col in data.columns if col not in ['time', 'event_occurred']]]
cox = CoxPHFitter()
cox.fit(cox_data, duration_col='time', event_col='event_occurred')
cox.print_summary()

# Feature Importance from Cox Model
cox.plot()
plt.title('Feature Importance - Cox Proportional Hazards')
plt.show()

# Predictive Modeling - Random Forest
y = data['time']
X = data.drop(['time', 'event_occurred'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

# Evaluate Random Forest Model
mae = mean_absolute_error(y_test, rf_predictions)
r2 = r2_score(y_test, rf_predictions)
print(f"Random Forest Model Performance:\nMAE: {mae:.2f}\nR2: {r2:.2f}")

# Monte Carlo Simulation
def monte_carlo_simulation(model, X, num_simulations=1000):
    simulations = []
    for _ in range(num_simulations):
        perturbed_X = X.copy() + np.random.normal(0, 0.01, X.shape)
        simulations.append(model.predict(perturbed_X))
    return np.array(simulations)

simulated_scenarios = monte_carlo_simulation(rf, X_test)
mean_simulated = simulated_scenarios.mean(axis=0)

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Times')
plt.plot(rf_predictions, label='Random Forest Predictions')
plt.fill_between(
    np.arange(len(mean_simulated)),
    mean_simulated - simulated_scenarios.std(axis=0),
    mean_simulated + simulated_scenarios.std(axis=0),
    color='gray', alpha=0.2, label='Monte Carlo Confidence Interval'
)
plt.legend()
plt.title('Survival Forecasting Results')
plt.xlabel('Sample Index')
plt.ylabel('Time')
plt.show()

# Generate Business Recommendations
cox_summary = cox.summary.sort_values('p', ascending=True)
important_features = cox_summary[cox_summary['p'] < 0.05].index.tolist()

print("Recommendations:")
print("1. Focus on the following features as they significantly impact survival time:")
print(important_features)
print("2. Use the Kaplan-Meier survival curve to estimate the survival probabilities for different time horizons.")
print("3. Utilize the Random Forest model for scenario analysis and risk prediction.")

# Save Results
cox_summary.to_csv('cox_feature_importance.csv', index=True)
pd.DataFrame({'Actual': y_test, 'Predicted': rf_predictions}).to_csv('random_forest_predictions.csv', index=False)
np.savetxt('monte_carlo_simulations.csv', simulated_scenarios, delimiter=',')

# generate_gdm_dataset.py
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 5000
data = {
    'Age': np.random.randint(18, 45, n_samples),
    'BMI': np.random.uniform(18, 40, n_samples),
    'Fasting_Glucose': np.random.randint(70, 200, n_samples),
    'Family_History': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'Previous_GDM': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    'Risk_Level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.5, 0.3, 0.2])
}

df = pd.DataFrame(data)
df.to_csv('gdm_dataset.csv', index=False)
print("Dataset saved as 'gdm_dataset.csv'")
print(df.head())
print("\nRisk Level Distribution:")
print(df['Risk_Level'].value_counts())

# Create DataFrame
df = pd.DataFrame(data)

# Calculate risk score and level
def calculate_risk(row):
    score = 0
    if row['Age'] >= 35:
        score += 2
    if row['BMI'] >= 30:
        score += 3
    if 100 <= row['Fasting_Glucose'] < 126:
        score += 2
    if row['Fasting_Glucose'] >= 126:
        score += 4
    if row['Family_History'] == 1:
        score += 2
    if row['Previous_GDM'] == 1:
        score += 3
    if score <= 3:
        return 'Low'
    elif score <= 6:
        return 'Medium'
    else:
        return 'High'

df['Risk_Level'] = df.apply(calculate_risk, axis=1)

# Save to CSV
df.to_csv('gdm_dataset.csv', index=False)
print("Dataset saved as 'gdm_dataset.csv'")
print(df.head())
print("\nRisk Level Distribution:")
print(df['Risk_Level'].value_counts())

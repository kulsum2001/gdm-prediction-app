import pandas as pd
import joblib

# Load your full dataset
data = pd.read_csv('../gdm_dataset.csv')  # Adjust path if needed

# Select only the features used for prediction
features = ['Age', 'BMI', 'Fasting_Glucose', 'Family_History', 'Previous_GDM']
background = data[features].sample(100, random_state=42).reset_index(drop=True)

# Load your scaler and apply only on numerical columns
scaler = joblib.load('static/models/scaler.joblib')
background[['Age', 'BMI', 'Fasting_Glucose']] = scaler.transform(background[['Age', 'BMI', 'Fasting_Glucose']])

# Save background data
background.to_csv('static/models/background_data.csv', index=False)
print("✅ Background data saved successfully.")

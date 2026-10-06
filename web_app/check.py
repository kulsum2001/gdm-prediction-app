import joblib
import pandas as pd

# Load model and other components
model = joblib.load('C:/Users/sakina kulsum/GDM_Prediction/web_app/static/models/xgb_model.joblib')
scaler = joblib.load('C:/Users/sakina kulsum/GDM_Prediction/web_app/static/models/scaler.joblib')
label_encoder = joblib.load('C:/Users/sakina kulsum/GDM_Prediction/web_app/static/models/label_encoder.joblib')

# Load sample data (same columns used in training)
df = pd.read_csv('C:/Users/sakina kulsum/GDM_Prediction/gdm_dataset.csv')
X = df[['Age', 'BMI', 'Fasting_Glucose', 'Family_History', 'Previous_GDM']]
X_scaled = X.copy()
X_scaled[['Age', 'BMI', 'Fasting_Glucose']] = scaler.transform(X_scaled[['Age', 'BMI', 'Fasting_Glucose']])

# Predict
probs = model.predict_proba(X_scaled)
preds = probs.argmax(axis=1)
labels = label_encoder.inverse_transform(preds)

# Print results
for i, (label, prob) in enumerate(zip(labels, probs)):
    confidence = prob.max() * 100
    print(f"Sample {i+1} → Risk: {label} | Confidence: {confidence:.2f}%")

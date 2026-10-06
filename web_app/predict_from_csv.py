import pandas as pd
import joblib

# Step 1: Load the model from the correct path
model_path = "static/models/xgb_model.joblib"
model = joblib.load(model_path)

# Step 2: Load your CSV file
csv_path = "static/shap_data/test_input.csv"  # <- Place your CSV here!
data = pd.read_csv(csv_path)

# Step 3: Ensure correct feature names
features = ['Age', 'BMI', 'Fasting_Glucose', 'Family_History', 'Previous_GDM']
X = data[features]

# Step 4: Predict
preds = model.predict(X)
probs = model.predict_proba(X)

# Step 5: Show predictions
# Add this dictionary to map class numbers to labels
label_map = {
    0: 'Low',
    1: 'Medium',
    2: 'High'
}

# Updated loop with label names
for i in range(len(preds)):
    risk_label = label_map[preds[i]]
    confidence = max(probs[i]) * 100
    print(f"Sample {i+1} → Risk: {risk_label} | Confidence: {confidence:.2f}%")

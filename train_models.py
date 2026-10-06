import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
import joblib
import shap
import os

warnings.filterwarnings("ignore")

# Create directories
os.makedirs('web_app/static/models', exist_ok=True)
os.makedirs('web_app/static/shap_data', exist_ok=True)
os.makedirs('mobile_app/models', exist_ok=True)
os.makedirs('mobile_app/shap_data', exist_ok=True)

# Load dataset
df = pd.read_csv('gdm_dataset.csv')
X = df[['Age', 'BMI', 'Fasting_Glucose', 'Family_History', 'Previous_GDM']]
y = df['Risk_Level']

# Scale numerical features
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['Age', 'BMI', 'Fasting_Glucose']] = scaler.fit_transform(X[['Age', 'BMI', 'Fasting_Glucose']])

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train models with explicit multi-class support
logistic_model = LogisticRegression(max_iter=2000, class_weight='balanced')
logistic_model.fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42, objective='multi:softprob', num_class=3, scale_pos_weight=1)
xgb_model.fit(X_train, y_train)

# Save models and scaler
joblib.dump(logistic_model, 'web_app/static/models/logistic_model.joblib')
joblib.dump(rf_model, 'web_app/static/models/rf_model.joblib')
joblib.dump(xgb_model, 'web_app/static/models/xgb_model.joblib')
joblib.dump(label_encoder, 'web_app/static/models/label_encoder.joblib')
joblib.dump(scaler, 'web_app/static/models/scaler.joblib')
joblib.dump(logistic_model, 'mobile_app/models/logistic_model.joblib')
joblib.dump(rf_model, 'mobile_app/models/rf_model.joblib')
joblib.dump(xgb_model, 'mobile_app/models/xgb_model.joblib')
joblib.dump(label_encoder, 'mobile_app/models/label_encoder.joblib')
joblib.dump(scaler, 'mobile_app/models/scaler.joblib')

# Generate and save SHAP values for multi-class
def predict_proba_wrapper(x):
    probs = xgb_model.predict_proba(x)
    print(f"predict_proba output shape: {probs.shape}")  # Debug: Should be (n_samples, 3)
    return probs

# Use a reasonable sample size for background data
n_samples = 100
X_train_sample = shap.sample(X_train, n_samples, random_state=42)

# Initialize explainer and compute SHAP values with multi-class support
print(f"X_train_sample shape: {X_train_sample.shape}")
explainer = shap.KernelExplainer(predict_proba_wrapper, X_train_sample, link="identity")
print(f"Explainer initialized with background shape: {X_train_sample.shape}")

# Compute SHAP values for all classes explicitly
print("Computing SHAP values...")
shap_values = explainer.shap_values(X_train_sample, nsamples=100)
print(f"Raw SHAP values type: {type(shap_values)}, shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}")
print(f"Full SHAP values: {shap_values}")  # Add full output for inspection

# Convert 3D array to list of 2D arrays for each class
if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[2] == len(label_encoder.classes_):
    shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    print(f"Converted SHAP values type: {type(shap_values_list)}, length: {len(shap_values_list)}")
    print(f"Converted SHAP values shapes: {[arr.shape for arr in shap_values_list]}")
else:
    raise ValueError(f"Unexpected SHAP values structure: {type(shap_values)}, shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}")

# Validate SHAP values structure
n_classes = len(label_encoder.classes_)
if len(shap_values_list) != n_classes:
    raise ValueError(f"SHAP values should be a list of {n_classes} arrays, got {len(shap_values_list)} arrays")
if not all(x.shape == (n_samples, X_train.shape[1]) for x in shap_values_list):
    raise ValueError(f"SHAP values should have shape ({n_samples}, {X_train.shape[1]}), got {[x.shape for x in shap_values_list]}")

# Save SHAP values
joblib.dump(shap_values_list, 'web_app/static/shap_data/shap_values.joblib')
joblib.dump(shap_values_list, 'mobile_app/shap_data/shap_values.joblib')

print("Models, scaler, and SHAP values saved successfully!")
print(f"SHAP values structure: {len(shap_values_list)} arrays, shapes: {[x.shape for x in shap_values_list]}")

# ------------------ Evaluation Section ------------------ #

def evaluate_model(model, model_name):
    print(f"\n--- {model_name} Evaluation ---")
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("Classification Report:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Cross-Validation
    scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
    print(f"Cross-Validation Accuracy (5-Fold): {scores.mean():.4f} (+/- {scores.std():.4f})")

# Run evaluation for each model
evaluate_model(logistic_model, "Logistic Regression")
evaluate_model(rf_model, "Random Forest")
evaluate_model(xgb_model, "XGBoost")

# ------------------ End of Evaluation ------------------ #
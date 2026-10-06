from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty
import pandas as pd
import numpy as np
import joblib
import shap
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

class GDMForm(BoxLayout):
    risk_level = StringProperty('')
    risk_details = StringProperty('')
    shap_explanation = StringProperty('')
    confidence_scores = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        load_dotenv()
        self.email_address = os.getenv('EMAIL_ADDRESS')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        try:
            self.df = pd.read_csv('../gdm_dataset.csv')
        except FileNotFoundError:
            self.df = None
            print("Warning: gdm_dataset.csv not found. Sample data loading disabled.")
        self.logistic_model = joblib.load('models/logistic_model.joblib')
        self.rf_model = joblib.load('models/rf_model.joblib')
        self.xgb_model = joblib.load('models/xgb_model.joblib')
        self.label_encoder = joblib.load('models/label_encoder.joblib')
        self.shap_explainer = joblib.load('shap_data/shap_explainer.joblib')

    def send_notification(self, inputs, risk_level, confidence_scores, doctor_email):
        try:
            msg = MIMEText(
                f"High Risk GDM Prediction Alert\n\n"
                f"Patient Inputs:\n"
                f"Age: {inputs['age']} years\n"
                f"BMI: {inputs['bmi']} kg/m²\n"
                f"Fasting Glucose: {inputs['glucose']} mg/dL\n"
                f"Family History: {'Yes' if inputs['family_history'] else 'No'}\n"
                f"Previous GDM: {'Yes' if inputs['previous_gdm'] else 'No'}\n"
                f"Model: {inputs['model']}\n"
                f"Risk Level: {risk_level}\n"
                f"Confidence Scores: Low: {confidence_scores['Low']}%, Medium: {confidence_scores['Medium']}%, High: {confidence_scores['High']}%"
            )
            msg['Subject'] = 'High Risk GDM Prediction Alert'
            msg['From'] = self.email_address
            msg['To'] = doctor_email

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.email_address, self.email_password)
                server.sendmail(self.email_address, doctor_email, msg.as_string())
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    def assess_risk(self):
        try:
            inputs = {
                'age': int(self.ids.age_input.text),
                'bmi': float(self.ids.bmi_input.text),
                'glucose': int(self.ids.glucose_input.text),
                'family_history': 1 if self.ids.family_history.active else 0,
                'previous_gdm': 1 if self.ids.previous_gdm.active else 0,
                'doctor_email': self.ids.doctor_email_input.text.strip(),
                'model': self.ids.model_spinner.text
            }

            if not inputs['doctor_email']:
                self.risk_level = 'Error'
                self.risk_details = 'Please enter a valid doctor email address.'
                self.shap_explanation = ''
                self.confidence_scores = ''
                return

            # Prepare input data
            input_data = np.array([[inputs['age'], inputs['bmi'], inputs['glucose'], 
                                   inputs['family_history'], inputs['previous_gdm']]])

            # Select model
            if inputs['model'] == 'Logistic Regression':
                model = self.logistic_model
            elif inputs['model'] == 'Random Forest':
                model = self.rf_model
            else:
                model = self.xgb_model

            # Predict and get confidence scores
            prediction = model.predict(input_data)[0]
            risk_level = self.label_encoder.inverse_transform([prediction])[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence_scores = {self.label_encoder.classes_[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities)}
            self.confidence_scores = f"Low: {confidence_scores['Low']}%\nMedium: {confidence_scores['Medium']}%\nHigh: {confidence_scores['High']}%"

            # Risk details
            if risk_level == 'Low':
                self.risk_level = 'Low'
                self.risk_details = 'Your risk of Gestational Diabetes Mellitus is low. Maintain a healthy lifestyle and consult your healthcare provider.'
            elif risk_level == 'Medium':
                self.risk_level = 'Medium'
                self.risk_details = 'Your risk of Gestational Diabetes Mellitus is moderate. Consider lifestyle changes and consult your healthcare provider.'
            else:
                self.risk_level = 'High'
                self.risk_details = 'Your risk of Gestational Diabetes Mellitus is high. Healthcare provider notified. Consult them for a detailed assessment.'
                self.send_notification(inputs, risk_level, confidence_scores, inputs['doctor_email'])

            # SHAP explanation
            shap_values = self.shap_explainer.shap_values(input_data)[prediction]
            feature_names = ['Age', 'BMI', 'Fasting Glucose', 'Family History', 'Previous GDM']
            explanation = "Feature Contributions (SHAP):\n"
            for i, (name, value) in enumerate(zip(feature_names, shap_values)):
                explanation += f"{name}: {value:.3f}\n"
            self.shap_explanation = explanation

        except ValueError:
            self.risk_level = 'Error'
            self.risk_details = 'Please enter valid numerical values for Age, BMI, and Glucose.'
            self.shap_explanation = ''
            self.confidence_scores = ''

    def save_data(self):
        try:
            inputs = {
                'age': int(self.ids.age_input.text),
                'bmi': float(self.ids.bmi_input.text),
                'glucose': int(self.ids.glucose_input.text),
                'family_history': 1 if self.ids.family_history.active else 0,
                'previous_gdm': 1 if self.ids.previous_gdm.active else 0,
                'doctor_email': self.ids.doctor_email_input.text.strip(),
                'model': self.ids.model_spinner.text
            }
            risk_level = self.risk_level
            confidence_scores = {k: float(v.strip('%')) for k, v in [line.split(': ') for line in self.confidence_scores.split('\n') if line]}

            if not inputs['doctor_email']:
                self.risk_level = 'Error'
                self.risk_details = 'Please enter a valid doctor email address.'
                self.shap_explanation = ''
                self.confidence_scores = ''
                return

            user_data = {
                'Age': inputs['age'],
                'BMI': inputs['bmi'],
                'Fasting_Glucose': inputs['glucose'],
                'Family_History': inputs['family_history'],
                'Previous_GDM': inputs['previous_gdm'],
                'Doctor_Email': inputs['doctor_email'],
                'Model': inputs['model'],
                'Risk_Level': risk_level,
                'Confidence_Low': confidence_scores.get('Low', 0),
                'Confidence_Medium': confidence_scores.get('Medium', 0),
                'Confidence_High': confidence_scores.get('High', 0)
            }
            user_df = pd.DataFrame([user_data])
            user_df.to_csv('../user_data.csv', mode='a', header=not os.path.exists('../user_data.csv'), index=False)
            self.risk_details = 'Data saved successfully! ' + self.risk_details
            if risk_level == 'High':
                self.send_notification(inputs, risk_level, confidence_scores, inputs['doctor_email'])
                self.risk_details = 'Healthcare provider notified. ' + self.risk_details
        except (ValueError, KeyError):
            self.risk_level = 'Error'
            self.risk_details = 'Please assess risk before saving data.'
            self.shap_explanation = ''
            self.confidence_scores = ''

    def load_sample(self):
        if self.df is not None:
            sample = self.df.sample(1).iloc[0]
            self.ids.age_input.text = str(int(sample['Age']))
            self.ids.bmi_input.text = str(sample['BMI'])
            self.ids.glucose_input.text = str(int(sample['Fasting_Glucose']))
            self.ids.family_history.active = bool(sample['Family_History'])
            self.ids.previous_gdm.active = bool(sample['Previous_GDM'])
            self.ids.doctor_email_input.text = ''
            self.risk_level = ''
            self.risk_details = ''
            self.shap_explanation = ''
            self.confidence_scores = ''
        else:
            self.risk_level = 'Error'
            self.risk_details = 'Sample data not available. Ensure gdm_dataset.csv exists.'
            self.shap_explanation = ''
            self.confidence_scores = ''

    def reset_form(self):
        self.ids.age_input.text = ''
        self.ids.bmi_input.text = ''
        self.ids.glucose_input.text = ''
        self.ids.family_history.active = False
        self.ids.previous_gdm.active = False
        self.ids.doctor_email_input.text = ''
        self.ids.model_spinner.text = 'Logistic Regression'
        self.risk_level = ''
        self.risk_details = ''
        self.shap_explanation = ''
        self.confidence_scores = ''

class GDMApp(App):
    def build(self):
        return GDMForm()

if __name__ == '__main__':
    GDMApp().run() 

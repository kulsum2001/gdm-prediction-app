from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user  # type: ignore
import pandas as pd
import joblib
import shap
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import bcrypt
import numpy as np
from fpdf import FPDF  # type: ignore

# IMPORTANT: Set non-GUI backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt


app = Flask(__name__)

# ---- load .env before reading SECRET_KEY ----
load_dotenv()
app.secret_key = os.getenv('SECRET_KEY')
# ---------------------------------------------

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    def __init__(self, username, email):
        self.id = username
        self.email = email


def load_users():
    try:
        return pd.read_csv('../users.csv')
    except FileNotFoundError:
        return pd.DataFrame(columns=['username', 'email', 'password'])


def save_user(username, email, password):
    users = load_users()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).hex()
    new_user = pd.DataFrame([[username, email, hashed_password]],
                            columns=['username', 'email', 'password'])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv('../users.csv', index=False)


# -------------------- LOAD MODELS & SCALER ONCE --------------------
logistic_model = joblib.load('static/models/logistic_model.joblib')
rf_model = joblib.load('static/models/rf_model.joblib')
xgb_model = joblib.load('static/models/xgb_model.joblib')
label_encoder = joblib.load('static/models/label_encoder.joblib')
scaler = joblib.load('static/models/scaler.joblib')

# ✅ Cache SHAP explainer globally (only once at startup)
shap_explainer = None
_background_columns = None
try:
    background = pd.read_csv('static/models/background_data.csv')

    cols_to_scale = [c for c in ['Age', 'BMI', 'Fasting_Glucose'] if c in background.columns]
    background_scaled = background.copy()
    if cols_to_scale:
        background_scaled[cols_to_scale] = scaler.transform(background_scaled[cols_to_scale])

    _background_columns = list(background_scaled.columns)
    shap_explainer = shap.TreeExplainer(xgb_model, data=background_scaled,
                                        feature_names=_background_columns)
except Exception as e:
    try:
        shap_explainer = shap.TreeExplainer(xgb_model)
    except Exception as e2:
        shap_explainer = None
    with open("error_log.txt", "a") as log:
        log.write(f"SHAP init error; falling back: {e}\n")
# ------------------------------------------------------------------


@login_manager.user_loader
def load_user(username):
    users = load_users()
    row = users[users['username'] == username]
    if not row.empty:
        return User(row.iloc[0]['username'], row.iloc[0]['email'])
    return None


@app.route('/')
def index():
    return render_template('index.html') if current_user.is_authenticated else redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        row = users[users['username'] == username]
        if not row.empty:
            if bcrypt.checkpw(password.encode(), bytes.fromhex(row.iloc[0]['password'])):
                login_user(User(row.iloc[0]['username'], row.iloc[0]['email']))
                return redirect(url_for('predict'))
        flash('Invalid username or password', 'error')
    elif current_user.is_authenticated:
        return redirect(url_for('predict'))

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if username in load_users()['username'].values:
            flash('Username already exists', 'error')
        else:
            save_user(username, email, password)
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            if 'assess' in request.form:
                # Input validation
                try:
                    age = float(request.form['age'])
                    bmi = float(request.form['bmi'])
                    glucose = float(request.form['glucose'])
                except ValueError:
                    flash('Invalid numeric input. Please enter valid numbers.', 'error')
                    return redirect(url_for('predict'))

                if age < 18 or age > 50:
                    flash('Age should be between 18 and 50 for accurate results.', 'error')
                    return redirect(url_for('predict'))

                family_history = int(request.form['family_history'])
                previous_gdm = int(request.form['previous_gdm'])
                doctor_email = request.form['doctor_email']

                # Always use XGBoost
                selected_model = xgb_model

                original_input = pd.DataFrame([[age, bmi, glucose, family_history, previous_gdm]],
                                              columns=['Age', 'BMI', 'Fasting_Glucose',
                                                       'Family_History', 'Previous_GDM'])

                input_df = original_input.copy()
                input_df[['Age', 'BMI', 'Fasting_Glucose']] = scaler.transform(
                    input_df[['Age', 'BMI', 'Fasting_Glucose']]
                )

                pred = selected_model.predict(input_df)[0]
                risk_level = label_encoder.inverse_transform([pred])[0]
                prob = selected_model.predict_proba(input_df)[0]
                confidence = {label: round(float(p) * 100, 2)
                              for label, p in zip(label_encoder.classes_, prob)}

                # ---- SHAP Section ----
                plot_path = None
                shap_explanation = None
                shap_details = []
                try:
                    background = pd.read_csv('static/models/background_data.csv')
                    explainer = shap.Explainer(selected_model, background)
                    shap_values = explainer(input_df)

                    if len(shap_values.values.shape) == 3:
                        shap_val = shap_values.values[0][:, pred]
                        base_value = explainer.expected_value[pred]
                    else:
                        shap_val = shap_values.values[0]
                        base_value = explainer.expected_value[pred] if isinstance(
                            explainer.expected_value, list) else explainer.expected_value

                    timestamp = int(datetime.now().timestamp())
                    filename = f"shap_plot_{current_user.id}_{timestamp}.png"
                    plot_path = f"images/{filename}"
                    full_path = os.path.join("static", plot_path)

                    fig = plt.figure(figsize=(10, 1.2))
                    shap.plots.force(
                        base_value=base_value,
                        shap_values=shap_val,
                        features=input_df.iloc[0],
                        matplotlib=True,
                        show=False
                    )
                    plt.savefig(full_path, bbox_inches="tight", dpi=150)
                    plt.close(fig)

                    feature_idx = int(np.argmax(np.abs(shap_val)))
                    feature_name = original_input.columns[feature_idx]
                    direction = "increased" if shap_val[feature_idx] > 0 else "decreased"
                    shap_explanation = f"🔍 Most influential factor: {feature_name} (it {direction} the predicted risk)."

                    top_indices = np.argsort(np.abs(shap_val))[::-1][:3]
                    for idx in top_indices:
                        fname = original_input.columns[idx]
                        val = shap_val[idx]
                        impact = "increased" if val > 0 else "decreased"
                        shap_details.append(f"{fname} {impact} the risk by {abs(val):.6f}")

                except Exception as e:
                    with open("error_log.txt", "a") as log:
                        log.write(f"SHAP Error: {e}\n")

                # Save user data (now correctly INSIDE the if 'assess' block)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                result_data = pd.DataFrame([[current_user.id, age, bmi, glucose,
                                             family_history, previous_gdm, risk_level,
                                             doctor_email, confidence['Low'],
                                             confidence['Medium'], confidence['High'],
                                             timestamp]],
                                           columns=['username', 'Age', 'BMI', 'Fasting_Glucose',
                                                    'Family_History', 'Previous_GDM', 'Risk_Level',
                                                    'Doctor_Email', 'Conf_Low', 'Conf_Medium',
                                                    'Conf_High', 'Timestamp'])
                result_data.to_csv('../user_data.csv', mode='a',
                                   header=not os.path.exists('../user_data.csv'),
                                   index=False)

                if risk_level == 'High':
                    try:
                        msg = MIMEText(
                            f'GDM Risk Assessment Result\n\nUser: {current_user.id}\n'
                            f'Risk Level: {risk_level}\n'
                            f'Details: Age={age}, BMI={bmi}, Glucose={glucose}, '
                            f'Family History={family_history}, Previous GDM={previous_gdm}'
                        )
                        msg['Subject'] = 'High GDM Risk Alert'
                        msg['From'] = os.getenv('EMAIL_USER')
                        msg['To'] = doctor_email

                        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                            server.login(msg['From'], os.getenv('EMAIL_PASS'))
                            server.send_message(msg)
                    except Exception as e:
                        with open('error_log.txt', 'a') as log:
                            log.write(f"Email Error: {e}\n")

                # Store in session and go to result
                session['risk_level'] = risk_level
                session['plot_path'] = plot_path
                session['confidence_scores'] = confidence
                session['shap_explanation'] = shap_explanation
                session['shap_details'] = shap_details

                return redirect(url_for('result'))

        except Exception as e:
            flash('Something went wrong. Please try again.', 'error')
            with open("error_log.txt", "a") as log:
                log.write(f"Predict Error: {e}\n")

    return render_template('index.html')


@app.route('/result')
@login_required
def result():
    if not session.get('risk_level'):
        flash('No recent prediction found.', 'info')
        return redirect(url_for('predict'))

    return render_template(
        'result.html',
        risk_level=session.get('risk_level'),
        confidence_scores=session.get('confidence_scores'),
        plot_path=session.get('plot_path'),   # ✅ add this line
        shap_explanation=session.get('shap_explanation'),
        shap_details=session.get('shap_details')
    )


@app.route('/save_data', methods=['POST'])
@login_required
def save_data():
    try:
        age = request.form.get('age')
        bmi = request.form.get('bmi')
        glucose = request.form.get('glucose')
        family_history = request.form.get('family_history')
        previous_gdm = request.form.get('previous_gdm')
        doctor_email = request.form.get('doctor_email')
        risk_level = session.get('risk_level')
        confidence = session.get('confidence_scores', {})

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        result_data = pd.DataFrame([[current_user.id, age, bmi, glucose,
                                     family_history, previous_gdm, risk_level,
                                     doctor_email,
                                     confidence.get('Low', 0),
                                     confidence.get('Medium', 0),
                                     confidence.get('High', 0),
                                     timestamp]],
                                   columns=['username', 'Age', 'BMI', 'Fasting_Glucose',
                                            'Family_History', 'Previous_GDM', 'Risk_Level',
                                            'Doctor_Email', 'Conf_Low', 'Conf_Medium',
                                            'Conf_High', 'Timestamp'])

        result_data.to_csv('../user_data.csv', mode='a',
                           header=not os.path.exists('../user_data.csv'),
                           index=False)

        flash("✅ Data saved successfully!", "success")
        return redirect(url_for('dashboard'))

    except Exception as e:
        with open("error_log.txt", "a") as log:
            log.write(f"Save Data Error: {e}\n")
        flash("❌ Something went wrong while saving data.", "error")
        return redirect(url_for('result'))


@app.route('/dashboard')
@login_required
def dashboard():
    try:
        df = pd.read_csv('../user_data.csv')
        user_entries = df[df['username'] == current_user.id]

        if user_entries.empty:
            flash("No entries found for your account.", "info")
            return render_template('dashboard.html', user_data=[], latest=None, confidence_scores={})

        latest = user_entries.iloc[-1]

        try:
            confidence = {
                'Low': float(latest.get('Conf_Low', 0)),
                'Medium': float(latest.get('Conf_Medium', 0)),
                'High': float(latest.get('Conf_High', 0))
            }
        except Exception as e:
            confidence = {'Low': 0.0, 'Medium': 0.0, 'High': 0.0}
            with open("error_log.txt", "a") as f:
                f.write(f"Confidence parsing error: {e}\n")

        return render_template('dashboard.html',
                               user_data=user_entries.to_dict(orient='records'),
                               latest=latest.to_dict(),
                               confidence_scores=confidence)

    except Exception as e:
        return f"❌ Error loading dashboard: {str(e)}"


@app.route('/download_report')
@login_required
def download_report():
    try:
        df = pd.read_csv('../user_data.csv')
        user_data = df[df['username'] == current_user.id]

        if user_data.empty:
            return "No data available."

        latest = user_data.iloc[-1]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="GDM Risk Prediction Report", ln=True, align="C")
        pdf.ln(10)
        for field in ['Age', 'BMI', 'Fasting_Glucose',
                      'Family_History', 'Previous_GDM', 'Risk_Level', 'Timestamp']:
            pdf.cell(200, 10, txt=f"{field}: {latest[field]}", ln=True)

        file_path = f"static/reports/report_{current_user.id}.pdf"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pdf.output(file_path)

        return redirect(f"/{file_path}")
    except Exception as e:
        return f"Error generating report: {str(e)}"


@app.route('/home')
def home():
    last_prediction = session.get('last_prediction')
    return render_template('home.html', last_prediction=last_prediction)


@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.cache_control.no_cache = True
    response.cache_control.must_revalidate = True
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == '__main__':
    app.run(debug=True)

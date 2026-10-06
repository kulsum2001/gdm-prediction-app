import pandas as pd

# Load the CSV
df = pd.read_csv('C:/Users/sakina kulsum/GDM_Prediction/user_data.csv')

# Drop rows where Timestamp is missing or NaN
df_cleaned = df.dropna(subset=['Timestamp'])

# Save back
df_cleaned.to_csv('../user_data.csv', index=False)

print("✅ Cleaned successfully! Only rows with valid timestamps retained.")

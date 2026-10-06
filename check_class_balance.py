import pandas as pd

# Path to your original training dataset
df = pd.read_csv("C:/Users/sakina kulsum/GDM_Prediction/gdm_dataset.csv")  # <-- update path if needed

# Show how many samples are in each class
print("Class Distribution:")
print(df['Risk_Level'].value_counts())

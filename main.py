import pandas as pd

df = pd.read_csv("./data/diabetes_risk_prediction_dataset.csv")
df = df.drop(columns=['Patient_ID'])

# Numerical Columns
num_cols = ["Age", "Height_cm", "Weight_kg", "Blood_Glucose", "HbA1c", "Fasting_Blood_Sugar", "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic", "Total_Cholesterol", "Sleep_Hours"]

df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()

# Binary Columns
bin_cols = ["Family_History_Diabetes", "Hypertension", "Heart_Disease", "Fatty_Liver", "PCOS"]

df[bin_cols] = df[bin_cols].replace({"Yes": 1, "No": 0})

# Categorical Columns
cat_cols = ["Physical_Activity_Level", "Diet_Quality", "Stress_Level", "Smoking_Status", "Alcohol_Consumption"]

# Final Dataframe
cols = num_cols + bin_cols + cat_cols + ["Diabetes_Risk"]

df = df[cols]
df = pd.get_dummies(df, columns=cat_cols)


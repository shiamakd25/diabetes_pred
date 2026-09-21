import pandas as pd

# DATAFRAME PREPARATION:

df = pd.read_csv("./data/diabetes_risk_prediction_dataset.csv")
df = df.drop(columns=['Patient_ID'])

# Numerical Columns
num_cols = ["Age", "Height_cm", "Weight_kg", "Blood_Glucose", "HbA1c", "Fasting_Blood_Sugar", "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic", "Total_Cholesterol", "Sleep_Hours"]

# Binary Columns
bin_cols = ["Family_History_Diabetes", "Hypertension", "Heart_Disease", "Fatty_Liver", "PCOS"]

df[bin_cols] = df[bin_cols].replace({"Yes": 1, "No": 0})

# Categorical Columns
cat_cols = ["Physical_Activity_Level", "Diet_Quality", "Stress_Level", "Smoking_Status", "Alcohol_Consumption"]

# Final Dataframe
cols = num_cols + bin_cols + cat_cols + ["Diabetes_Risk"]

df = df[cols]
df = pd.get_dummies(df, columns=cat_cols, dtype=int)

# DEFINE DATA:

df_train = df.sample(frac=0.8, random_state=17)
df_val = df.drop(index=df_train.index)

y_train = df_train["Diabetes_Risk"]
y_val = df_val["Diabetes_Risk"]

X_train = df_train.drop(columns=["Diabetes_Risk"])
X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean) / X_train_std

X_val = df_val.drop(columns=["Diabetes_Risk"])
X_val = (X_val - X_train_mean) / X_train_std

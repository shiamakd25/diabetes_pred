import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import math
import sys

# DATAFRAME PREPARATION:

df = pd.read_csv("./data/diabetes_risk_prediction_dataset.csv")
df = df.drop(columns=['Patient_ID'])

# Numerical Columns
num_cols = ["Age", "Height_cm", "Weight_kg", "Blood_Glucose", "HbA1c", "Fasting_Blood_Sugar", "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic", "Total_Cholesterol", "Sleep_Hours"]

# Binary Columns
bin_cols = ["Family_History_Diabetes", "Hypertension", "Heart_Disease", "Fatty_Liver", "PCOS"]

df[bin_cols] = df[bin_cols].replace({"Yes": 1, "No": 0})
for col in bin_cols:
    df[col] = pd.to_numeric(df[col])

# Categorical Columns
cat_cols = ["Physical_Activity_Level", "Diet_Quality", "Stress_Level", "Smoking_Status", "Alcohol_Consumption"]

# Output Column
df["Diabetes_Risk"] = pd.to_numeric(df["Diabetes_Risk"].replace({"Low": 0, "Moderate": 1, "High":2}))

# Final Dataframe
cols = num_cols + bin_cols + cat_cols + ["Diabetes_Risk"]

df = df[cols]
df = pd.get_dummies(df, columns=cat_cols, dtype=int)

# DEFINE DATA:

df_train = df.sample(frac=0.8, random_state=17)
df_val = df.drop(index=df_train.index)

medians = df_train.median()
df_train = df_train.fillna(medians)
df_val = df_val.fillna(medians)

y_train = df_train["Diabetes_Risk"]
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_val = df_val["Diabetes_Risk"]
y_val = torch.tensor(y_val.values, dtype=torch.long)

X_train = df_train.drop(columns=["Diabetes_Risk"])

X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean) / X_train_std

X_train = torch.tensor(X_train.values, dtype=torch.float32)

X_val = df_val.drop(columns=["Diabetes_Risk"])
X_val = (X_val - X_train_mean) / X_train_std
X_val = torch.tensor(X_val.values, dtype=torch.float32)

# CREATE MODEL:

num_entries = X_train.size(0)
batch_size = 64

model = nn.Sequential(
    nn.Linear(30, 30),
    nn.ReLU(),
    nn.Linear(30, 10),
    nn.ReLU(),
    nn.Linear(10, 3)
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(0, 400):
    loss_sum = 0
    
    for start in range(0, num_entries, batch_size):
        end = min(start + batch_size, num_entries)
        X_data = X_train[start:end]
        y_data = y_train[start:end]
        
        optimizer.zero_grad()
        outputs = model(X_data)
        loss = loss_fn(outputs, y_data)
        loss.backward()
        loss_sum += loss.item()
        optimizer.step()
        
    if i % 50 == 0:
        print(
            f"Iteration {i}:\n Sum - {loss_sum}\n Average - {loss_sum/(math.floor(num_entries/batch_size))}\n------------------------------------------------"
        )
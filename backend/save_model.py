"""
Script to train and save the model from the notebook.
Run this after training the model in the notebook, or modify to train directly.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import os
import pickle

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load the CSV exported from Microsoft Forms
# Update the filename if needed - check the actual filename in the parent directory
# Try common filenames
csv_files = [
    '../Workplace Dynamics & Career Sentiment Survey 2026(1-35).csv',
    '../survey_data.csv',
    'survey_data.csv'
]

df_raw = None
for csv_file in csv_files:
    try:
        df_raw = pd.read_csv(csv_file)
        print(f"Loaded CSV from: {csv_file}")
        break
    except FileNotFoundError:
        continue

if df_raw is None:
    raise FileNotFoundError("Could not find CSV file. Please update the path in save_model.py")

# Mapping long survey questions to technical feature names
column_mapping = {
    'Primary Department Question': 'Department',
    'Average Monthly Overtime': 'Overtime',
    'How many years has it been since your last job title change or promotion?': 'Promotion_Gap',
    'Satisfaction': 'Job_Satisfaction',
    'Risk': 'AI_Automation_Risk',
    'Has your specific department experienced staff layoffs or "firing" in the last 12 months?': 'Recent_Layoffs',
    'Security': 'Job_Security',
    'If you left today, how easy would it be to find a similar role elsewhere?': 'Market_Demand',
    'Are you actively planning to leave your current company or looking for a new job within the next 6 months?': 'Attrition'
}

df = df_raw.rename(columns=column_mapping)
df = df[list(column_mapping.values())]

# -----------------------------
# Data cleaning / normalization
# -----------------------------
# Normalize en-dash to hyphen in Overtime column (fixes character mismatch issue)
df['Overtime'] = df['Overtime'].astype(str).str.replace('–', '-', regex=False)

# Ensure numeric column is numeric (avoid poisoning numerics with 'Neutral')
df['Promotion_Gap'] = pd.to_numeric(df['Promotion_Gap'], errors='coerce')

# Convert Target to Binary (1 for Yes, 0 for No)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0}).astype('Int64')

# Fill missing categorical values only
cat_cols = [
    'Department',
    'Overtime',
    'Job_Satisfaction',
    'AI_Automation_Risk',
    'Recent_Layoffs',
    'Job_Security',
    'Market_Demand',
]
df[cat_cols] = df[cat_cols].fillna('Neutral')

# Fill missing numeric values
df['Promotion_Gap'] = df['Promotion_Gap'].fillna(df['Promotion_Gap'].median())

# Finalize target dtype
df['Attrition'] = df['Attrition'].fillna(0).astype(int)

# Define categorical columns for the CatBoost algorithm
cat_features = ['Department', 'Overtime', 'Job_Satisfaction', 'AI_Automation_Risk', 
                'Recent_Layoffs', 'Job_Security', 'Market_Demand']

# Train / Validation / Test Split
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Split the 80% again to get a 10% Validation set for Early Stopping
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.2,  # larger validation set to stabilize early stopping on small data
    random_state=42,
    stratify=y_temp,
)

print(f"Dataset Split Complete: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
print("Class distribution (train):")
print(y_train.value_counts(dropna=False).to_string())
print("Class distribution (val):")
print(y_val.value_counts(dropna=False).to_string())
print("Class distribution (test):")
print(y_test.value_counts(dropna=False).to_string())

# Train the model
model = CatBoostClassifier(
    loss_function='Logloss',
    # Use Logloss for early stopping so training continues to improve probability quality
    # (AUC can hit 1.0 very early on small datasets, causing tiny models and ~0.5 outputs).
    eval_metric='Logloss',
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    early_stopping_rounds=200,
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=100
)

# Train the model
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val))

# Save the model as pickle file
model_data = {
    'model': model,
    'cat_features': cat_features,
    'feature_order': list(X.columns)
}
pickle_path = 'model/attrition_model.pkl'
with open(pickle_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"\nModel saved successfully as pickle file: {pickle_path}")

# Also save in CatBoost native format (optional, for compatibility)
cbm_path = 'model/attrition_model.cbm'
model.save_model(cbm_path)
print(f"Model also saved in CatBoost format: {cbm_path}")

# Print some metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\nTest Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")


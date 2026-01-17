import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import os
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


os.makedirs('model', exist_ok=True)

# Load the CSV exported from Microsoft Forms
csv_files = [
    'survey_data.csv'
]

df_raw = None

for csv_file in csv_files:
    try:
        df_raw = pd.read_csv(csv_file)
        break
    except FileNotFoundError:
        continue

if df_raw is None:
    raise FileNotFoundError("Could not find CSV file")

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


# Data cleaning 
df['Overtime'] = df['Overtime'].astype(str).str.replace('–', '-', regex=False)
df['Promotion_Gap'] = pd.to_numeric(df['Promotion_Gap'], errors='coerce')
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0}).astype('Int64')

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
df['Promotion_Gap'] = df['Promotion_Gap'].fillna(df['Promotion_Gap'].median())
df['Attrition'] = df['Attrition'].fillna(0).astype(int)

# Define categorical columns for the CatBoost algorithm
cat_features = ['Department', 'Overtime', 'Job_Satisfaction', 'AI_Automation_Risk', 
                'Recent_Layoffs', 'Job_Security', 'Market_Demand']

# Train, Validation and Test Split
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# larger validation set to stabilize early stopping on small data
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.2,
    random_state=42,
    stratify=y_temp,
)

print("Class distribution (train):")
print(y_train.value_counts(dropna=False).to_string())
print("Class distribution (val):")
print(y_val.value_counts(dropna=False).to_string())
print("Class distribution (test):")
print(y_test.value_counts(dropna=False).to_string())

# Train the model
model = CatBoostClassifier(
    loss_function='Logloss',
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

# Save the model
model_data = {
    'model': model,
    'cat_features': cat_features,
    'feature_order': list(X.columns)
}
pickle_path = 'model/attrition_model.pkl'
with open(pickle_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"\nModel saved successfully as pickle file: {pickle_path}")



y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\nTest Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")


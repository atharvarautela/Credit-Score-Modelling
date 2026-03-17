# Credit-Score-Modelling
# ============================
# Credit Scoring Model Example
# ============================

# 📌 Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 📌 Step 2: Simulate dataset (if no dataset is available)
np.random.seed(42)
n_samples = 1000

data = {
    "income": np.random.randint(20000, 150000, n_samples),
    "debts": np.random.randint(0, 50000, n_samples),
    "payment_history": np.random.randint(0, 100, n_samples),   # score out of 100
    "age": np.random.randint(18, 70, n_samples),
    "loan_amount": np.random.randint(1000, 50000, n_samples),
    "employment_status": np.random.choice(["employed", "unemployed", "self-employed"], n_samples),
    "credit_utilization_ratio": np.random.uniform(0.1, 1.0, n_samples),
    "number_of_loans": np.random.randint(0, 10, n_samples),
    "credit_inquiries": np.random.randint(0, 5, n_samples),
}

# Target variable: creditworthy (1 = good, 0 = bad)
# Rule of thumb for simulation: high income + good payment history → more creditworthy
data["creditworthy"] = ((data["income"] > 50000) & (data["payment_history"] > 50)).astype(int)

df = pd.DataFrame(data)

# 📌 Step 3: Preprocessing
# Encode categorical variable
encoder = LabelEncoder()
df["employment_status"] = encoder.fit_transform(df["employment_status"])

# Features & target
X = df.drop("creditworthy", axis=1)
y = df["creditworthy"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 📌 Step 4: Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# 📌 Step 5: Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred), 3))
    print("Recall:", round(recall_score(y_test, y_pred), 3))
    print("F1-Score:", round(f1_score(y_test, y_pred), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

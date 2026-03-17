# Machine Learning-based Credit Scoring (Loan Approval Prediction)

## Overview
This project demonstrates a machine learning approach to predict customer creditworthiness for loan approval decisions.  
It simulates a dataset with financial and behavioral features and trains multiple classification models to identify creditworthy customers.

---

## Features
- Synthetic dataset creation with features like income, debts, payment history, loan amount, employment status, and credit utilization  
- Preprocessing:
  - Encoding categorical variables
  - Feature scaling using StandardScaler  
- Model training and comparison:
  - Logistic Regression
  - Decision Tree
  - Random Forest  
- Model evaluation using multiple metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC  

---

## Dataset
- The dataset is **synthetically generated** using NumPy to simulate realistic customer credit profiles.  
- Target variable: `creditworthy` (1 = good, 0 = bad) based on income and payment history.

---

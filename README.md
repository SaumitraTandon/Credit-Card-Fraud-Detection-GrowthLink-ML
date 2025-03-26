# Credit Card Fraud Detection

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Fraudulent transactions can lead to significant financial losses, and detecting them is crucial for preventing fraud. The goal of this project is to develop a robust classification model that accurately differentiates between fraudulent and legitimate transactions.

## Dataset
The dataset used in this project consists of:
- **Training Dataset:** Contains labeled transactions used to train the machine learning model.
- **Test Dataset:** Used to evaluate the performance of the trained model.
- You can access the dataset at : https://www.kaggle.com/datasets/kartik2112/fraud-detection

Each transaction in the dataset includes features that represent transaction details, such as the amount, time, and anonymized numerical features derived from PCA (Principal Component Analysis). The dataset is highly imbalanced, with fraudulent transactions being significantly less frequent than legitimate ones.

## Steps Followed in the Notebook
### 1. Importing Libraries
The necessary libraries are imported:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split
```

### 2. Loading the Dataset
The dataset is read and the first few rows are displayed:
```python
training_set = pd.read_csv('fraudTrain.csv')
test_set = pd.read_csv('fraudTest.csv')
training_set.head(10)
test_set.head(10)
```

### 3. Data Preprocessing
- **Dropping Unnecessary Columns**
```python
training_set.drop(columns=['trans_date_trans_time', 'first', 'last', 'street', 'city', 'state', 'dob', 'trans_num'], inplace=True)
training_set.drop(training_set.columns[0], axis=1, inplace=True)
```
- **Encoding Categorical Variables**
```python
encoder = LabelEncoder()
def encode(data):
    data['merchant'] = encoder.fit_transform(data['merchant'])
    data['category'] = encoder.fit_transform(data['category'])
    data['gender'] = encoder.fit_transform(data['gender'])
    return data

training_set = encode(training_set)
test_set = encode(test_set)
```
- **Feature Scaling**
```python
scaler = StandardScaler()
training_set.iloc[:, :] = scaler.fit_transform(training_set)
test_set.iloc[:, :] = scaler.transform(test_set)
```

### 4. Splitting the Data
```python
X_train, X_test, y_train, y_test = train_test_split(training_set.drop('is_fraud', axis=1), training_set['is_fraud'], test_size=0.2, random_state=42)
```

### 5. Model Training
A Random Forest model is trained:
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 6. Model Evaluation
- **Predictions & Accuracy Score**
```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'ROC-AUC Score: {roc_auc}')
```
- **Feature Importance Visualization**
```python
importances = model.feature_importances_
plt.figure(figsize=(10,5))
sns.barplot(x=importances, y=X_train.columns)
plt.title("Feature Importance")
plt.show()
```

## Installation & Requirements
Ensure you have the necessary dependencies installed before running the notebook. You can install them using the following command:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost lightgbm
```

## Usage
1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/your-repository/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```
2. Open the Jupyter Notebook and run the cells step by step:
```bash
jupyter notebook Credit_Card_Fraud_Detection_GrothLink_ML.ipynb
```
3. Follow the instructions in the notebook to preprocess the data, train models, and evaluate performance.

## Results & Analysis
After training and evaluating the models, the best-performing model is selected based on the evaluation metrics. The results are visualized using:
- **Confusion Matrix:** To understand the classification performance.
- **ROC Curve:** To visualize the trade-off between true positive and false positive rates.

## Contribution
Contributions are welcome! If you would like to improve the model or add new features, follow these steps:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature-branch`
5. Open a pull request.


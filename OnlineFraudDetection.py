import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
data = pd.read_csv("OnlinePaymentFraudDetection.csv")

# Print column names and check for null values
print("Column Names:", data.columns)
print("Null values in each column:\n", data.isnull().sum())

# Drop non-numeric columns that cannot be used for correlation
non_numeric_cols = ['nameOrig', 'nameDest']
data = data.drop(columns=non_numeric_cols)

# Display distribution of transaction types
transaction_types = data["type"].value_counts()
transactions = transaction_types.index
quantity = transaction_types.values

# Plot distribution of transaction types
figure = px.pie(data, values=quantity, names=transactions, hole=0.5, title="Distribution of Transaction Type")
figure.show()

# Map transaction types to numeric values
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})

# Ensure 'isFraud' is numeric for correlation calculation
data["isFraud"] = data["isFraud"].map({0: 0, 1: 1})

# Calculate correlation matrix
correlation = data.corr()
print("Correlation Matrix:\n", correlation["isFraud"].sort_values(ascending=False))

# Select numeric columns for correlation analysis
numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
print("Numeric Data Correlation Matrix:\n", correlation)

# Split data into features and target variable
X = data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]]
y = data["isFraud"]

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Additional evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Make a prediction using the trained model with feature names
features = pd.DataFrame([[4, 9000.60, 9000.60, 0.0]], columns=["type", "amount", "oldbalanceOrg", "newbalanceOrig"])
prediction = model.predict(features)
prediction_label = "Fraud" if prediction[0] == 1 else "No Fraud"
print(f"Prediction for provided features: {prediction_label}")
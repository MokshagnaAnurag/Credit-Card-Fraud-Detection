# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# Load the dataset
# Update the file path with the correct location of your dataset
df = pd.read_csv("/content/fraudTest.csv")

# Convert date-related columns into a datetime format for easier processing
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"])

# Extract meaningful time-based features from the transaction timestamp
df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
df["transaction_day"] = df["trans_date_trans_time"].dt.day
df["transaction_month"] = df["trans_date_trans_time"].dt.month

# Calculate the age of the cardholder at the time of the transaction
current_date = pd.to_datetime("2025-03-21")  # Replace with the actual current date
df["age"] = (current_date - df["dob"]).dt.days // 365

# Remove unnecessary columns that are not useful for fraud detection
columns_to_drop = ["trans_date_trans_time", "dob", "cc_num", "trans_num", "street"]
df.drop(columns=columns_to_drop, inplace=True)

# Encode categorical variables (convert text-based data into numerical values)
label_encoders = {}
for column in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store encoders for potential future decoding

# Handle missing values by filling them with the median of each column
df.fillna(df.median(numeric_only=True), inplace=True)

# Define features (X) and the target variable (y)
X = df.drop(columns=["is_fraud"])  # Features
y = df["is_fraud"]  # Target variable (fraud or not)

# Normalize numerical features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train a Random Forest classifier
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Fraud", "Fraud"],
    yticklabels=["No Fraud", "Fraud"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot the ROC Curve to visualize model performance
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Save the processed dataset to a new CSV file
df.to_csv("processed_dataset.csv", index=False)
print("Preprocessing and model training complete. Processed dataset saved as 'processed_dataset.csv'.")

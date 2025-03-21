# Fraud Detection Using Machine Learning  

## 📌 Project Description  
This project implements a **Random Forest Classifier** to detect fraudulent credit card transactions. It involves **data preprocessing, feature engineering, model training, and evaluation** using real-world transaction data.

## 📂 Folder Structure  
```
fraud-detection/
│── README.md
│── src/
│   ├── fraud_detection.py  # Main script
│── requirements.txt
```

## 🛠️ Installation & Setup  

### **1️⃣ Clone the repository**  
```sh
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### **2️⃣ Create a virtual environment (optional but recommended)**  
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3️⃣ Install dependencies**  
```sh
pip install -r requirements.txt
```

### **4️⃣ Run the script**  
```sh
python src/fraud_detection.py
```

## 📊 Model Performance  

### **Accuracy:** 0.9986  

### **Classification Report:**  
```
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    110715
           1       0.98      0.66      0.79       429

    accuracy                           1.00    111144
   macro avg       0.99      0.83      0.89    111144
weighted avg       1.00      1.00      1.00    111144
```

- **Confusion Matrix & ROC Curve visualizations are included** in the script.
- The processed dataset is saved as **'processed_dataset.csv'**.

## 📜 License  
This project is open-source and available under the **MIT License**.


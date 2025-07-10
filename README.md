# 💳 Financial Fraud Detection System

A machine learning project for detecting fraudulent transactions in financial systems. This solution includes robust preprocessing, model training, and an interactive Streamlit dashboard for real-time and batch fraud prediction.

---

## 🚀 Overview

This project leverages real-world transaction data to classify whether a transaction is fraudulent or legitimate using machine learning techniques. It provides:

- Cleaned and feature-engineered dataset
- Supervised learning model (Random Forest)
- Real-time predictions via a **Streamlit UI**
- Batch predictions with CSV upload
- Risk scoring and detailed evaluation metrics

---

## 📁 Project Structure

```
fraud-detection/
│
├── dashboard/                 # Streamlit frontend
│   └── dashboard.py
│
├── dataprocessing/                 # Python scripts for core ML pipeline
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── prediction.py
│
├── data/
|   └── archive/               #Original dataset
│   └── processed/             # Cleaned & transformed datasets
│
├── model/                     # Saved models and encoders
│   ├── fraud_detection_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
│
├── requirements.txt           # pip-based dependencies
├── .gitignore
└── README.md
```

---

## 🛠️ Installation

### ✅ Clone the Repository

```bash
git clone https://github.com/HartinderSingh/financial-fraud-detection.git
cd fraud-detection
```

### ✅ Set Up Environment


With pip:

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Workflow

### 1. Preprocess Data
```bash
python dataprocessing/data_preprocessing.py
```

### 2. Train Model
```bash
python dataprocessing/model_training.py
```

### 3. Launch Dashboard
```bash
streamlit run dashboard/dashboard.py
```

---

## 🧪 Model Performance

| Metric     | Score   |
|------------|---------|
| Accuracy   | 99.99%  |
| Precision  | 1.0000  |
| Recall     | 0.9976  |
| F1-Score   | 0.9988  |
| ROC-AUC    | 0.9999  |

✅ Excellent performance on highly imbalanced data using Random Forest.

---

## 📊 Features

- Real-time prediction via manual entry form
- Batch fraud detection with CSV upload
- Risk classification (Minimal, Low, Moderate, High)
- Interactive visualizations:
  - Confusion matrix
  - Fraud distribution pie charts
  - Feature importance bar graphs

---

## 📌 Future Work

- Add XGBoost or LightGBM models
- Integrate model explainability tools (SHAP/LIME)
- API-based deployment or integration with alerting systems

---

## 🙌 Contributing

Contributions are welcome! Please fork the repo and submit a pull request with enhancements.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Hartinder Singh**
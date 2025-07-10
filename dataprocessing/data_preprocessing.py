
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("🔹 Loading dataset...")
df = pd.read_csv('../data/archive/PS_20174392719_1491204439457_log.csv')
print(f"✅ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")

columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
df_clean = df.drop(columns=columns_to_drop)
print(f"✅ Dropped columns: {columns_to_drop}")

label_encoder = LabelEncoder()
df_clean['type_encoded'] = label_encoder.fit_transform(df_clean['type'])
print("✅ Label encoded 'type':", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


df_clean['errorBalanceOrig'] = df_clean['newbalanceOrig'] + df_clean['amount'] - df_clean['oldbalanceOrg']

df_clean['errorBalanceDest'] = df_clean['oldbalanceDest'] + df_clean['amount'] - df_clean['newbalanceDest']


df_clean['isPayment'] = (df_clean['type'] == 'PAYMENT').astype(int)
df_clean['isDebit'] = (df_clean['type'] == 'DEBIT').astype(int)
df_clean['isCashOut'] = (df_clean['type'] == 'CASH_OUT').astype(int)
df_clean['isTransfer'] = (df_clean['type'] == 'TRANSFER').astype(int)

print("✅ Feature engineering completed: balance errors and transaction flags added")

feature_columns = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'type_encoded',
    'errorBalanceOrig', 'errorBalanceDest',
    'isPayment', 'isDebit', 'isCashOut', 'isTransfer'
]


X = df_clean[feature_columns]
y = df_clean['isFraud']

print(f"✅ Feature matrix shape: {X.shape}, Target shape: {y.shape}")
print("Fraud cases:", y.sum(), "| Legitimate cases:", len(y) - y.sum())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features scaled using StandardScaler")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean():.4f}, Test fraud rate: {y_test.mean():.4f}")


os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../model", exist_ok=True)

np.save("../data/processed/X_train.npy", X_train)
np.save("../data/processed/X_test.npy", X_test)
np.save("../data/processed/y_train.npy", y_train)
np.save("../data/processed/y_test.npy", y_test)


joblib.dump(scaler, "../model/scaler.pkl")
joblib.dump(label_encoder, "../model/label_encoder.pkl")
joblib.dump(feature_columns, "../model/feature_columns.pkl")


df_clean.to_csv("../data/processed/processed_dataset.csv", index=False)

print("\n📦 Saved:")
print("• Train/test sets → ../data/processed/")
print("• Preprocessors → ../model/")
print("• Processed CSV for Streamlit → ../data/processed/processed_dataset.csv")
print("\n✅ Data preprocessing completed successfully!")

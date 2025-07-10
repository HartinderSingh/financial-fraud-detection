
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

print("🔹 Loading preprocessed data and artifacts...")

X_train = np.load("../data/processed/X_train.npy")
X_test = np.load("../data/processed/X_test.npy")
y_train = np.load("../data/processed/y_train.npy")
y_test = np.load("../data/processed/y_test.npy")

scaler = joblib.load("../model/scaler.pkl")
label_encoder = joblib.load("../model/label_encoder.pkl")
feature_columns = joblib.load("../model/feature_columns.pkl")

print(f"✅ Loaded: X_train {X_train.shape}, y_train {y_train.shape}")

print("\n🔹 Training RandomForestClassifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("✅ Model training completed!")

print("\n🔹 Evaluating model...")

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]


print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")
print(f"Average Precision Score: {avg_precision:.4f}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nConfusion Matrix:")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
print("\n🔹 Saving model artifacts to ../model/...")

os.makedirs("../model", exist_ok=True)

joblib.dump(rf_model, "../model/fraud_detection_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl") 
joblib.dump(label_encoder, "../model/label_encoder.pkl")
joblib.dump(feature_columns, "../model/feature_columns.pkl")

model_metrics = {
    'roc_auc': roc_auc,
    'avg_precision': avg_precision,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'confusion_matrix': cm.tolist(),
    'feature_importance': feature_importance.to_dict(orient='records')
}
joblib.dump(model_metrics, "../model/model_metrics.pkl")

print("✅ All artifacts saved in ../model/")
print("🎉 Model training and evaluation completed successfully!")


import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def load_model_artifacts():
    
    try:
        model = joblib.load('../model/fraud_detection_model.pkl')
        scaler = joblib.load('../model/scaler.pkl')
        encoder = joblib.load('../model/label_encoder.pkl')
        feature_columns = joblib.load('../model/feature_columns.pkl')
        
        print("✅ Model artifacts loaded successfully!")
        return model, scaler, encoder, feature_columns
    except FileNotFoundError as e:
        print(f"❌ Error loading model artifacts: {e}")
        return None, None, None, None

def predict_transaction(input_dict: Dict) -> Tuple[str, float, Dict]:
  
    model, scaler, encoder, feature_columns = load_model_artifacts()
    if model is None:
        return "Error", 0.0, {}

    try:
       
        required_keys = [
            'step', 'type', 'amount', 
            'oldbalanceOrig', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest'
        ]
        for key in required_keys:
            if key not in input_dict:
                raise ValueError(f"Missing key: {key}")

        features = input_dict.copy()

        try:
            features['type_encoded'] = encoder.transform([features['type']])[0]
        except ValueError:
            print(f"⚠️ Unknown type '{features['type']}', using 0")
            features['type_encoded'] = 0

      
        features['errorBalanceOrig'] = features['newbalanceOrig'] + features['amount'] - features['oldbalanceOrig']
        features['errorBalanceDest'] = features['oldbalanceDest'] + features['amount'] - features['newbalanceDest']
        features['isPayment'] = int(features['type'] == 'PAYMENT')
        features['isDebit'] = int(features['type'] == 'DEBIT')
        features['isCashOut'] = int(features['type'] == 'CASH_OUT')
        features['isTransfer'] = int(features['type'] == 'TRANSFER')

      
        input_vector = np.array([features[col] for col in feature_columns]).reshape(1, -1)
        scaled_vector = scaler.transform(input_vector)

      
        pred = model.predict(scaled_vector)[0]
        prob = model.predict_proba(scaled_vector)[0][1]
        label = "FRAUD" if pred == 1 else "LEGITIMATE"

        return label, prob, {col: features[col] for col in feature_columns}

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return "Error", 0.0, {}

def get_risk_level(probability: float) -> str:
    """
    Convert probability to risk level
    """
    if probability < 0.1:
        return "LOW"
    elif probability < 0.3:
        return "MEDIUM"
    elif probability < 0.7:
        return "HIGH"
    else:
        return "VERY HIGH"

if __name__ == "__main__":
    print("="*60)
    print("🔍 TESTING FRAUD PREDICTION MODULE")
    print("="*60)

    example_transactions = [
        {
            'step': 1,
            'type': 'PAYMENT',
            'amount': 9839.64,
            'oldbalanceOrig': 170136.0,
            'newbalanceOrig': 160296.36,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        },
        {
            'step': 1,
            'type': 'CASH_OUT',
            'amount': 181.0,
            'oldbalanceOrig': 181.0,
            'newbalanceOrig': 0.0,
            'oldbalanceDest': 21182.0,
            'newbalanceDest': 0.0
        }
    ]

    for idx, tx in enumerate(example_transactions, 1):
        print(f"\n📋 Transaction {idx}: {tx}")
        label, prob, features = predict_transaction(tx)
        risk = get_risk_level(prob)
        print(f"🔮 Prediction: {label}")
        print(f"🎯 Probability: {prob:.4f}")
        print(f"⚠️ Risk Level: {risk}")
        if label == "FRAUD":
            print("🚨 Potential fraud detected!")
        else:
            print("✅ Transaction appears normal.")

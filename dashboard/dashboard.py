
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict


st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_model_artifacts():
    try:
        base_path = Path(__file__).resolve().parent
        model_dir = base_path.parent / "model"
        model = joblib.load(model_dir / "fraud_detection_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        encoder = joblib.load(model_dir / "label_encoder.pkl")
        feature_columns = joblib.load(model_dir / "feature_columns.pkl")
        metrics = joblib.load(model_dir / "model_metrics.pkl")
        return model, scaler, encoder, feature_columns, metrics
    except FileNotFoundError as e:
        st.error(f"❌ Model artifact missing: {e}")
        return None, None, None, None, None

@st.cache_data
def load_sample_data():
    try:
        base_path = Path(__file__).resolve().parent
        csv_path = base_path.parent / "data" / "processed" / "processed_dataset.csv"
        df = pd.read_csv(csv_path)
        return df.sample(n=min(10000, len(df)))
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at: {csv_path}")
        return None

def predict_transaction_streamlit(input_dict: Dict, model, scaler, encoder, feature_columns):
    try:
        features = input_dict.copy()
        features['type_encoded'] = encoder.transform([features['type']])[0]
        features['errorBalanceOrig'] = features['newbalanceOrig'] + features['amount'] - features['oldbalanceOrg']
        features['errorBalanceDest'] = features['oldbalanceDest'] + features['amount'] - features['newbalanceDest']
        features['isPayment'] = int(features['type'] == 'PAYMENT')
        features['isDebit'] = int(features['type'] == 'DEBIT')
        features['isCashOut'] = int(features['type'] == 'CASH_OUT')
        features['isTransfer'] = int(features['type'] == 'TRANSFER')

        feature_values = [features[col] for col in feature_columns]
        X_input = np.array(feature_values).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        prediction = model.predict(X_scaled)[0]
        fraud_probability = model.predict_proba(X_scaled)[0, 1]

        return prediction, fraud_probability, features, X_scaled
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None

def batch_predict(df, model, scaler, encoder, feature_columns):
    try:
        df = df.copy()
        df['type_encoded'] = encoder.transform(df['type'])
        df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
        df['isPayment'] = (df['type'] == 'PAYMENT').astype(int)
        df['isDebit'] = (df['type'] == 'DEBIT').astype(int)
        df['isCashOut'] = (df['type'] == 'CASH_OUT').astype(int)
        df['isTransfer'] = (df['type'] == 'TRANSFER').astype(int)

        X = df[feature_columns]
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]
        df['prediction'] = preds
        df['fraud_probability'] = probs
        return df
    except Exception as e:
        st.error(f"Batch prediction error: {str(e)}")
        return None

def main():
    st.title("🔍 Fraud Detection Dashboard")
    model, scaler, encoder, feature_columns, metrics = load_model_artifacts()
    if model is None:
        st.stop()

    df = load_sample_data()
    if df is None:
        st.stop()

    page = st.sidebar.radio("Select Page", ["📊 Data Overview", "📈 Model Performance", "✍️ Manual Entry", "📁 Batch Prediction"])

    if page == "📊 Data Overview":
        st.header("Data Overview")
        st.metric("Total Transactions", f"{len(df):,}")
        st.metric("Fraud Cases", f"{df['isFraud'].sum():,}")
        st.metric("Fraud Rate", f"{df['isFraud'].mean()*100:.2f}%")

        st.plotly_chart(px.pie(
            names=['Legitimate', 'Fraud'],
            values=df['isFraud'].value_counts(),
            title="Fraud Distribution",
            color_discrete_map={'Legitimate': 'green', 'Fraud': 'red'}
        ), use_container_width=True)

        st.dataframe(df.head(100))

    elif page == "📈 Model Performance":
        st.header("Model Performance")
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        st.metric("Precision", f"{metrics['precision']:.4f}")
        st.metric("Recall", f"{metrics['recall']:.4f}")
        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")

        cm = np.array(metrics['confusion_matrix'])
        fig_cm = px.imshow(cm, text_auto=True, x=['Legit', 'Fraud'], y=['Legit', 'Fraud'],
                           title="Confusion Matrix", color_continuous_scale='blues')
        st.plotly_chart(fig_cm)

        st.subheader("Top Features")
        feat_df = pd.DataFrame(metrics['feature_importance'])
        st.plotly_chart(px.bar(feat_df.head(10), x='importance', y='feature', orientation='h',
                               title="Top 10 Important Features", color='importance'))
        st.dataframe(feat_df)

    elif page == "✍️ Manual Entry":
        st.header("Transaction Prediction")

        with st.form("predict_form"):
            step = st.number_input("Step", value=1)
            type_ = st.selectbox("Transaction Type", ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
            amount = st.number_input("Amount", value=1000.0)
            oldbalanceOrg = st.number_input("Old Balance (Orig)", value=5000.0)
            newbalanceOrig = st.number_input("New Balance (Orig)", value=4000.0)
            oldbalanceDest = st.number_input("Old Balance (Dest)", value=0.0)
            newbalanceDest = st.number_input("New Balance (Dest)", value=1000.0)

            submitted = st.form_submit_button("Predict")

        if submitted:
            input_dict = {
                'step': step,
                'type': type_,
                'amount': amount,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrig,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest
            }

            pred, prob, _, X_scaled = predict_transaction_streamlit(input_dict, model, scaler, encoder, feature_columns)

            if pred is not None:
                label = "FRAUD" if pred == 1 else "LEGITIMATE"
                risk = "🔴 HIGH RISK" if prob > 0.8 else "🟠 MODERATE RISK" if prob > 0.5 else "🟡 LOW RISK" if prob > 0.3 else "🟢 MINIMAL RISK"

                st.success(f"Prediction: {label}")
                st.info(f"Fraud Probability: {prob:.4f} ({prob*100:.2f}%)")
                st.warning(f"Risk Level: {risk}")

    elif page == "📁 Batch Prediction":
        st.header("📁 Batch Prediction via CSV Upload")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write("📄 Uploaded Data Preview:", input_df.head())

                result_df = batch_predict(input_df, model, scaler, encoder, feature_columns)
                if result_df is not None:
                    st.success("✅ Batch prediction completed!")

                    fraud_count = (result_df['prediction'] == 1).sum()
                    legit_count = (result_df['prediction'] == 0).sum()
                    total = len(result_df)
                    st.metric("Total Transactions", total)
                    st.metric("Fraudulent", fraud_count)
                    st.metric("Legitimate", legit_count)
                    st.metric("Fraud Rate", f"{fraud_count / total * 100:.2f}%")

                    st.plotly_chart(px.pie(
                        names=['Legitimate', 'Fraudulent'],
                        values=[legit_count, fraud_count],
                        title="Batch Fraud Distribution",
                        color_discrete_map={'Legitimate': 'green', 'Fraudulent': 'red'}
                    ))

                    st.download_button("Download Results", result_df.to_csv(index=False).encode('utf-8'),
                                       file_name="batch_predictions.csv", mime="text/csv")
                    st.dataframe(result_df.head(50))
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()

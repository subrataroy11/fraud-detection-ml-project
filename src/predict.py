import joblib
import pandas as pd

model = joblib.load("fraud_detection_pipeline.pkl")

def predict_transaction(input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability
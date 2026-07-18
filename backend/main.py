import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Synchronized Inference Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

METADATA_PATH = os.path.join(os.path.dirname(__file__), "ensemble_model_metadata.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

if os.path.exists(METADATA_PATH) and os.path.exists(SCALER_PATH):
    artifacts = joblib.load(METADATA_PATH)
    scaler = joblib.load(SCALER_PATH)
    base_models = artifacts["base_models"]
    meta_model = artifacts["meta_model"]
    feature_columns = artifacts["feature_columns"]
    demo_mode = False
else:
    base_models, meta_model, feature_columns, scaler = None, None, None, None
    demo_mode = True

class PatientData(BaseModel):
    age_years: float
    gender: int
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: float
    gluc: float
    smoke: float
    alco: float
    active: float

@app.post("/api/v1/predict")
async def run_prediction(patient: PatientData):
    # Dynamic recalculations to match step 3 feature rules mapping
    bmi = patient.weight / (patient.height / 100.0) ** 2
    chol_gluc_ratio = patient.cholesterol / patient.gluc
    ap_ratio = patient.ap_hi * patient.ap_lo
    gender_mapped = patient.gender - 1

    if demo_mode:
        return {"status": "demo", "risk_score": 45.0, "high_risk_classification": False, "calculated_bmi": round(bmi, 2), "primary_risk_drivers": ["Weights Missing"]}

    try:
        # Construct raw dataset matching structure columns exactly
        raw_dict = {
            "age": patient.age_years, "gender": gender_mapped, "height": patient.height,
            "weight": patient.weight, "ap_hi": patient.ap_hi, "ap_lo": patient.ap_lo,
            "cholesterol": patient.cholesterol, "gluc": patient.gluc, "smoke": patient.smoke,
            "alco": patient.alco, "active": patient.active, "bmi": bmi,
            "chol_gluc_ratio": chol_gluc_ratio, "ap_ratio": ap_ratio
        }

        # Arrange dict keys strictly to line up with the trained configuration order dataframe
        df_input = pd.DataFrame([raw_dict])[feature_columns]
        
        # Scale values using fitted MinMax configuration array 
        scaled_values = scaler.transform(df_input)
        df_scaled = pd.DataFrame(scaled_values, columns=feature_columns)

        # Re-build out-of-fold level 1 meta-features from your saved base array
        meta_features = []
        for name, model in base_models:
            pred_proba = model.predict_proba(df_scaled)[0, 1]
            meta_features.append(pred_proba)

        # Stack predictions with the standard raw feature values
        meta_vector = np.hstack([np.array([meta_features]), df_scaled.values])

        # Execute final meta-inference call
        probability = float(meta_model.predict_proba(meta_vector)[0, 1])
        classification = int(meta_model.predict(meta_vector)[0])

        drivers = []
        if patient.ap_hi >= 140 or patient.ap_lo >= 90: drivers.append("High Blood Pressure (Hypertension)")
        if patient.cholesterol > 1: drivers.append("Elevated Cholesterol Levels")
        if bmi >= 25.0: drivers.append("Elevated Body Mass Index (BMI)")
        if chol_gluc_ratio > 1.0: drivers.append("High Metabolic Lipid-to-Sugar Ratio")

        return {
            "status": "success",
            "risk_score": round(probability * 100, 2),
            "high_risk_classification": bool(classification),
            "calculated_bmi": round(bmi, 2),
            "primary_risk_drivers": drivers if drivers else ["Vitals fall within normal tolerances."]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

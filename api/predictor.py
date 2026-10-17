import joblib
import pandas as pd
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "models" / "xgboost.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler_enrichi.pkl"

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

SEUIL_ALERTE = 0.65

COLONNES = [
    's2','s3','s4','s7','s8','s9','s11','s12','s13','s14',
    's15','s17','s20','s21',
    's15_mean30','s15_std30','s15_slope',
    's8_mean30','s8_std30','s8_slope',
    's13_mean30','s13_std30','s13_slope',
    's21_mean30','s21_std30','s21_slope',
    's20_mean30','s20_std30','s20_slope',
    's11_mean30','s11_std30','s11_slope',
    's2_mean30','s2_std30','s2_slope',
    's12_mean30','s12_std30','s12_slope',
    's7_mean30','s7_std30','s7_slope',
    's17_mean30','s17_std30','s17_slope',
    's3_mean30','s3_std30','s3_slope',
    's4_mean30','s4_std30','s4_slope',
    's14_mean30','s14_std30','s14_slope',
    's9_mean30','s9_std30','s9_slope',
    's15_rms','s15_fft_max',
    's8_rms','s8_fft_max',
    's13_rms','s13_fft_max',
    's21_rms','s21_fft_max',
    's20_rms','s20_fft_max'
]

def predict(data: dict) -> dict:
    # Construire le DataFrame avec toutes les colonnes
    # Les valeurs manquantes sont remplacées par 0.0
    row = {col: float(data.get(col, 0.0)) for col in COLONNES}

    features        = pd.DataFrame([row], columns=COLONNES)
    features_scaled = scaler.transform(features)
    proba_panne     = model.predict_proba(features_scaled)[0][1]
    alerte          = proba_panne >= SEUIL_ALERTE

    return {
        "probabilite": round(float(proba_panne), 4),
        "alerte":      alerte,
        "prediction":  "panne_imminente" if alerte else "normal"
    }

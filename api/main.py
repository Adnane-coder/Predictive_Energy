from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas   import SensorReading, PredictionResult
from predictor import predict

app = FastAPI(
    title       = "PredictiveEnergy API",
    description = "Détection de pannes IIoT — XGBoost 65 features",
    version     = "2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/")
def root():
    return {"message": "PredictiveEnergy API opérationnelle ✓"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost", "features": 65}

@app.post("/predict", response_model=PredictionResult)
def predict_fault(data: SensorReading):
    try:
        # Convertir en dict sans device_id
        input_data = data.model_dump(exclude={"device_id"})
        result     = predict(input_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    message = (
        f" ALERTE panne imminente — probabilité {result['probabilite']*100:.1f}%"
        if result["alerte"]
        else
        f"Fonctionnement normal — probabilité panne {result['probabilite']*100:.1f}%"
    )

    return PredictionResult(
        device_id   = data.device_id,
        prediction  = result["prediction"],
        probabilite = result["probabilite"],
        alerte      = result["alerte"],
        message     = message
    )
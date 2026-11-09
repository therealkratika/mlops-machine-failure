
import joblib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

model = None


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

def load_model():
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run python -m src.train first."
        )

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")


# --------------------------------------------------
# APPLICATION LIFESPAN
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Machine Failure Prediction using "
        "Random Forest and MLflow"
    ),
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------
# INPUT SCHEMA
# --------------------------------------------------

class MachineInput(BaseModel):

    Type: Literal["L", "M", "H"] = Field(
        ...,
        description="Machine product type"
    )

    air_temperature: float = Field(
        ...,
        gt=0,
        description="Air temperature in Kelvin"
    )

    process_temperature: float = Field(
        ...,
        gt=0,
        description="Process temperature in Kelvin"
    )

    rotational_speed: int = Field(
        ...,
        gt=0,
        description="Rotational speed in RPM"
    )

    torque: float = Field(
        ...,
        gt=0,
        description="Torque in Nm"
    )

    tool_wear: int = Field(
        ...,
        ge=0,
        description="Tool wear in minutes"
    )


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Predictive Maintenance API",
        "status": "running",
        "documentation": "/docs"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": "RandomForestClassifier"
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_failure(data: MachineInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    try:

        # Convert input into model-compatible format
        input_data = [{
            "Type": data.Type,
            "air_temperature": data.air_temperature,
            "process_temperature": data.process_temperature,
            "rotational_speed": data.rotational_speed,
            "torque": data.torque,
            "tool_wear": data.tool_wear
        }]

        import pandas as pd

        input_df = pd.DataFrame(input_data)

        # Predict machine failure
        prediction = int(model.predict(input_df)[0])

        # Calculate failure probability
        probabilities = model.predict_proba(input_df)[0]

        classes = list(model.classes_)

        failure_index = classes.index(1)

        failure_probability = float(
            probabilities[failure_index]
        )

        return {
            "prediction": prediction,
            "failure_detected": prediction == 1,
            "failure_probability": round(
                failure_probability, 4
            ),
            "message": (
                "Machine failure detected!"
                if prediction == 1
                else "Machine operating normally."
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

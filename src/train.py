
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.data_loader import load_data
from src.preprocessing import prepare_data


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"

MODEL_NAME = "machine-failure-classifier"
EXPERIMENT_NAME = "machine-failure-classification"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:7001"
)

# Model quality thresholds
MIN_F1 = 0.25
MIN_RECALL = 0.50

# Random Forest configuration
MODEL_PARAMS = {
    "n_estimators": 100,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    print("\nEvaluation Metrics:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return metrics, cm


# --------------------------------------------------
# QUALITY GATE
# --------------------------------------------------

def check_model_quality(metrics):

    print("\nChecking model quality...")

    if (
        metrics["f1_score"] < MIN_F1
        or metrics["recall"] < MIN_RECALL
    ):
        raise ValueError(
            "Model rejected: Quality gate failed! "
            f"F1={metrics['f1_score']:.4f}, "
            f"Recall={metrics['recall']:.4f}"
        )

    print("Model passed quality gate!")


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

def train_model():

    print("\nLoading dataset...")

    df = load_data()

    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_data(df)
    )

    # Create complete preprocessing + model pipeline
    print("\nCreating model pipeline...")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(**MODEL_PARAMS)
            )
        ]
    )

    # Train model
    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Training completed!")

    # Evaluate model
    print("\nEvaluating model...")

    metrics, cm = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Quality gate
    check_model_quality(metrics)

    # --------------------------------------------------
    # MLFLOW CONFIGURATION
    # --------------------------------------------------

    print("\nConnecting to MLflow...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    # --------------------------------------------------
    # MLFLOW EXPERIMENT TRACKING
    # --------------------------------------------------

    print("\nLogging experiment to MLflow...")

    with mlflow.start_run() as run:

        # Log model parameters
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            **MODEL_PARAMS,
            "test_size": 0.20,
            "quality_min_f1": MIN_F1,
            "quality_min_recall": MIN_RECALL
        })

        # Log evaluation metrics
        mlflow.log_metrics(metrics)

        # Log confusion matrix
        mlflow.log_dict(
            {
                "confusion_matrix": cm.tolist()
            },
            "evaluation/confusion_matrix.json"
        )

        # Log classification report
        predictions = model.predict(X_test)

        mlflow.log_dict(
            classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0
            ),
            "evaluation/classification_report.json"
        )

        # Log and register model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        run_id = run.info.run_id

        print(f"\nMLflow Run ID: {run_id}")
        print(f"Model URI: {model_info.model_uri}")

    # --------------------------------------------------
    # MODEL REGISTRATION
    # --------------------------------------------------

    print("\nRegistering model...")

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    matching_versions = [
        version
        for version in versions
        if version.run_id == run_id
    ]

    if not matching_versions:
        raise RuntimeError(
            "Model registration failed in MLflow."
        )

    latest_version = max(
        matching_versions,
        key=lambda version: int(version.version)
    )

    # Assign champion alias
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=latest_version.version
    )

    print(
        f"Registered model version: "
        f"{latest_version.version}"
    )

    print("Champion alias assigned!")

    # --------------------------------------------------
    # SAVE MODEL LOCALLY
    # --------------------------------------------------

    print("\nSaving model locally...")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved at: {MODEL_PATH}")

    print("\nTraining pipeline completed successfully!")

    return metrics


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    try:
        train_model()

    except Exception as error:
        print(f"\nTraining pipeline failed: {error}")
        raise

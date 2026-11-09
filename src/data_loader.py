
from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset location
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"


# Original dataset columns mapped to simpler names
COLUMN_MAPPING = {
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
}


REQUIRED_COLUMNS = [
    "Type",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "machine_failure",
]


def load_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load and validate the AI4I predictive maintenance dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Rename selected columns
    df = df.rename(columns=COLUMN_MAPPING)

    # Validate required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check missing values in required columns
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values in required columns."
        )

    # Validate target values
    if not set(df["machine_failure"].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print("\nFailure distribution:")
    print(df["machine_failure"].value_counts())

    return df


if __name__ == "__main__":
    df = load_data()
    print("\nDataset preview:")
    print(df.head())

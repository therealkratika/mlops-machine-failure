
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# Features selected for model training
CATEGORICAL_FEATURES = [
    "Type"
]

NUMERICAL_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear"
]

TARGET = "machine_failure"


def create_preprocessor():
    """
    Create preprocessing transformer.

    Categorical:
        OneHotEncoder

    Numerical:
        Pass through without scaling.
        Random Forest does not require feature scaling.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_data(df):
    """
    Separate features and target,
    then perform stratified train-test split.
    """

    # Select only approved features
    feature_columns = (
        CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    )

    X = df[feature_columns].copy()
    y = df[TARGET].copy()

    # Stratified split preserves class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = create_preprocessor()

    print("Preprocessing preparation completed!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test, preprocessor

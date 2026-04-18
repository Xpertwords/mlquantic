from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import COMMON_FEATURE_COLUMNS, TARGET_COLUMN


def get_feature_columns() -> List[str]:
    return COMMON_FEATURE_COLUMNS


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    X = df[get_feature_columns()].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_numeric_preprocessor(scale: bool = True) -> ColumnTransformer:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, get_feature_columns()),
        ],
        remainder="drop",
    )

    return preprocessor


def get_model_preprocessor(model_name: str) -> ColumnTransformer:
    tree_models = {
        "decision_tree",
        "random_forest",
        "extra_trees",
        "grad_boost",
        "hist_grad_boost",
    }

    if model_name in tree_models:
        return build_numeric_preprocessor(scale=False)

    return build_numeric_preprocessor(scale=True)


def validate_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    required = get_feature_columns()
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    cleaned = df[required].copy()

    for col in required:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    return cleaned
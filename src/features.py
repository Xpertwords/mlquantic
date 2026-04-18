from typing import List

import numpy as np
import pandas as pd

from src.config import COMMON_FEATURE_COLUMNS


def get_skewed_columns(df: pd.DataFrame, threshold: float = 1.0) -> List[str]:
    numeric_df = df[COMMON_FEATURE_COLUMNS].select_dtypes(include=[np.number])
    skew_values = numeric_df.skew(numeric_only=True)
    skewed = skew_values[skew_values.abs() > threshold].index.tolist()
    return skewed


def log_transform_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    transformed = df.copy()

    for col in columns:
        if col in transformed.columns:
            transformed[col] = pd.to_numeric(transformed[col], errors="coerce")
            transformed[col] = transformed[col].clip(lower=0)
            transformed[col] = np.log1p(transformed[col])

    return transformed


def apply_basic_feature_engineering(df: pd.DataFrame, use_log_transform: bool = False) -> pd.DataFrame:
    engineered = df.copy()

    if use_log_transform:
        skewed_columns = get_skewed_columns(engineered)
        engineered = log_transform_columns(engineered, skewed_columns)

    return engineered
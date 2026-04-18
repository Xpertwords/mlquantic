import pandas as pd
import pytest

from src.config import COMMON_FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocess import (
    build_numeric_preprocessor,
    split_features_target,
    validate_input_schema,
)


def make_sample_df(include_target: bool = True) -> pd.DataFrame:
    data = {col: [1, 2, 3] for col in COMMON_FEATURE_COLUMNS}
    if include_target:
        data[TARGET_COLUMN] = [0, 1, 0]
    return pd.DataFrame(data)


def test_split_features_target():
    df = make_sample_df(include_target=True)
    X, y = split_features_target(df)

    assert list(X.columns) == COMMON_FEATURE_COLUMNS
    assert y.name == TARGET_COLUMN
    assert len(X) == len(y) == 3


def test_validate_input_schema_success():
    df = make_sample_df(include_target=False)
    validated = validate_input_schema(df)

    assert list(validated.columns) == COMMON_FEATURE_COLUMNS
    assert validated.shape[1] == len(COMMON_FEATURE_COLUMNS)


def test_validate_input_schema_missing_column():
    df = make_sample_df(include_target=False).drop(columns=[COMMON_FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError):
        validate_input_schema(df)


def test_numeric_preprocessor_runs():
    df = make_sample_df(include_target=False)
    preprocessor = build_numeric_preprocessor(scale=True)
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] == len(COMMON_FEATURE_COLUMNS)
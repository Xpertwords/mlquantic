import numpy as np
import pandas as pd

from src.config import COMMON_FEATURE_COLUMNS
from src.models_sklearn import get_all_model_pipelines


def make_training_data(rows: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(rows, len(COMMON_FEATURE_COLUMNS))),
        columns=COMMON_FEATURE_COLUMNS,
    )
    y = pd.Series(rng.integers(0, 2, size=rows))
    return X, y


def test_sklearn_pipeline_prediction():
    X, y = make_training_data()
    pipelines = get_all_model_pipelines()

    model = pipelines["logreg"]
    model.fit(X, y)

    preds = model.predict(X.head(5))
    probs = model.predict_proba(X.head(5))[:, 1]

    assert len(preds) == 5
    assert len(probs) == 5
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
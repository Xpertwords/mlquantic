import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from app.app import app
from src.config import BEST_MODEL_FILE, COMMON_FEATURE_COLUMNS, FEATURE_COLUMNS_FILE
from src.preprocess import build_numeric_preprocessor


@pytest.fixture(scope="session", autouse=True)
def prepare_fake_model():
    BEST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_FILE.parent.mkdir(parents=True, exist_ok=True)

    X = pd.DataFrame(
        np.random.rand(30, len(COMMON_FEATURE_COLUMNS)),
        columns=COMMON_FEATURE_COLUMNS
    )
    y = np.random.randint(0, 2, size=30)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_numeric_preprocessor(scale=False)),
            ("model", RandomForestClassifier(n_estimators=20, random_state=42)),
        ]
    )
    pipeline.fit(X, y)

    joblib.dump(pipeline, BEST_MODEL_FILE)

    with open(FEATURE_COLUMNS_FILE, "w", encoding="utf-8") as f:
        json.dump(COMMON_FEATURE_COLUMNS, f, indent=2)

    yield


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predict_endpoint(client):
    payload = {col: "1.0" for col in COMMON_FEATURE_COLUMNS}

    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    assert b"Prediction Result" in response.data


def test_batch_upload_endpoint(client):
    sample_df = pd.DataFrame([{col: 1.0 for col in COMMON_FEATURE_COLUMNS} for _ in range(3)])
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/batch",
        data={"file": (io.BytesIO(csv_bytes), "sample.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Prediction Results Preview" in response.data
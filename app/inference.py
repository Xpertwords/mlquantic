import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from src.config import BEST_MODEL_FILE, FEATURE_COLUMNS_FILE, TARGET_COLUMN
from src.preprocess import validate_input_schema


class InferenceService:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.load_artifacts()

    def load_artifacts(self) -> None:
        if not BEST_MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Trained model not found at: {BEST_MODEL_FILE}. "
                "Run training first using src/train.py"
            )

        if not FEATURE_COLUMNS_FILE.exists():
            raise FileNotFoundError(
                f"Feature schema file not found at: {FEATURE_COLUMNS_FILE}"
            )

        self.model = joblib.load(BEST_MODEL_FILE)

        with open(FEATURE_COLUMNS_FILE, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        validated_df = validate_input_schema(df)

        probabilities = self.model.predict_proba(validated_df)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        result_df = df.copy()
        result_df["prediction_probability"] = probabilities
        result_df["prediction_label"] = predictions

        return result_df

    def evaluate_if_labeled(self, result_df: pd.DataFrame) -> Dict:
        if TARGET_COLUMN not in result_df.columns:
            return {}

        y_true = pd.to_numeric(result_df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
        y_prob = result_df["prediction_probability"]
        y_pred = result_df["prediction_label"]

        auc = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        return {
            "auc": round(float(auc), 6),
            "accuracy": round(float(acc), 6),
            "confusion_matrix": cm.tolist(),
        }

    def predict_csv_file(self, file_path: Path) -> Tuple[pd.DataFrame, Dict]:
        df = pd.read_csv(file_path)
        result_df = self.predict_dataframe(df)
        metrics = self.evaluate_if_labeled(result_df)
        return result_df, metrics
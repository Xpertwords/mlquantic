from typing import Dict

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import SEED
from src.preprocess import get_model_preprocessor


def get_sklearn_models() -> Dict[str, object]:
    return {
        "logreg": LogisticRegression(
            max_iter=3000,
            random_state=SEED,
            class_weight="balanced",
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=SEED,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "grad_boost": GradientBoostingClassifier(
            random_state=SEED,
        ),
        "hist_grad_boost": HistGradientBoostingClassifier(
            random_state=SEED,
            max_iter=300,
        ),
    }


def build_model_pipeline(model_name: str, model: object) -> Pipeline:
    preprocessor = get_model_preprocessor(model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def get_all_model_pipelines() -> Dict[str, Pipeline]:
    models = get_sklearn_models()
    pipelines = {
        name: build_model_pipeline(name, model)
        for name, model in models.items()
    }
    return pipelines


def get_scoring() -> Dict[str, str]:
    return {
        "auc": "roc_auc",
        "accuracy": "accuracy",
    }
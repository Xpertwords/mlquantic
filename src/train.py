import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    BEST_MODEL_FILE,
    CV_RESULTS_FILE,
    MODELS_DIR,
    SEED,
)
from src.data_loader import load_processed_train_test, split_and_save_dataset
from src.models_pytorch import cross_validate_pytorch_mlp
from src.models_sklearn import get_all_model_pipelines, get_scoring
from src.preprocess import split_features_target
from src.utils import set_global_seed


def train_and_compare_models() -> pd.DataFrame:
    set_global_seed(SEED)

    try:
        train_df, _ = load_processed_train_test()
    except Exception:
        train_df, _ = split_and_save_dataset()

    X_train, y_train = split_features_target(train_df)

    results = []
    pipelines = get_all_model_pipelines()
    scoring = get_scoring()
    cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    for model_name, pipeline in pipelines.items():
        cv_result = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        results.append(
            {
                "model_name": model_name,
                "cv_auc_mean": cv_result["test_auc"].mean(),
                "cv_auc_std": cv_result["test_auc"].std(),
                "cv_accuracy_mean": cv_result["test_accuracy"].mean(),
                "cv_accuracy_std": cv_result["test_accuracy"].std(),
            }
        )

    pytorch_result = cross_validate_pytorch_mlp(X_train, y_train, n_splits=10)
    results.append(pytorch_result)

    results_df = pd.DataFrame(results).sort_values(by="cv_auc_mean", ascending=False).reset_index(drop=True)
    CV_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(CV_RESULTS_FILE, index=False)

    return results_df


def fit_best_sklearn_model() -> tuple[str, object]:
    train_df, _ = load_processed_train_test()
    X_train, y_train = split_features_target(train_df)

    results_df = pd.read_csv(CV_RESULTS_FILE)
    pipelines = get_all_model_pipelines()

    ranked_names = results_df["model_name"].tolist()

    best_sklearn_name = None
    for model_name in ranked_names:
        if model_name in pipelines:
            best_sklearn_name = model_name
            break

    if best_sklearn_name is None:
        raise ValueError("No sklearn model available to save for production.")

    best_pipeline = pipelines[best_sklearn_name]
    best_pipeline.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_FILE)

    return best_sklearn_name, best_pipeline


if __name__ == "__main__":
    cv_df = train_and_compare_models()
    print("Cross-validation results:")
    print(cv_df)

    best_name, _ = fit_best_sklearn_model()
    print(f"Saved best production model: {best_name}")
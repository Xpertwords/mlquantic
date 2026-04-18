import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    BEST_MODEL_FILE,
    REPORTS_DIR,
    TEST_RESULTS_FILE,
)
from src.data_loader import load_processed_train_test
from src.preprocess import split_features_target


def evaluate_on_test_set() -> dict:
    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError("Best model file not found. Train the model first.")

    _, test_df = load_processed_train_test()
    X_test, y_test = split_features_target(test_df)

    model = joblib.load(BEST_MODEL_FILE)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    results = {
        "test_auc": float(auc),
        "test_accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEST_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_roc_curve(y_test, y_prob)
    save_confusion_matrix(cm)

    return results


def save_roc_curve(y_true, y_prob) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png")
    plt.close()


def save_confusion_matrix(cm) -> None:
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    results = evaluate_on_test_set()
    print("Final test-set evaluation:")
    print(pd.Series(results))
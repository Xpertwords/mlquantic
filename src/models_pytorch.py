import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import SEED
from src.preprocess import get_model_preprocessor


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: Tuple[int, int] = (64, 32)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 30
    patience: int = 5


class MalwareMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 32), dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchMLPWrapper:
    def __init__(
        self,
        input_dim: int,
        hidden_dims=(64, 32),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 30,
        patience: int = 5,
        device: str | None = None,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MalwareMLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

    def _set_seed(self) -> None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    def _create_batches(self, X: np.ndarray, y: np.ndarray):
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for start in range(0, len(X), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            X_batch = torch.tensor(X[batch_idx], dtype=torch.float32, device=self.device)
            y_batch = torch.tensor(y[batch_idx], dtype=torch.float32, device=self.device).view(-1, 1)
            yield X_batch, y_batch

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        self._set_seed()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_auc = -np.inf
        best_state = None
        best_acc = 0.0
        patience_counter = 0

        for _ in range(self.epochs):
            self.model.train()

            for X_batch, y_batch in self._create_batches(X_train, y_train):
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            val_probs = self.predict_proba(X_val)
            val_preds = (val_probs >= 0.5).astype(int)

            val_auc = roc_auc_score(y_val, val_probs)
            val_acc = accuracy_score(y_val, val_preds)

            if val_auc > best_auc:
                best_auc = val_auc
                best_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {"auc": float(best_auc), "accuracy": float(best_acc)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)


def cross_validate_pytorch_mlp(X, y, n_splits: int = 10) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    auc_scores: List[float] = []
    acc_scores: List[float] = []

    preprocessor = get_model_preprocessor("logreg")

    for train_idx, val_idx in skf.split(X, y):
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)

        model = TorchMLPWrapper(input_dim=X_train.shape[1])
        metrics = model.fit(X_train, y_train, X_val, y_val)

        auc_scores.append(metrics["auc"])
        acc_scores.append(metrics["accuracy"])

    return {
        "model_name": "pytorch_mlp",
        "cv_auc_mean": float(np.mean(auc_scores)),
        "cv_auc_std": float(np.std(auc_scores)),
        "cv_accuracy_mean": float(np.mean(acc_scores)),
        "cv_accuracy_std": float(np.std(acc_scores)),
    }
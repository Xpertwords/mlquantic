import json
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    GOODWARE_FILE,
    MALWARE_FILE,
    COMMON_FEATURE_COLUMNS,
    TARGET_COLUMN,
    COMBINED_DATA_FILE,
    TRAIN_FILE,
    TEST_FILE,
    FEATURE_COLUMNS_FILE,
    PROCESSED_DATA_DIR,
    SEED,
)


def ensure_directories() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not GOODWARE_FILE.exists():
        raise FileNotFoundError(f"Goodware file not found: {GOODWARE_FILE}")

    if not MALWARE_FILE.exists():
        raise FileNotFoundError(f"Malware file not found: {MALWARE_FILE}")

    goodware_df = pd.read_csv(GOODWARE_FILE)
    malware_df = pd.read_csv(MALWARE_FILE)
    return goodware_df, malware_df


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], df_name: str) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def harmonize_datasets() -> pd.DataFrame:
    goodware_df, malware_df = load_raw_datasets()

    validate_required_columns(goodware_df, COMMON_FEATURE_COLUMNS, "goodware.csv")
    validate_required_columns(malware_df, COMMON_FEATURE_COLUMNS + [TARGET_COLUMN], "brazilian-malware.csv")

    good_df = goodware_df[COMMON_FEATURE_COLUMNS].copy()
    good_df[TARGET_COLUMN] = 0

    mal_df = malware_df[COMMON_FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    combined_df = pd.concat([good_df, mal_df], axis=0, ignore_index=True)
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    return combined_df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()

    for col in COMMON_FEATURE_COLUMNS:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    cleaned_df[TARGET_COLUMN] = pd.to_numeric(
        cleaned_df[TARGET_COLUMN], errors="coerce"
    ).fillna(0).astype(int)

    return cleaned_df


def split_and_save_dataset(test_size: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_directories()

    combined_df = harmonize_datasets()
    combined_df = clean_dataset(combined_df)

    train_df, test_df = train_test_split(
        combined_df,
        test_size=test_size,
        stratify=combined_df[TARGET_COLUMN],
        random_state=SEED,
    )

    combined_df.to_csv(COMBINED_DATA_FILE, index=False)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    with open(FEATURE_COLUMNS_FILE, "w", encoding="utf-8") as f:
        json.dump(COMMON_FEATURE_COLUMNS, f, indent=2)

    return train_df, test_df


def load_processed_train_test() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        return split_and_save_dataset()

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = split_and_save_dataset()
    print("Dataset preparation complete.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
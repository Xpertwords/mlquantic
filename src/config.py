from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

GOODWARE_FILE = RAW_DATA_DIR / "goodware.csv"
MALWARE_FILE = RAW_DATA_DIR / "brazilian-malware.csv"

COMBINED_DATA_FILE = PROCESSED_DATA_DIR / "combined_dataset.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
FEATURE_COLUMNS_FILE = PROCESSED_DATA_DIR / "feature_columns.json"
CV_RESULTS_FILE = REPORTS_DIR / "cv_results.csv"
TEST_RESULTS_FILE = REPORTS_DIR / "test_results.json"

BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
PYTORCH_MODEL_FILE = MODELS_DIR / "pytorch_mlp.pt"

SEED = 42
TARGET_COLUMN = "Label"

COMMON_FEATURE_COLUMNS = [
    "BaseOfCode",
    "BaseOfData",
    "Characteristics",
    "DllCharacteristics",
    "Entropy",
    "FileAlignment",
    "Identify",
    "ImageBase",
    "ImportedDlls",
    "ImportedSymbols",
    "Machine",
    "Magic",
    "NumberOfRvaAndSizes",
    "NumberOfSections",
    "NumberOfSymbols",
    "PE_TYPE",
    "PointerToSymbolTable",
    "SHA1",
    "Size",
    "SizeOfCode",
    "SizeOfHeaders",
    "SizeOfImage",
    "SizeOfInitializedData",
    "SizeOfOptionalHeader",
    "SizeOfUninitializedData",
    "TimeDateStamp",
]

EXCLUDED_COLUMNS = [
    "MD5",
    "Name",
    "FormatedTimeDateStamp",
    "FirstSeenDate",
]
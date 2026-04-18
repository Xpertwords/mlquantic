# Malware vs Goodware Classification Project

This project builds and deploys a machine learning system that classifies Portable Executable (PE) files as either:

- `0` = goodware
- `1` = malware

The system includes:

- dataset harmonization for `goodware.csv` and `brazilian-malware.csv`
- preprocessing and training pipelines
- multiple machine learning models
- final model evaluation on a hold-out test set
- Flask web application for manual and batch inference
- CI/CD pipeline with automated testing
- deployment support for Render

## Setup Instructions

### Create virtual environment
```bash
python -m venv .venv
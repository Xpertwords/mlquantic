import io

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from app.inference import InferenceService
from src.config import COMMON_FEATURE_COLUMNS, TARGET_COLUMN

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def get_inference_service():
    try:
        return InferenceService(), None
    except Exception as exc:
        return None, str(exc)


@app.route("/")
def index():
    demo_row = {col: "" for col in COMMON_FEATURE_COLUMNS}
    _, model_error = get_inference_service()

    return render_template(
        "index.html",
        feature_columns=COMMON_FEATURE_COLUMNS,
        demo_row=demo_row,
        model_error=model_error,
    )


@app.route("/predict", methods=["POST"])
def predict():
    service, model_error = get_inference_service()
    if model_error:
        return render_template(
            "results.html",
            error=model_error,
            result=None,
            metrics=None,
            table_html=None,
        )

    try:
        row = {}
        for col in COMMON_FEATURE_COLUMNS:
            value = request.form.get(col, "").strip()
            row[col] = value

        input_df = pd.DataFrame([row])
        result_df = service.predict_dataframe(input_df)
        record = result_df.to_dict(orient="records")[0]

        return render_template(
            "results.html",
            error=None,
            result=record,
            metrics=None,
            table_html=None,
        )
    except Exception as exc:
        return render_template(
            "results.html",
            error=f"Prediction error: {exc}",
            result=None,
            metrics=None,
            table_html=None,
        )


@app.route("/batch", methods=["GET", "POST"])
def batch():
    service, model_error = get_inference_service()

    if request.method == "GET":
        return render_template(
            "batch.html",
            feature_columns=COMMON_FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            model_error=model_error,
            table_html=None,
            metrics=None,
            error=None,
            downloadable_csv=None,
        )

    if model_error:
        return render_template(
            "batch.html",
            feature_columns=COMMON_FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            model_error=model_error,
            table_html=None,
            metrics=None,
            error=model_error,
            downloadable_csv=None,
        )

    try:
        if "file" not in request.files:
            raise ValueError("No file part found in the request.")

        file = request.files["file"]

        if file.filename == "":
            raise ValueError("No file selected.")

        if not file.filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")

        df = pd.read_csv(file)
        result_df = service.predict_dataframe(df)
        metrics = service.evaluate_if_labeled(result_df)

        table_html = result_df.head(100).to_html(
            classes="table table-striped table-bordered table-sm",
            index=False,
        )

        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        return render_template(
            "batch.html",
            feature_columns=COMMON_FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            model_error=None,
            table_html=table_html,
            metrics=metrics,
            error=None,
            downloadable_csv=csv_data,
        )

    except Exception as exc:
        return render_template(
            "batch.html",
            feature_columns=COMMON_FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            model_error=None,
            table_html=None,
            metrics=None,
            error=f"Batch prediction error: {exc}",
            downloadable_csv=None,
        )


@app.route("/download_predictions", methods=["POST"])
def download_predictions():
    try:
        csv_content = request.form.get("csv_data", "")
        if not csv_content:
            raise ValueError("No CSV content available for download.")

        buffer = io.BytesIO(csv_content.encode("utf-8"))
        return send_file(
            buffer,
            as_attachment=True,
            download_name="predictions.csv",
            mimetype="text/csv",
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "results.html",
        error="Uploaded file is too large. Maximum allowed size is 16 MB.",
        result=None,
        metrics=None,
        table_html=None,
    ), 413


@app.errorhandler(500)
def internal_error(_error):
    return render_template(
        "results.html",
        error="Internal server error occurred.",
        result=None,
        metrics=None,
        table_html=None,
    ), 500
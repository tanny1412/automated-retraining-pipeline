import logging
import os
import subprocess
import sys
import urllib.request
import mlflow
from mlflow.tracking import MlflowClient
from evaluate import load_model, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

EPOCHS = int(os.getenv("EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.80"))
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sentiment-classifier")
S3_BUCKET = os.getenv("S3_BUCKET", "ml-pipeline-models-tanish")

def check_drift():
    result = subprocess.run(
        [sys.executable, "drift_detector.py"],
        capture_output=False
    )
    return result.returncode == 1

def get_accuracy(model_path=None):
    model, tokenizer, device = load_model(model_path=model_path)
    accuracy, f1 = evaluate(model, tokenizer, device, batch_size=BATCH_SIZE)
    logger.info(f"F1 (weighted): {f1:.4f}")
    return accuracy

def retrain(epochs=EPOCHS, batch_size=BATCH_SIZE):
    logger.info("Starting retraining...")
    cmd = [sys.executable, "train.py", "--epochs", str(epochs), "--batch-size", str(batch_size)]
    max_samples = os.environ.get("MAX_SAMPLES")
    if max_samples:
        cmd += ["--max-samples", max_samples]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        logger.info("Retraining complete — new model saved to models/")
    else:
        logger.error("Retraining failed")

def upload_model_to_s3(bucket=S3_BUCKET):
    logger.info("Uploading new model weights to S3...")
    result = subprocess.run(
        ["aws", "s3", "cp", "models/", f"s3://{bucket}/models/", "--recursive"],
        capture_output=False
    )
    if result.returncode == 0:
        logger.info("Model weights uploaded to S3")
    else:
        logger.error("S3 upload failed — new weights not persisted to S3")

def promote_latest_to_production(model_name=MLFLOW_MODEL_NAME):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(model_name, "Production", latest.version)
    logger.info(f"Promoted version {latest.version} to Production")

def notify_api_reload(api_url=os.getenv("INFERENCE_SERVICE_URL", "http://localhost:8000")):
    try:
        req = urllib.request.Request(f"{api_url}/reload", method="POST")
        urllib.request.urlopen(req)
        logger.info("API reloaded with new model")
    except Exception as e:
        logger.warning(f"Could not reach API to reload: {e}")

if __name__ == "__main__":
    try:
        MlflowClient().get_model_version_by_alias(MLFLOW_MODEL_NAME, "Production")
    except Exception:
        logger.info("No Production model found — running initial training")
        retrain()
        promote_latest_to_production()
        upload_model_to_s3()
        logger.info("Initial model trained and promoted to Production")

    drift = check_drift()
    production_accuracy = get_accuracy()
    logger.info(f"Production model accuracy: {production_accuracy:.4f}")

    if not drift and production_accuracy >= ACCURACY_THRESHOLD:
        logger.info("No drift, model is healthy — no retraining needed")
    else:
        if drift:
            logger.info("Drift detected — triggering retraining")
        if production_accuracy < ACCURACY_THRESHOLD:
            logger.info("Accuracy below threshold — triggering retraining")
        retrain()
        new_accuracy = get_accuracy(model_path="models/")
        logger.info(f"New model accuracy: {new_accuracy:.4f}, Production accuracy: {production_accuracy:.4f}")
        if new_accuracy > production_accuracy:
            logger.info("New model is better — promoting to Production")
            promote_latest_to_production()
            upload_model_to_s3()
            notify_api_reload()
        else:
            logger.info(f"New model not better than Production — keeping current")

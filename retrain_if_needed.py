import logging
import subprocess
import sys
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def check_drift():
    result = subprocess.run(
        [sys.executable, "drift_detector.py"],
        capture_output=False
    )
    return result.returncode == 1

def check_model_health(threshold=0.90):
    result = subprocess.run(
        [sys.executable, "evaluate.py", "--threshold", str(threshold)],
        capture_output=False
    )
    return result.returncode == 0

def retrain(epochs=3, batch_size=16):
    logger.info("Starting retraining...")
    result = subprocess.run(
        [sys.executable, "train.py", "--epochs", str(epochs), "--batch-size", str(batch_size)],
        capture_output=False
    )
    if result.returncode == 0:
        logger.info("Retraining complete — new model saved to models/")
    else:
        logger.error("Retraining failed")

def upload_model_to_s3(bucket="ml-pipeline-models-tanish"):
    logger.info("Uploading new model weights to S3...")
    result = subprocess.run(
        ["aws", "s3", "cp", "models/", f"s3://{bucket}/models/", "--recursive"],
        capture_output=False
    )
    if result.returncode == 0:
        logger.info("Model weights uploaded to S3")
    else:
        logger.error("S3 upload failed — new weights not persisted to S3")

def notify_api_reload(api_url="http://localhost:8000"):
    try:
        req = urllib.request.Request(f"{api_url}/reload", method="POST")
        urllib.request.urlopen(req)
        logger.info("API reloaded with new model")
    except Exception as e:
        logger.warning(f"Could not reach API to reload: {e}")

if __name__ == "__main__":
    drift = check_drift()
    healthy = check_model_health(threshold=0.80)

    if not drift and healthy:
        logger.info("No drift, model is healthy — no retraining needed")
    else:
        if drift:
            logger.info("Drift detected — triggering retraining")
        if not healthy:
            logger.info("Accuracy below threshold — triggering retraining")
        retrain(epochs=3, batch_size=16)
        upload_model_to_s3()
        notify_api_reload()

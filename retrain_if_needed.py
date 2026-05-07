import logging
import subprocess
import sys
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

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

def notify_api_reload(api_url="http://localhost:8000"):
    try:
        req = urllib.request.Request(f"{api_url}/reload", method="POST")
        urllib.request.urlopen(req)
        logger.info("API reloaded with new model")
    except Exception as e:
        logger.warning(f"Could not reach API to reload: {e}")

if __name__ == "__main__":
    healthy = check_model_health(threshold=0.90)
    if healthy:
        logger.info("Model is healthy — no retraining needed")
    else:
        logger.info("Model needs retraining — triggering train.py")
        retrain(epochs=3, batch_size=16)
        notify_api_reload()

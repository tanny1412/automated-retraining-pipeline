import os
import logging
from collections import Counter
from db.database import SessionLocal
from db.models import Prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BASELINE_CONFIDENCE = float(os.getenv("BASELINE_CONFIDENCE", "0.90"))
BASELINE_DOMINANT_RATIO = float(os.getenv("BASELINE_DOMINANT_RATIO", "0.50"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.10"))
DISTRIBUTION_THRESHOLD = float(os.getenv("DISTRIBUTION_THRESHOLD", "0.20"))
MIN_PREDICTIONS = int(os.getenv("MIN_PREDICTIONS", "10"))

def load_predictions():
    from db.database import Base, engine
    from db.models import Prediction as _P  # ensure model registered before create_all
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        rows = db.query(Prediction).all()
        logger.info(f"Loaded {len(rows)} predictions from database")
        return rows
    finally:
        db.close()

def detect_drift(predictions):
    if len(predictions) < MIN_PREDICTIONS:
        logger.info(f"Not enough predictions to detect drift — need at least {MIN_PREDICTIONS}")
        return False

    confidences = [row.confidence for row in predictions]
    avg_confidence = sum(confidences) / len(confidences)

    label_counts = Counter(row.prediction for row in predictions)
    most_common_count = label_counts.most_common(1)[0][1]
    dominant_ratio = most_common_count / len(predictions)

    confidence_drop = BASELINE_CONFIDENCE - avg_confidence
    distribution_shift = abs(BASELINE_DOMINANT_RATIO - dominant_ratio)

    logger.info(f"Avg confidence: {avg_confidence:.4f} (baseline: {BASELINE_CONFIDENCE}) — drop: {confidence_drop:.4f}")
    logger.info(f"Dominant label ratio: {dominant_ratio:.4f} (baseline: {BASELINE_DOMINANT_RATIO}) — shift: {distribution_shift:.4f}")

    drift_detected = False
    if confidence_drop > CONFIDENCE_THRESHOLD:
        logger.warning(f"Confidence drift detected — dropped {confidence_drop:.4f} from baseline")
        drift_detected = True
    if distribution_shift > DISTRIBUTION_THRESHOLD:
        logger.warning(f"Distribution drift detected — shifted {distribution_shift:.4f} from baseline")
        drift_detected = True

    return drift_detected

if __name__ == "__main__":
    predictions = load_predictions()
    drift = detect_drift(predictions)
    if drift:
        logger.warning("Drift detected — retraining recommended")
        exit(1)
    else:
        logger.info("No drift detected — model is stable")
        exit(0)

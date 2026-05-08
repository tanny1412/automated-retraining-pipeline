import logging
from db.database import SessionLocal
from db.models import Prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BASELINE_CONFIDENCE = 0.90
BASELINE_POSITIVE_RATIO = 0.50
CONFIDENCE_THRESHOLD = 0.10
DISTRIBUTION_THRESHOLD = 0.20

def load_predictions():
    db = SessionLocal()                                                                                                                                                                                                         
    try:                         
        rows = db.query(Prediction).all()                                                                                                                                                                                       
        logger.info(f"Loaded {len(rows)} predictions from database")
        return rows                          
    finally:                                
        db.close() 

def detect_drift(predictions):
    if len(predictions) < 10:
        logger.info("Not enough predictions to detect drift — need at least 10")
        return False

    confidences = [row.confidence for row in predictions]
    avg_confidence = sum(confidences) / len(confidences)

    positive_count = sum(1 for row in predictions if row.prediction == "positive")
    positive_ratio = positive_count / len(predictions)

    confidence_drop = BASELINE_CONFIDENCE - avg_confidence
    distribution_shift = abs(BASELINE_POSITIVE_RATIO - positive_ratio)

    logger.info(f"Avg confidence: {avg_confidence:.4f} (baseline: {BASELINE_CONFIDENCE}) — drop: {confidence_drop:.4f}")
    logger.info(f"Positive ratio: {positive_ratio:.4f} (baseline: {BASELINE_POSITIVE_RATIO}) — shift: {distribution_shift:.4f}")

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

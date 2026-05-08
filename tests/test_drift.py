import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_detector import detect_drift

class FakePrediction:
    def __init__(self, prediction, confidence):
        self.prediction = prediction
        self.confidence = float(confidence)

def make_predictions(count, prediction="positive", confidence=0.95):
    return [FakePrediction(prediction, confidence) for _ in range(count)]

def test_not_enough_predictions():
    predictions = make_predictions(5)
    assert detect_drift(predictions) == False

def test_no_drift_balanced_data():
    positives = make_predictions(50, "positive", 0.93)
    negatives = make_predictions(50, "negative", 0.93)
    assert detect_drift(positives + negatives) == False

def test_confidence_drift():
    predictions = make_predictions(50, "positive", 0.70) + \
                  make_predictions(50, "negative", 0.70)
    assert detect_drift(predictions) == True

def test_distribution_drift():
    predictions = make_predictions(90, "positive", 0.93) + \
                  make_predictions(10, "negative", 0.93)
    assert detect_drift(predictions) == True

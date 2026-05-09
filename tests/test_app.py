import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
import torch

with patch("transformers.AutoTokenizer.from_pretrained"), \
     patch("transformers.AutoModelForSequenceClassification.from_pretrained"), \
     patch("torch.load"), \
     patch("mlflow.artifacts.download_artifacts", return_value="/tmp/mock_model"):
    from fastapi.testclient import TestClient
    import app
    client = TestClient(app.app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_returns_correct_schema():
    mock_logits = torch.tensor([[0.1, 0.9]])
    app.model = MagicMock()
    app.model.return_value = MagicMock(logits=mock_logits)
    app.tokenizer = MagicMock()
    app.tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]),
                                   "attention_mask": torch.tensor([[1, 1, 1]])}

    response = client.post("/predict", json={"text": "this movie was great"})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in ["positive", "negative"]
    assert 0 <= data["confidence"] <= 1

def test_rollback():
    with patch("app.MlflowClient") as mock_client, \
         patch("app.load_model_from_registry", return_value=MagicMock()):
        response = client.post("/rollback", json={"version": "2"})
        assert response.status_code == 200
        assert response.json() == {"status": "rolled back to version 2"}
        mock_client.return_value.set_registered_model_alias.assert_called_once_with(
            "sentiment-classifier", "Production", "2"
        )

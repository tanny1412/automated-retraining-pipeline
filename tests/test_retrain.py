import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock, call
import retrain_if_needed


def test_promote_latest_to_production():
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        MagicMock(version="1"),
        MagicMock(version="3"),
        MagicMock(version="2"),
    ]
    with patch("retrain_if_needed.MlflowClient", return_value=mock_client):
        retrain_if_needed.promote_latest_to_production()
        mock_client.set_registered_model_alias.assert_called_once_with(
            "sentiment-classifier", "Production", "3"
        )


def test_promote_picks_highest_version():
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        MagicMock(version="10"),
        MagicMock(version="9"),
        MagicMock(version="2"),
    ]
    with patch("retrain_if_needed.MlflowClient", return_value=mock_client):
        retrain_if_needed.promote_latest_to_production()
        mock_client.set_registered_model_alias.assert_called_once_with(
            "sentiment-classifier", "Production", "10"
        )

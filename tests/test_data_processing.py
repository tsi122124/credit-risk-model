import pandas as pd
import pytest

import sys
import os

from src.data_processing import (
    DataValidator,
    DateFeatureExtractor,
    create_proxy_target
)

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

# =====================================================
# Test DataValidator
# =====================================================


def test_data_validator_accepts_valid_data():

    df = pd.DataFrame({
        "CustomerId": [1],
        "Amount": [100],
        "Value": [100],
        "TransactionStartTime": ["2024-01-01"],
        "CurrencyCode": ["UGX"],
        "ProviderId": ["Provider1"],
        "ProductId": ["Product1"],
        "ProductCategory": ["airtime"],
        "ChannelId": ["Channel1"],
        "CountryCode": [256],
        "PricingStrategy": [1],
        "FraudResult": [0]
    })

    validator = DataValidator()

    result = validator.transform(df)

    assert result.shape == df.shape


# =====================================================
# Test DataValidator Missing Columns
# =====================================================

def test_data_validator_raises_error_for_missing_columns():

    df = pd.DataFrame({
        "CustomerId": [1],
        "Amount": [100]
    })

    validator = DataValidator()

    with pytest.raises(ValueError):
        validator.transform(df)


# =====================================================
# Test DateFeatureExtractor
# =====================================================

def test_date_feature_extractor_creates_features():

    df = pd.DataFrame({
        "TransactionStartTime": ["2024-05-15 14:30:00"]
    })

    transformer = DateFeatureExtractor()

    result = transformer.transform(df)

    assert "TransactionHour" in result.columns
    assert "TransactionDay" in result.columns
    assert "TransactionMonth" in result.columns
    assert "TransactionYear" in result.columns

    assert result["TransactionHour"].iloc[0] == 14
    assert result["TransactionDay"].iloc[0] == 15
    assert result["TransactionMonth"].iloc[0] == 5
    assert result["TransactionYear"].iloc[0] == 2024


# =====================================================
# Test Proxy Target Creation
# =====================================================

def test_create_proxy_target_returns_expected_columns():

    df = pd.DataFrame({
        "CustomerId": [1, 1, 2, 2, 3, 3],
        "TransactionId": [1, 2, 3, 4, 5, 6],
        "Amount": [100, 150, 200, 250, 50, 60],
        "TransactionStartTime": [
            "2024-01-01",
            "2024-01-05",
            "2024-01-02",
            "2024-01-06",
            "2024-01-03",
            "2024-01-07"
        ]
    })

    result = create_proxy_target(df)

    assert "CustomerId" in result.columns
    assert "is_high_risk" in result.columns

    assert len(result) == 3

    assert set(result["is_high_risk"].unique()).issubset({0, 1})

"""
Data Processing Pipeline

Task 3:
Feature Engineering Pipeline for Credit Risk Modeling
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# =====================================================
# Configure Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =====================================================
# Load Data Function
# =====================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load raw transaction data.
    """

    try:

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(
                "Dataset is empty."
            )

        logging.info(
            f"Dataset loaded successfully: {df.shape}"
        )

        return df

    except FileNotFoundError:

        logging.error(
            f"File not found: {path}"
        )

        raise

    except Exception as e:

        logging.error(e)

        raise


# =====================================================
# Data Validation Transformer
# =====================================================

class DataValidator(
    BaseEstimator,
    TransformerMixin
):
    """
    Validate required columns before processing.
    """

    REQUIRED_COLUMNS = [
        "CustomerId",
        "Amount",
        "Value",
        "TransactionStartTime",
        "CurrencyCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "CountryCode",
        "PricingStrategy",
        "FraudResult"
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        missing_cols = [
            col
            for col in self.REQUIRED_COLUMNS
            if col not in X.columns
        ]

        if missing_cols:

            raise ValueError(
                f"Missing required columns: {missing_cols}"
            )

        return X


# =====================================================
# Date Feature Extraction
# =====================================================

class DateFeatureExtractor(
    BaseEstimator,
    TransformerMixin
):
    """
    Extract date-based features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["TransactionStartTime"] = pd.to_datetime(
            X["TransactionStartTime"]
        )

        X["TransactionHour"] = (
            X["TransactionStartTime"].dt.hour
        )

        X["TransactionDay"] = (
            X["TransactionStartTime"].dt.day
        )

        X["TransactionMonth"] = (
            X["TransactionStartTime"].dt.month
        )

        X["TransactionYear"] = (
            X["TransactionStartTime"].dt.year
        )

        return X


# =====================================================
# Aggregate Customer Features
# =====================================================

class AggregateFeatures(
    BaseEstimator,
    TransformerMixin
):
    """
    Create customer-level aggregate features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        agg_df = (
            X.groupby("CustomerId")
            .agg(
                TotalTransactionAmount=(
                    "Amount",
                    "sum"
                ),
                AverageTransactionAmount=(
                    "Amount",
                    "mean"
                ),
                TransactionCount=(
                    "Amount",
                    "count"
                ),
                StdTransactionAmount=(
                    "Amount",
                    "std"
                )
            )
            .reset_index()
        )

        X = X.merge(
            agg_df,
            on="CustomerId",
            how="left"
        )

        return X


# =====================================================
# Feature Lists
# =====================================================

categorical_features = [
    "CurrencyCode",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId"
]

numerical_features = [
    "Amount",
    "Value",
    "CountryCode",
    "PricingStrategy",
    "FraudResult",
    "TransactionHour",
    "TransactionDay",
    "TransactionMonth",
    "TransactionYear",
    "TotalTransactionAmount",
    "AverageTransactionAmount",
    "TransactionCount",
    "StdTransactionAmount"
]


# =====================================================
# Numerical Pipeline
# =====================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# =====================================================
# Categorical Pipeline
# =====================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# =====================================================
# Column Transformer
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# =====================================================
# Full Pipeline
# =====================================================

full_pipeline = Pipeline(
    steps=[
        (
            "validation",
            DataValidator()
        ),
        (
            "date_features",
            DateFeatureExtractor()
        ),
        (
            "aggregate_features",
            AggregateFeatures()
        ),
        (
            "preprocessing",
            preprocessor
        )
    ]
)


# =====================================================
# Main Execution
# =====================================================

if __name__ == "__main__":

    RAW_DATA_PATH = (
        "data/raw/data.csv"
    )

    PROCESSED_DATA_PATH = (
        "data/processed/processed_data.csv"
    )

    PIPELINE_PATH = (
        "models/feature_pipeline.pkl"
    )

    try:

        # Load dataset
        df = load_data(
            RAW_DATA_PATH
        )

        # Run pipeline
        processed_data = (
            full_pipeline.fit_transform(df)
        )

        # Extract feature names
        feature_names = (
            full_pipeline
            .named_steps["preprocessing"]
            .get_feature_names_out()
        )

        # Convert to DataFrame
        processed_df = pd.DataFrame(
            processed_data,
            columns=feature_names
        )

        # Create folders
        os.makedirs(
            "data/processed",
            exist_ok=True
        )

        os.makedirs(
            "models",
            exist_ok=True
        )

        # Save processed dataset
        processed_df.to_csv(
            PROCESSED_DATA_PATH,
            index=False
        )

        # Save fitted pipeline
        joblib.dump(
            full_pipeline,
            PIPELINE_PATH
        )

        logging.info(
            f"Processed dataset saved to "
            f"{PROCESSED_DATA_PATH}"
        )

        logging.info(
            f"Pipeline saved to "
            f"{PIPELINE_PATH}"
        )

        print(
            "\nProcessed Data Shape:",
            processed_df.shape
        )

        print(
            "\nFirst 10 Feature Names:"
        )

        print(
            processed_df.columns[:10]
        )

    except Exception as e:

        logging.error(
            f"Pipeline failed: {e}"
        )

        raise
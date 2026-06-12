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
from sklearn.cluster import KMeans


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
# Task 4 - Proxy Target Engineering
# =====================================================

def create_proxy_target(df):
    """
    Create high-risk proxy target using
    RFM analysis and KMeans clustering.
    """

    logging.info(
        "Creating proxy target variable..."
    )

    rfm_df = df.copy()

    rfm_df["TransactionStartTime"] = pd.to_datetime(
        rfm_df["TransactionStartTime"]
    )

    snapshot_date = (
        rfm_df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        rfm_df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=(
                "TransactionId",
                "count"
            ),
            Monetary=(
                "Amount",
                "sum"
            )
        )
        .reset_index()
    )

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    cluster_summary = (
        rfm.groupby("Cluster")
        [
            [
                "Recency",
                "Frequency",
                "Monetary"
            ]
        ]
        .mean()
    )

    logging.info(
        "\nCluster Summary:\n%s",
        cluster_summary
    )

    high_risk_cluster = (
        cluster_summary["Frequency"]
        .idxmin()
    )

    rfm["is_high_risk"] = np.where(
        rfm["Cluster"] == high_risk_cluster,
        1,
        0
    )

    logging.info(
        f"High-risk cluster: {high_risk_cluster}"
    )

    return rfm[
        [
            "CustomerId",
            "is_high_risk"
        ]
    ]

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

        # ==========================================
        # Task 4 - Create Proxy Target
        # ==========================================

        target_df = create_proxy_target(
            df
        )

        df = df.merge(
            target_df,
            on="CustomerId",
            how="left"
        )

        # ==========================================
        # Run Feature Engineering Pipeline
        # ==========================================

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

        # Add target column
        processed_df["is_high_risk"] = (
            df["is_high_risk"].values
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

        # Save pipeline
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
            "\nTarget Distribution:"
        )

        print(
            processed_df["is_high_risk"]
            .value_counts()
        )

    except Exception as e:

        logging.error(
            f"Pipeline failed: {e}"
        )

        raise

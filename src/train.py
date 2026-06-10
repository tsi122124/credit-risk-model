import os
import joblib
import logging
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from mlflow.tracking import MlflowClient

# =====================================================
# Configure Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =====================================================
# Load Processed Dataset
# =====================================================

DATA_PATH = (
    "data/processed/processed_data.csv"
)

df = pd.read_csv(DATA_PATH)

logging.info(
    f"Dataset loaded successfully: {df.shape}"
)


# =====================================================
# Features and Target
# =====================================================

X = df.drop(
    columns=["is_high_risk"]
)

y = df["is_high_risk"]


# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

logging.info(
    "Train-test split completed."
)


# =====================================================
# Models
# =====================================================

models = {

    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "DecisionTree": DecisionTreeClassifier(
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    )
}


# =====================================================
# Hyperparameter Grids
# =====================================================

param_grids = {

    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10]
    },

    "DecisionTree": {
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5]
    },

    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10]
    },

    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1]
    }
}


# =====================================================
# MLflow Setup
# =====================================================

os.makedirs(
    "mlruns",
    exist_ok=True
)

mlflow.set_tracking_uri(
    "file:./mlruns"
)

mlflow.set_experiment(
    "credit_risk_modeling"
)


# =====================================================
# Training Loop
# =====================================================

results = []

best_auc = 0

best_model = None

best_model_name = None


for model_name, model in models.items():

    logging.info(
        f"Training {model_name}"
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_estimator = (
        grid_search.best_estimator_
    )

    y_pred = best_estimator.predict(
        X_test
    )

    y_prob = best_estimator.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    with mlflow.start_run(
        run_name=model_name
    ):

        mlflow.log_params(
            grid_search.best_params_
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        mlflow.sklearn.log_model(
            best_estimator,
            "model"
        )

    results.append({

        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc

    })

    if roc_auc > best_auc:

        best_auc = roc_auc

        best_model = best_estimator

        best_model_name = model_name


# =====================================================
# Results Summary
# =====================================================

results_df = pd.DataFrame(
    results
)

print("\nModel Comparison:\n")

print(results_df)


# =====================================================
# Save Best Model
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

logging.info(
    "Best model saved successfully."
)


# =====================================================
# Register Best Model in MLflow
# =====================================================

with mlflow.start_run(
    run_name="Best_Model"
) as run:

    mlflow.sklearn.log_model(
        best_model,
        name="best_model"
    )

    mlflow.log_metric(
        "best_roc_auc",
        best_auc
    )

    model_uri = (
        f"runs:/{run.info.run_id}/best_model"
    )

    mlflow.register_model(
        model_uri=model_uri,
        name="credit_risk_model"
    )

logging.info(
    "Model registered successfully."
)


# =====================================================
# Final Output
# =====================================================

print(
    f"\nBest Model: {best_model_name}"
)

print(
    f"Best ROC-AUC: {best_auc:.4f}"
)
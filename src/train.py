import os
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

from xgboost import XGBRegressor

from preprocessing import (
    wrangle,
    feat_gengeenering
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv("../data/train.csv")


# ============================================================
# PREPROCESSING
# ============================================================

df = feat_gengeenering(
    wrangle(df)
)


# ============================================================
# X AND y
# ============================================================

X = df.drop(
    columns=[
        "SalePrice",
        "LogSalePrice",
        "Id"
    ]
)

y = df["LogSalePrice"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()


# ============================================================
# NUMERIC PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# XGBOOST
# ============================================================

xgb_pipeline = Pipeline(
    [
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# PARAMETERS TO TEST
# ============================================================

param_grid = {

    "model__n_estimators": [
        300,
        500
    ],

    "model__max_depth": [
        3,
        4,
        5
    ],

    "model__learning_rate": [
        0.03,
        0.05,
        0.1
    ],

    "model__subsample": [
        0.8,
        1.0
    ]
}


# ============================================================
# GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)


print("Training XGBoost...")

grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# BEST MODEL
# ============================================================

model = grid_search.best_estimator_


print("\nBest parameters:")
print(grid_search.best_params_)

print(
    f"Best CV R²: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# PREDICTIONS
# ============================================================

train_predictions = model.predict(
    X_train
)

test_predictions = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

training_r2 = r2_score(
    y_train,
    train_predictions
)

testing_r2 = r2_score(
    y_test,
    test_predictions
)

training_mae = mean_absolute_error(
    y_train,
    train_predictions
)

testing_mae = mean_absolute_error(
    y_test,
    test_predictions
)

training_rmse = root_mean_squared_error(
    y_train,
    train_predictions
)

testing_rmse = root_mean_squared_error(
    y_test,
    test_predictions
)


print("\nMODEL PERFORMANCE")

print(
    f"Training R²: {training_r2:.4f}"
)

print(
    f"Testing R²: {testing_r2:.4f}"
)

print(
    f"Training MAE: {training_mae:.4f}"
)

print(
    f"Testing MAE: {testing_mae:.4f}"
)

print(
    f"Training RMSE: {training_rmse:.4f}"
)

print(
    f"Testing RMSE: {testing_rmse:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = "../models/house_price_model.pkl"

with open(
    model_path,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print(
    "\nModel saved:"
)

print(model_path)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model": "Tuned XGBoost",

    "training_r2": training_r2,

    "testing_r2": testing_r2,

    "training_mae": training_mae,

    "testing_mae": testing_mae,

    "training_rmse": training_rmse,

    "testing_rmse": testing_rmse,

    "cross_validation_r2_mean":
        grid_search.best_score_,

    "best_parameters":
        grid_search.best_params_
}


with open(
    "../results/metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print(
    "Metrics saved successfully."
)

print("\nTraining completed.")

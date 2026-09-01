import os
import json

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

from xgboost import XGBRegressor

from preprocessing import (
    wrangle,
    feat_gengeenering,
    prepare_features
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    "../data/train.csv"
)

print("Dataset shape:", df.shape)


# ============================================================
# CLEAN DATA
# ============================================================

print("\nCleaning data...")

df = wrangle(df)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("Creating features...")

df = feat_gengeenering(df)


# ============================================================
# TARGET
# ============================================================

X = prepare_features(df)

y = df["LogSalePrice"]


print("\nFinal feature shape:", X.shape)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)


print("\nTraining:", X_train.shape)
print("Testing :", X_test.shape)


# ============================================================
# XGBOOST
# ============================================================

xgb_model = XGBRegressor(

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# PARAMETER GRID
# ============================================================

param_grid = {

    "n_estimators": [
        300,
        500
    ],

    "max_depth": [
        3,
        4,
        5
    ],

    "learning_rate": [
        0.03,
        0.05,
        0.1
    ],

    "subsample": [
        0.8,
        1.0
    ],

    "colsample_bytree": [
        0.8,
        1.0
    ]
}


# ============================================================
# GRID SEARCH
# ============================================================

print("\nStarting XGBoost tuning...")

grid_search = GridSearchCV(

    estimator=xgb_model,

    param_grid=param_grid,

    cv=3,

    scoring="r2",

    n_jobs=-1,

    verbose=1
)


grid_search.fit(
    X_train,
    y_train
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_


print("\nBest parameters:")

print(
    grid_search.best_params_
)


print(
    f"\nBest CV R²: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# CROSS VALIDATION
# ============================================================

cv_scores = cross_val_score(

    best_model,

    X_train,

    y_train,

    cv=5,

    scoring="r2",

    n_jobs=-1
)


print("\nCross-validation scores:")

print(cv_scores)

print(
    f"Mean CV R²: "
    f"{cv_scores.mean():.4f}"
)

print(
    f"CV Std: "
    f"{cv_scores.std():.4f}"
)


# ============================================================
# PREDICTIONS
# ============================================================

train_predictions = best_model.predict(
    X_train
)

test_predictions = best_model.predict(
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
# SAVE XGBOOST MODEL
# ============================================================

model_path = (
    "../models/house_price_model.json"
)


best_model.save_model(
    model_path
)


print(
    "\nModel saved successfully!"
)

print(
    f"Location: {model_path}"
)


# ============================================================
# SAVE FEATURE COLUMNS INSIDE METRICS
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
        cv_scores.mean(),

    "cross_validation_r2_std":
        cv_scores.std(),

    "best_parameters":
        grid_search.best_params_,

    "feature_columns":
        X.columns.tolist()
}


# ============================================================
# SAVE METRICS
# ============================================================

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
    "Metrics saved successfully!"
)

print("\nTraining completed.")
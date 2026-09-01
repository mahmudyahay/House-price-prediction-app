import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import (LinearRegression, Lasso, Ridge, ElasticNet)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import(OneHotEncoder, StandardScaler)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (r2_score, mean_absolute_error, root_mean_squared_error)
from preprocessing import feat_gengeenering, wrangle
import pickle
import json

# ======================================================
## load data
# ======================================================
df = pd.read_csv('../data/train.csv')

# ======================================================
# Clean and engineer data 
# ======================================================
df = feat_gengeenering(
    wrangle(df)
    )

# ======================================================
# Spliting Data
# ======================================================
X = df.drop(
    columns = [
        'SalePrice',
        'LogSalePrice',
        'Id'
    ]
)
y = df['LogSalePrice']

# ====================== Train/Test Split ================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =======================================================================
# preprocesing
# =======================================================================
numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_pipeline = Pipeline(
    [
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    [
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]

)

preprocessor = ColumnTransformer(
    [
        ('numeric', numeric_pipeline, numeric_features),
        ('categorical', categorical_pipeline, categorical_features)
    ]
)

# ============================================================
# models
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(alpha=1.0),

    "Lasso": Lasso(alpha=0.001),

    "ElasticNet": ElasticNet(
        alpha=0.001,
        l1_ratio=0.5
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror"
    )
}

# ============================================================
# train and evaluate models
# ============================================================

results = []

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 60)

    print(f"Training {name}...")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    # Predictions
    train_predictions = pipeline.predict(
        X_train
    )

    test_predictions = pipeline.predict(
        X_test
    )

    # Metrics
    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    train_mae = mean_absolute_error(
        y_train,
        train_predictions
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    train_rmse = root_mean_squared_error(
        y_train,
        train_predictions
    )

    test_rmse = root_mean_squared_error(
        y_test,
        test_predictions
    )

    results.append(
        {
            "Model": name,

            "Train_R2": train_r2,
            "Test_R2": test_r2,

            "Train_MAE": train_mae,
            "Test_MAE": test_mae,

            "Train_RMSE": train_rmse,
            "Test_RMSE": test_rmse
        }
    )

    trained_models[name] = pipeline

    print(f"Train R² : {train_r2:.4f}")
    print(f"Test R²  : {test_r2:.4f}")

    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE : {test_mae:.4f}")

    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE : {test_rmse:.4f}")



# ====================== saving evaluation results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by ='Test_R2',
    ascending=False
)
results_df.to_csv('../results/model_comparison.csv', index=False)


best_model_name = results_df.iloc[0]['Model']
best_pipeline = trained_models[best_model_name]

cv_scores = cross_val_score(
    best_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print(f' {cv_scores.mean():.4f} ')
print(f'{cv_scores.std():.4f}')



print("\n")
print("=" * 70)
print("XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)


xgb_pipeline = Pipeline(
    steps=[
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


grid_search = GridSearchCV(
    estimator=xgb_pipeline,
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


print("\nBest XGBoost parameters:")

print(
    grid_search.best_params_
)

print(
    f"Best CV R²: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# EVALUATE TUNED XGBOOST
# ============================================================

tuned_xgb = grid_search.best_estimator_


tuned_train_predictions = tuned_xgb.predict(
    X_train
)

tuned_test_predictions = tuned_xgb.predict(
    X_test
)


tuned_train_r2 = r2_score(
    y_train,
    tuned_train_predictions
)

tuned_test_r2 = r2_score(
    y_test,
    tuned_test_predictions
)

tuned_train_mae = mean_absolute_error(
    y_train,
    tuned_train_predictions
)

tuned_test_mae = mean_absolute_error(
    y_test,
    tuned_test_predictions
)

tuned_train_rmse = root_mean_squared_error(
    y_train,
    tuned_train_predictions
)

tuned_test_rmse = root_mean_squared_error(
    y_test,
    tuned_test_predictions
)


print("\nTUNED XGBOOST PERFORMANCE")

print(
    f"Training R²: {tuned_train_r2:.4f}"
)

print(
    f"Testing R²: {tuned_test_r2:.4f}"
)

print(
    f"Training MAE: {tuned_train_mae:.4f}"
)

print(
    f"Testing MAE: {tuned_test_mae:.4f}"
)

print(
    f"Training RMSE: {tuned_train_rmse:.4f}"
)

print(
    f"Testing RMSE: {tuned_test_rmse:.4f}"
)


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

model_path = "../models/house_price_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(tuned_xgb, file)


print("\nModel saved successfully!")

print(
    f"Location: {model_path}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model": "Tuned XGBoost",

    "training_r2": tuned_train_r2,

    "testing_r2": tuned_test_r2,

    "training_mae": tuned_train_mae,

    "testing_mae": tuned_test_mae,

    "training_rmse": tuned_train_rmse,

    "testing_rmse": tuned_test_rmse,

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
    "Metrics saved successfully!"
)

print("\nTraining pipeline completed.")
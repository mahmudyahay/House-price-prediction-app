
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from xgboost import XGBRegressor


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/house_price_model.json"
FEATURES_PATH = "models/feature_columns.json"
METRICS_PATH = "results/metrics.json"
DATA_PATH = "data/train.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = XGBRegressor()

    model.load_model(MODEL_PATH)

    return model


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

@st.cache_data
def load_feature_columns():

    with open(FEATURES_PATH, "r") as file:

        return json.load(file)


# ============================================================
# LOAD METRICS
# ============================================================

@st.cache_data
def load_metrics():

    if not os.path.exists(METRICS_PATH):

        return None

    with open(METRICS_PATH, "r") as file:

        return json.load(file)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    df = data.copy()

    current_year = 2026

    df["HouseAge"] = (
        current_year - df["YearBuilt"]
    )

    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["3SsnPorch"]
        + df["EnclosedPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )

    df["TotalSF"] = (
        df["GrLivArea"]
        + df["TotalBsmtSF"]
    )

    df["YearsSinceRemodel"] = (
        df["YearRemodAdd"]
        - df["YearBuilt"]
    )

    df["IsRemodeled"] = (
        df["YearRemodAdd"]
        != df["YearBuilt"]
    ).astype(int)

    df["HasGarage"] = (
        df["GarageArea"] > 0
    ).astype(int)

    df["HasFireplace"] = (
        df["Fireplaces"] > 0
    ).astype(int)

    df["HasPool"] = (
        df["PoolArea"] > 0
    ).astype(int)

    df["HasBasement"] = (
        df["TotalBsmtSF"] > 0
    ).astype(int)

    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
    )

    df["GarageSpacePerCar"] = np.where(
        df["GarageCars"] > 0,
        df["GarageArea"] / df["GarageCars"],
        0
    )

    return df


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(
    overall_qual,
    gr_liv_area,
    year_built,
    total_bsmt_sf,
    garage_cars,
    garage_area,
    full_bath,
    half_bath,
    neighborhood,
    kitchen_qual,
    exter_qual
):

    data = {

        "MSSubClass": 20,
        "MSZoning": "RL",
        "LotFrontage": 70,
        "LotArea": 8000,
        "Street": "Pave",
        "Alley": "None",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",

        "Neighborhood": neighborhood,

        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",

        "OverallQual": overall_qual,
        "OverallCond": 5,

        "YearBuilt": year_built,
        "YearRemodAdd": year_built,

        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",

        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",

        "MasVnrType": "None",
        "MasVnrArea": 0,

        "ExterQual": exter_qual,
        "ExterCond": "TA",

        "Foundation": "PConc",

        "BsmtQual": (
            "TA"
            if total_bsmt_sf > 0
            else "None"
        ),

        "BsmtCond": (
            "TA"
            if total_bsmt_sf > 0
            else "None"
        ),

        "BsmtExposure": (
            "No"
            if total_bsmt_sf > 0
            else "None"
        ),

        "BsmtFinType1": (
            "GLQ"
            if total_bsmt_sf > 0
            else "None"
        ),

        "BsmtFinSF1": total_bsmt_sf * 0.6,

        "BsmtFinType2": (
            "Unf"
            if total_bsmt_sf > 0
            else "None"
        ),

        "BsmtFinSF2": 0,

        "BsmtUnfSF": total_bsmt_sf * 0.4,

        "TotalBsmtSF": total_bsmt_sf,

        "BsmtFullBath": (
            1
            if total_bsmt_sf > 1000
            else 0
        ),

        "BsmtHalfBath": 0,

        "1stFlrSF": gr_liv_area * 0.6,

        "2ndFlrSF": gr_liv_area * 0.4,

        "LowQualFinSF": 0,

        "GrLivArea": gr_liv_area,

        "FullBath": full_bath,

        "HalfBath": half_bath,

        "BedroomAbvGr": 3,

        "KitchenAbvGr": 1,

        "KitchenQual": kitchen_qual,

        "TotRmsAbvGrd": 7,

        "Functional": "Typ",

        "Fireplaces": 1,

        "FireplaceQu": "Gd",

        "GarageType": (
            "Attchd"
            if garage_cars > 0
            else "None"
        ),

        "GarageYrBlt": (
            year_built
            if garage_cars > 0
            else 0
        ),

        "GarageFinish": (
            "Unf"
            if garage_cars > 0
            else "None"
        ),

        "GarageCars": garage_cars,

        "GarageArea": garage_area,

        "GarageQual": (
            "TA"
            if garage_cars > 0
            else "None"
        ),

        "GarageCond": (
            "TA"
            if garage_cars > 0
            else "None"
        ),

        "PavedDrive": "Y",

        "WoodDeckSF": 0,
        "OpenPorchSF": 50,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,

        "PoolArea": 0,
        "PoolQC": "None",

        "Fence": "None",
        "MiscFeature": "None",
        "MiscVal": 0,

        "MoSold": 6,
        "YrSold": 2026,

        "SaleType": "WD",
        "SaleCondition": "Normal",

        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr"
    }

    df = pd.DataFrame([data])

    df = create_features(df)

    return df


# ============================================================
# PREPARE DATA FOR XGBOOST
# ============================================================

def prepare_for_model(df, feature_columns):

    df = pd.get_dummies(
        df,
        dtype=int
    )

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model = load_model()

    feature_columns = load_feature_columns()

    metrics = load_metrics()

    df = load_data()

except Exception as e:

    st.error(
        f"Application loading failed: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏠 House Price App")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict Price",
        "Model Performance",
        "Explore Data",
        "About Project"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.title(
        "🏠 House Price Prediction System"
    )

    st.subheader(
        "Machine Learning Regression Project"
    )

    st.write(
        """
        Predict house prices using a machine learning
        model trained on the Ames Housing dataset.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Dataset Size",
            f"{df.shape[0]:,}"
        )

    with col2:

        st.metric(
            "Features",
            f"{len(feature_columns):,}"
        )

    with col3:

        if metrics:

            st.metric(
                "Test R²",
                f"{metrics['testing_r2']:.3f}"
            )

    st.divider()

    st.header("How the system works")

    st.markdown(
        """
        **1. Data Collection**

        Ames Housing dataset is loaded.

        **2. Data Cleaning**

        Missing values are handled.

        **3. Feature Engineering**

        Features such as TotalSF, HouseAge,
        TotalBathrooms and others are created.

        **4. Encoding**

        Categorical variables are converted into
        numerical variables.

        **5. XGBoost**

        The trained XGBoost model predicts the
        house price.

        **6. Deployment**

        The model is deployed using Streamlit.
        """
    )


# ============================================================
# PREDICT PRICE
# ============================================================

elif page == "Predict Price":

    st.title("🔮 Predict House Price")

    st.write(
        "Enter the characteristics of the house."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        overall_qual = st.slider(
            "Overall Quality",
            1,
            10,
            5
        )

        gr_liv_area = st.number_input(
            "Above Ground Living Area (sq ft)",
            100,
            10000,
            1500
        )

        year_built = st.number_input(
            "Year Built",
            1800,
            2026,
            2000
        )

        total_bsmt_sf = st.number_input(
            "Total Basement Area (sq ft)",
            0,
            5000,
            1000
        )

    with col2:

        garage_cars = st.number_input(
            "Garage Capacity",
            0,
            5,
            2
        )

        garage_area = st.number_input(
            "Garage Area (sq ft)",
            0,
            2000,
            400
        )

        full_bath = st.number_input(
            "Full Bathrooms",
            0,
            5,
            2
        )

        half_bath = st.number_input(
            "Half Bathrooms",
            0,
            3,
            1
        )

    st.divider()

    neighborhood = st.selectbox(
        "Neighborhood",
        sorted(
            df["Neighborhood"]
            .dropna()
            .unique()
        )
    )

    kitchen_qual = st.selectbox(
        "Kitchen Quality",
        sorted(
            df["KitchenQual"]
            .dropna()
            .unique()
        )
    )

    exter_qual = st.selectbox(
        "Exterior Quality",
        sorted(
            df["ExterQual"]
            .dropna()
            .unique()
        )
    )

    st.divider()

    if st.button(
        "🏠 Predict House Price",
        type="primary"
    ):

        input_df = prepare_input(
            overall_qual,
            gr_liv_area,
            year_built,
            total_bsmt_sf,
            garage_cars,
            garage_area,
            full_bath,
            half_bath,
            neighborhood,
            kitchen_qual,
            exter_qual
        )

        model_input = prepare_for_model(
            input_df,
            feature_columns
        )

        try:

            log_prediction = model.predict(
                model_input
            )[0]

            predicted_price = np.expm1(
                log_prediction
            )

            st.success(
                f"### Estimated House Price: "
                f"${predicted_price:,.2f}"
            )

            st.info(
                """
                This is a machine-learning estimate
                based on patterns learned from the
                training dataset.
                """
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    if metrics:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Testing R²",
                f"{metrics['testing_r2']:.4f}"
            )

        with col2:

            st.metric(
                "Testing MAE",
                f"{metrics['testing_mae']:.4f}"
            )

        with col3:

            st.metric(
                "Testing RMSE",
                f"{metrics['testing_rmse']:.4f}"
            )

        st.divider()

        st.subheader(
            "Training vs Testing"
        )

        performance_df = pd.DataFrame(
            {
                "Metric": [
                    "R²",
                    "MAE",
                    "RMSE"
                ],

                "Training": [
                    metrics["training_r2"],
                    metrics["training_mae"],
                    metrics["training_rmse"]
                ],

                "Testing": [
                    metrics["testing_r2"],
                    metrics["testing_mae"],
                    metrics["testing_rmse"]
                ]
            }
        )

        st.dataframe(
            performance_df,
            use_container_width=True
        )

        st.subheader(
            "Cross-Validation"
        )

        st.metric(
            "Mean CV R²",
            f"{metrics['cross_validation_r2_mean']:.4f}"
        )

        st.subheader(
            "Best XGBoost Parameters"
        )

        st.json(
            metrics["best_parameters"]
        )

    else:

        st.warning(
            "Metrics file not found."
        )


# ============================================================
# EXPLORE DATA
# ============================================================

elif page == "Explore Data":

    st.title("📈 Explore the Dataset")

    st.write(
        f"""
        The dataset contains
        **{df.shape[0]:,} rows**
        and
        **{df.shape[1]:,} columns**.
        """
    )

    st.subheader("First 10 Rows")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    st.dataframe(
        missing.to_frame(
            "Missing Values"
        ),
        use_container_width=True
    )

    st.subheader(
        "Sale Price Distribution"
    )

    st.bar_chart(
        df["SalePrice"].value_counts(
            bins=30
        ).sort_index()
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ### Project Goal

        Build an end-to-end machine learning system
        capable of predicting house prices.

        ### Machine Learning Models

        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - ElasticNet
        - Random Forest
        - XGBoost

        ### Data Processing

        - Missing value treatment
        - Feature engineering
        - One-hot encoding
        - Train/test split

        ### Model Evaluation

        - R² Score
        - Mean Absolute Error
        - Root Mean Squared Error
        - Cross-validation

        ### Deployment

        The final XGBoost model is deployed using
        Streamlit.

        ### Author

        **Mahmud Umar Yahaya**

        AI Engineer | Computer Vision Engineer |
        Data Scientist
        """
    )


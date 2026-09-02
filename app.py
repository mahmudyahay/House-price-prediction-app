import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import wrangle, feat_gengeenering





MODEL_PATH = "models/house_price_model.pkl"

METRICS_PATH = "results/metrics.json"

DATA_PATH = "data/train.csv"


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        return pickle.load(file)


# ============================================================
# LOAD METRICS
# ============================================================

@st.cache_data
def load_metrics():

    with open(
        METRICS_PATH,
        "r"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )


# ============================================================
# LOAD
# ============================================================

try:

    model = load_model()
    metrics = load_metrics()
    df = load_data()

except Exception as e:

    st.error(
        f"Application loading failed: {e}"
    )

    st.stop()


# ============================================================
# CREATE INPUT
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

    # Start with one real row from the dataset.
    input_df = df.iloc[[0]].copy()

    # --------------------------------------------------------
    # Replace user-selected values
    # --------------------------------------------------------

    input_df["OverallQual"] = overall_qual

    input_df["GrLivArea"] = gr_liv_area

    input_df["YearBuilt"] = year_built

    input_df["YearRemodAdd"] = year_built

    input_df["TotalBsmtSF"] = total_bsmt_sf

    input_df["GarageCars"] = garage_cars

    input_df["GarageArea"] = garage_area

    input_df["FullBath"] = full_bath

    input_df["HalfBath"] = half_bath

    input_df["Neighborhood"] = neighborhood

    input_df["KitchenQual"] = kitchen_qual

    input_df["ExterQual"] = exter_qual

    # --------------------------------------------------------
    # Adjust related features
    # --------------------------------------------------------

    input_df["GarageType"] = (
        "Attchd"
        if garage_cars > 0
        else np.nan
    )

    input_df["GarageYrBlt"] = (
        year_built
        if garage_cars > 0
        else np.nan
    )

    input_df["GarageFinish"] = (
        "Unf"
        if garage_cars > 0
        else np.nan
    )

    input_df["GarageQual"] = (
        "TA"
        if garage_cars > 0
        else np.nan
    )

    input_df["GarageCond"] = (
        "TA"
        if garage_cars > 0
        else np.nan
    )

    # Basement
    if total_bsmt_sf > 0:

        input_df["BsmtQual"] = "TA"
        input_df["BsmtCond"] = "TA"
        input_df["BsmtExposure"] = "No"
        input_df["BsmtFinType1"] = "GLQ"
        input_df["BsmtFinSF1"] = total_bsmt_sf * 0.6
        input_df["BsmtFinType2"] = "Unf"
        input_df["BsmtFinSF2"] = 0
        input_df["BsmtUnfSF"] = total_bsmt_sf * 0.4
        input_df["BsmtFullBath"] = 1
        input_df["BsmtHalfBath"] = 0

    else:

        input_df["BsmtQual"] = np.nan
        input_df["BsmtCond"] = np.nan
        input_df["BsmtExposure"] = np.nan
        input_df["BsmtFinType1"] = np.nan
        input_df["BsmtFinSF1"] = 0
        input_df["BsmtFinType2"] = np.nan
        input_df["BsmtFinSF2"] = 0
        input_df["BsmtUnfSF"] = 0
        input_df["BsmtFullBath"] = 0
        input_df["BsmtHalfBath"] = 0

    # --------------------------------------------------------
    # Process exactly like training
    # --------------------------------------------------------

    input_df = wrangle(
        input_df
    )

    input_df = feat_gengeenering(
        input_df
    )

    # --------------------------------------------------------
    # Remove target and ID
    # --------------------------------------------------------

    input_df = input_df.drop(
        columns=[
            "SalePrice",
            "LogSalePrice",
            "Id"
        ],
        errors="ignore"
    )

    return input_df


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🏠 House Price App"
)

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
            "Houses",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Features",
            f"{df.shape[1]:,}"
        )

    with col3:

        st.metric(
            "Test R²",
            f"{metrics['testing_r2']:.3f}"
        )

    st.divider()

    st.header(
        "How it works"
    )

    st.markdown(
        """
        **1. Data Cleaning**

        Missing values are handled.

        **2. Feature Engineering**

        New housing features are created.

        **3. Preprocessing**

        Numerical features are scaled and categorical
        features are encoded.

        **4. Model Training**

        Several regression models are compared.

        **5. XGBoost Tuning**

        XGBoost is tuned using cross-validation.

        **6. Prediction**

        The trained model predicts the estimated house price.
        """
    )


# ============================================================
# PREDICT
# ============================================================

elif page == "Predict Price":

    st.title(
        "🔮 Predict House Price"
    )

    st.write(
        "Enter the characteristics of the house."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        overall_qual = st.slider(
            "⭐ Overall Quality",
            1,
            10,
            5
        )

        gr_liv_area = st.number_input(
            "🏠 Living Area (sq ft)",
            100,
            10000,
            1500
        )

        year_built = st.number_input(
            "📅 Year Built",
            1900,
            2026,
            2000
        )

        total_bsmt_sf = st.number_input(
            "Basement Area (sq ft)",
            0,
            5000,
            1000
        )

        full_bath = st.number_input(
            "Full Bathrooms",
            0,
            5,
            2
        )

    with col2:

        garage_cars = st.number_input(
            "🚗 Garage Capacity",
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

        half_bath = st.number_input(
            "Half Bathrooms",
            0,
            3,
            1
        )

        neighborhood = st.selectbox(
            "📍 Neighborhood",
            sorted(
                df["Neighborhood"]
                .dropna()
                .unique()
            )
        )

        kitchen_qual = st.selectbox(
            "🍳 Kitchen Quality",
            sorted(
                df["KitchenQual"]
                .dropna()
                .unique()
            )
        )

    exter_qual = st.selectbox(
        "🏡 Exterior Quality",
        sorted(
            df["ExterQual"]
            .dropna()
            .unique()
        )
    )

    st.divider()

    if st.button(
        "🏠 Predict House Price",
        type="primary",
        use_container_width=True
    ):

        try:

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

            log_prediction = model.predict(
                input_df
            )[0]

            predicted_price = np.expm1(
                log_prediction
            )

            st.success(
                f"## Estimated Price: ${predicted_price:,.2f}"
            )

            st.info(
                "This is a machine-learning estimate and "
                "not a professional property valuation."
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "📊 Model Performance"
    )

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


# ============================================================
# EXPLORE DATA
# ============================================================

elif page == "Explore Data":

    st.title(
        "📈 Explore Dataset"
    )

    st.write(
        f"""
        Dataset contains **{len(df):,} rows**
        and **{df.shape[1]:,} columns**.
        """
    )

    st.subheader(
        "First 10 rows"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader(
        "Statistics"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader(
        "Missing Values"
    )

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


# ============================================================
# ABOUT
# ============================================================

elif page == "About Project":

    st.title(
        "ℹ️ About This Project"
    )

    st.markdown(
        """
        ### Project Goal

        Build an end-to-end machine learning system
        for house price prediction.

        ### Models

        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - ElasticNet
        - Random Forest
        - XGBoost

        ### Techniques

        - Missing value treatment
        - Feature engineering
        - Scaling
        - One-hot encoding
        - Cross-validation
        - Hyperparameter tuning

        ### Deployment

        The final XGBoost model is deployed using Streamlit.

        ### Author

        **Mahmud Umar Yahaya**

        AI Engineer | Computer Vision Engineer |
        Data Scientist
        """
    )
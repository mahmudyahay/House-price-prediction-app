import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

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
        "models/house_price_model.pkl",
        "rb"
    ) as file:

        return pickle.load(file)


# ============================================================
# LOAD METRICS
# ============================================================

@st.cache_data
def load_metrics():

    with open(
        "results/metrics.json",
        "r"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/train.csv"
    )


# ============================================================
# LOAD EVERYTHING
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

def create_input(
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

    current_year = 2026

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

        # Basement
        "BsmtQual": "TA" if total_bsmt_sf > 0 else "None",
        "BsmtCond": "TA" if total_bsmt_sf > 0 else "None",
        "BsmtExposure": "No" if total_bsmt_sf > 0 else "None",
        "BsmtFinType1": "GLQ" if total_bsmt_sf > 0 else "None",

        "BsmtFinSF1": total_bsmt_sf * 0.6,
        "BsmtFinType2": "Unf" if total_bsmt_sf > 0 else "None",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": total_bsmt_sf * 0.4,
        "TotalBsmtSF": total_bsmt_sf,

        "BsmtFullBath":
            1 if total_bsmt_sf > 0 else 0,

        "BsmtHalfBath": 0,

        # Living area
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

        # Garage
        "GarageType":
            "Attchd" if garage_cars > 0 else "None",

        "GarageYrBlt":
            year_built if garage_cars > 0 else 0,

        "GarageFinish":
            "Unf" if garage_cars > 0 else "None",

        "GarageCars": garage_cars,
        "GarageArea": garage_area,

        "GarageQual":
            "TA" if garage_cars > 0 else "None",

        "GarageCond":
            "TA" if garage_cars > 0 else "None",

        # Outdoor
        "PavedDrive": "Y",
        "WoodDeckSF": 0,
        "OpenPorchSF": 50,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,

        # Pool / Misc
        "PoolArea": 0,
        "PoolQC": "None",
        "Fence": "None",
        "MiscFeature": "None",
        "MiscVal": 0,

        # Sale information
        "MoSold": 6,
        "YrSold": 2026,
        "SaleType": "WD",
        "SaleCondition": "Normal",

        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr"
    }

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    data["HouseAge"] = (
        current_year - year_built
    )

    data["TotalPorchSF"] = (
        data["OpenPorchSF"]
        + data["3SsnPorch"]
        + data["EnclosedPorch"]
        + data["ScreenPorch"]
        + data["WoodDeckSF"]
    )

    data["TotalSF"] = (
        data["GrLivArea"]
        + data["TotalBsmtSF"]
    )

    data["YearsSinceRemodel"] = (
        data["YearRemodAdd"]
        - data["YearBuilt"]
    )

    data["IsRemodeled"] = (
        data["YearRemodAdd"]
        != data["YearBuilt"]
    ).astype(int)

    data["HasGarage"] = int(
        data["GarageArea"] > 0
    )

    data["HasFireplace"] = int(
        data["Fireplaces"] > 0
    )

    data["HasPool"] = int(
        data["PoolArea"] > 0
    )

    data["HasBasement"] = int(
        data["TotalBsmtSF"] > 0
    )

    data["TotalBathrooms"] = (
        data["FullBath"]
        + 0.5 * data["HalfBath"]
    )

    data["GarageSpacePerCar"] = (
        data["GarageArea"] / data["GarageCars"]
        if data["GarageCars"] > 0
        else 0
    )

    return pd.DataFrame([data])


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏠 House Price")

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

    st.title("🏠 House Price Prediction")

    st.subheader(
        "Machine Learning Regression System"
    )

    st.write(
        """
        Predict house prices using a trained XGBoost
        regression model based on the Ames Housing dataset.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Dataset Rows",
            f"{df.shape[0]:,}"
        )

    with col2:

        st.metric(
            "Dataset Columns",
            f"{df.shape[1]:,}"
        )

    with col3:

        st.metric(
            "Test R²",
            f"{metrics['testing_r2']:.3f}"
        )

    st.divider()

    st.header("How it works")

    st.markdown(
        """
        **1. Data Cleaning**  
        Missing values are handled.

        **2. Feature Engineering**  
        New useful features are created.

        **3. Preprocessing**  
        Numerical features are scaled and categorical
        features are encoded.

        **4. Model Training**  
        XGBoost learns patterns from the housing data.

        **5. Hyperparameter Tuning**  
        GridSearchCV finds better XGBoost parameters.

        **6. Prediction**  
        The trained model estimates the house price.
        """
    )


# ============================================================
# PREDICT
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
            1800,
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

        input_df = create_input(
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

        try:

            log_prediction = model.predict(
                input_df
            )[0]

            predicted_price = np.expm1(
                log_prediction
            )

            st.success(
                f"### Estimated Price: ${predicted_price:,.2f}"
            )

            st.info(
                "This is an ML estimate based on patterns "
                "learned from the training dataset."
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

    st.subheader(
        "Training vs Testing"
    )

    st.dataframe(
        performance_df,
        use_container_width=True
    )

    st.subheader(
        "Cross-Validation R²"
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

    st.title("📈 Explore Dataset")

    st.write(
        f"""
        The dataset contains **{df.shape[0]:,} rows**
        and **{df.shape[1]:,} columns**.
        """
    )

    st.subheader("First 10 Rows")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Statistics")

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

    if len(missing) > 0:

        st.dataframe(
            missing.to_frame(
                "Missing Values"
            ),
            use_container_width=True
        )

    else:

        st.success(
            "No missing values."
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
# ABOUT
# ============================================================

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ### Project Goal

        Build an end-to-end machine learning system
        for house price prediction.

        ### Models Studied

        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - ElasticNet
        - Random Forest
        - XGBoost

        ### Final Model

        **Tuned XGBoost Regression**

        ### Evaluation

        - R² Score
        - Mean Absolute Error
        - Root Mean Squared Error
        - Cross-validation

        ### Deployment

        The application is deployed using Streamlit.

        ### Author

        **Mahmud Umar Yahaya**

        AI Engineer | Computer Vision Engineer |
        Data Scientist
        """
    )

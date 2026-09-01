
import os
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "models/house_price_model.pkl"


@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


# ============================================================
# LOAD METRICS
# ============================================================

@st.cache_data
def load_metrics():

    metrics_path = "results/metrics.json"

    if os.path.exists(metrics_path):

        with open(metrics_path, "r") as file:
            return json.load(file)

    return None


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv("data/train.csv")


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


metrics = load_metrics()
df = load_data()


# ============================================================
# PREPARE PREDICTION INPUT
# ============================================================

def prepare_prediction_input(
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

    # --------------------------------------------------------
    # Use the same reference year used during training
    # --------------------------------------------------------

    current_year = df["YearBuilt"].max()


    # --------------------------------------------------------
    # BASIC FEATURES
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # BASEMENT
        # ----------------------------------------------------

        "BsmtQual": (
            "TA" if total_bsmt_sf > 0 else "None"
        ),

        "BsmtCond": (
            "TA" if total_bsmt_sf > 0 else "None"
        ),

        "BsmtExposure": (
            "No" if total_bsmt_sf > 0 else "None"
        ),

        "BsmtFinType1": (
            "GLQ" if total_bsmt_sf > 0 else "None"
        ),

        "BsmtFinSF1": total_bsmt_sf * 0.6,

        "BsmtFinType2": (
            "Unf" if total_bsmt_sf > 0 else "None"
        ),

        "BsmtFinSF2": 0,

        "BsmtUnfSF": total_bsmt_sf * 0.4,

        "TotalBsmtSF": total_bsmt_sf,

        "BsmtFullBath": (
            1 if total_bsmt_sf > 0 else 0
        ),

        "BsmtHalfBath": 0,


        # ----------------------------------------------------
        # LIVING AREA
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # GARAGE
        # ----------------------------------------------------

        "GarageType": (
            "Attchd" if garage_cars > 0 else "None"
        ),

        "GarageYrBlt": (
            year_built if garage_cars > 0 else 0
        ),

        "GarageFinish": (
            "Unf" if garage_cars > 0 else "None"
        ),

        "GarageCars": garage_cars,

        "GarageArea": garage_area,

        "GarageQual": (
            "TA" if garage_cars > 0 else "None"
        ),

        "GarageCond": (
            "TA" if garage_cars > 0 else "None"
        ),


        # ----------------------------------------------------
        # OUTDOOR FEATURES
        # ----------------------------------------------------

        "PavedDrive": "Y",

        "WoodDeckSF": 0,

        "OpenPorchSF": 50,

        "EnclosedPorch": 0,

        "3SsnPorch": 0,

        "ScreenPorch": 0,


        # ----------------------------------------------------
        # POOL / MISC
        # ----------------------------------------------------

        "PoolArea": 0,

        "PoolQC": "None",

        "Fence": "None",

        "MiscFeature": "None",

        "MiscVal": 0,


        # ----------------------------------------------------
        # SALE INFORMATION
        # ----------------------------------------------------

        "MoSold": 6,

        "YrSold": 2010,

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
        current_year - data["YearBuilt"]
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


    data["IsRemodeled"] = int(

        data["YearRemodAdd"]
        != data["YearBuilt"]

    )


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

    st.title("🏠 House Price Prediction System")

    st.subheader(
        "Machine Learning Regression Project"
    )

    st.write(
        """
        This application predicts house sale prices
        using machine learning techniques trained on
        the Ames Housing dataset.
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
            f"{df.shape[1]:,}"
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

        New features such as TotalSF,
        HouseAge and TotalBathrooms are created.

        **4. Preprocessing**

        Numerical features are scaled and
        categorical features are one-hot encoded.

        **5. Machine Learning**

        Several regression models are trained
        and compared.

        **6. Model Selection**

        XGBoost is tuned using cross-validation.

        **7. Deployment**

        The final model is saved and served
        through Streamlit.
        """
    )


# ============================================================
# PREDICT PRICE
# ============================================================

elif page == "Predict Price":

    st.title("🔮 Predict House Price")

    st.write(
        "Enter the characteristics of the house below."
    )

    st.divider()


    col1, col2 = st.columns(2)


    with col1:

        overall_qual = st.slider(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=5
        )


        gr_liv_area = st.number_input(
            "Above Ground Living Area (sq ft)",
            min_value=100,
            max_value=10000,
            value=1500
        )


        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=int(df["YearBuilt"].max()),
            value=2000
        )


        total_bsmt_sf = st.number_input(
            "Total Basement Area (sq ft)",
            min_value=0,
            max_value=5000,
            value=1000
        )


    with col2:

        garage_cars = st.number_input(
            "Garage Capacity",
            min_value=0,
            max_value=5,
            value=2
        )


        garage_area = st.number_input(
            "Garage Area (sq ft)",
            min_value=0,
            max_value=2000,
            value=400
        )


        full_bath = st.number_input(
            "Full Bathrooms",
            min_value=0,
            max_value=5,
            value=2
        )


        half_bath = st.number_input(
            "Half Bathrooms",
            min_value=0,
            max_value=3,
            value=1
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


    predict_button = st.button(
        "🏠 Predict House Price",
        type="primary"
    )


    if predict_button:

        input_df = prepare_prediction_input(

            overall_qual=overall_qual,

            gr_liv_area=gr_liv_area,

            year_built=year_built,

            total_bsmt_sf=total_bsmt_sf,

            garage_cars=garage_cars,

            garage_area=garage_area,

            full_bath=full_bath,

            half_bath=half_bath,

            neighborhood=neighborhood,

            kitchen_qual=kitchen_qual,

            exter_qual=exter_qual
        )


        try:

            # Model predicts LogSalePrice

            log_prediction = model.predict(
                input_df
            )[0]


            # Convert back to original price

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


        st.subheader("Training vs Testing")


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


        st.subheader("Cross-Validation")


        st.write(
            f"""
            Mean cross-validation R²:

            **{metrics['cross_validation_r2_mean']:.4f}**
            """
        )


        st.subheader("Best XGBoost Parameters")


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


    st.subheader("First 10 rows")


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
        .sort_values(ascending=False)

    )


    missing = missing[missing > 0]


    st.dataframe(
        missing.to_frame("Missing Values"),
        use_container_width=True
    )


    st.subheader("Sale Price Distribution")


    st.bar_chart(
        df["SalePrice"]
        .value_counts(bins=30)
        .sort_index()
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

        ### Machine Learning Techniques

        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - ElasticNet
        - Random Forest
        - XGBoost

        ### Data Processing

        - Missing value treatment
        - Feature engineering
        - Numerical scaling
        - One-hot encoding
        - Train/test split

        ### Model Evaluation

        - R² Score
        - Mean Absolute Error
        - Root Mean Squared Error
        - Cross-validation

        ### Deployment

        The final model is deployed using Streamlit.

        ### Author

        **Mahmud Umar Yahaya**

        AI Engineer | Computer Vision Engineer |
        Data Scientist
        """
    )

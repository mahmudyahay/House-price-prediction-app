import numpy as np
import pandas as pd


# ============================================================
# CLEAN DATA
# ============================================================

def wrangle(df):

    df = df.copy()

    # Categorical missing values
    none_cols = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "MasVnrType",
        "FireplaceQu",
        "GarageType",
        "GarageCond",
        "GarageQual",
        "GarageFinish",
        "BsmtFinType2",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtQual",
        "BsmtCond"
    ]

    for col in none_cols:

        if col in df.columns:

            df[col] = df[col].fillna("None")


    # Numerical missing values
    zero_cols = [
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath"
    ]

    for col in zero_cols:

        if col in df.columns:

            df[col] = df[col].fillna(0)


    # Electrical
    if "Electrical" in df.columns:

        df["Electrical"] = df["Electrical"].fillna(
            df["Electrical"].mode()[0]
        )


    # LotFrontage
    if "LotFrontage" in df.columns:

        df["LotFrontage"] = (
            df.groupby("Neighborhood")["LotFrontage"]
            .transform(
                lambda x: x.fillna(x.median())
            )
        )

        df["LotFrontage"] = df["LotFrontage"].fillna(
            df["LotFrontage"].median()
        )


    # MasVnrArea
    if "MasVnrArea" in df.columns:

        df["MasVnrArea"] = np.where(
            df["MasVnrType"] == "None",
            0,
            df["MasVnrArea"]
        )

        df["MasVnrArea"] = df["MasVnrArea"].fillna(
            df["MasVnrArea"].median()
        )


    # Other categorical columns
    mode_cols = [
        "MSZoning",
        "Utilities",
        "Functional",
        "SaleType",
        "KitchenQual",
        "Exterior1st",
        "Exterior2nd"
    ]

    for col in mode_cols:

        if col in df.columns:

            if not df[col].mode().empty:

                df[col] = df[col].fillna(
                    df[col].mode()[0]
                )


    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feat_gengeenering(df):

    df = df.copy()

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


    # Target
    if "SalePrice" in df.columns:

        df["LogSalePrice"] = np.log1p(
            df["SalePrice"]
        )


    return df


# ============================================================
# FINAL PREPROCESSING FOR XGBOOST
# ============================================================

def prepare_features(df):

    df = df.copy()

    # Remove target columns
    df = df.drop(
        columns=[
            "SalePrice",
            "LogSalePrice",
            "Id"
        ],
        errors="ignore"
    )

    # Convert categorical columns to numerical
    df = pd.get_dummies(
        df,
        drop_first=False,
        dtype=float
    )

    # Fill anything remaining
    df = df.fillna(0)

    return df

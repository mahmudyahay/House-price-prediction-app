import numpy as np
import pandas as pd


def wrangle(df):
    """
    Clean the Ames Housing dataset.
    """

    df = df.copy()

    # Categorical columns where missing means the feature does not exist
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
        "BsmtCond",
    ]

    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Electrical
    if "Electrical" in df.columns:
        df["Electrical"] = df["Electrical"].fillna(
            df["Electrical"].mode()[0]
        )

    # LotFrontage
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = (
            df.groupby("Neighborhood")["LotFrontage"]
            .transform(lambda x: x.fillna(x.median()))
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

    return df


def feat_gengeenering(df):
    """
    Create additional features.
    """

    df = df.copy()

    # Use a fixed reference year so training and prediction
    # use the same logic.
    current_year = 2010

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
        df["YearRemodAdd"] != df["YearBuilt"]
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

    # Target transformation only when SalePrice exists
    if "SalePrice" in df.columns:

        df["LogSalePrice"] = np.log1p(
            df["SalePrice"]
        )

    return df
import numpy as np
import pandas as pd


def wrangle(df):
    """
    Clean the Ames Housing dataset.
    """

    df = df.copy()

    # Categorical columns where NaN means feature does not exist
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

    # Masonry veneer
    if "MasVnrArea" in df.columns:

        if "MasVnrType" in df.columns:
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

    # House age
    current_year = df["YearBuilt"].max()

    df["HouseAge"] = (
        current_year - df["YearBuilt"]
    )

    # Porch area
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["3SsnPorch"]
        + df["EnclosedPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )

    # Total area
    df["TotalSF"] = (
        df["GrLivArea"]
        + df["TotalBsmtSF"]
    )

    # Remodeling
    df["YearsSinceRemodel"] = (
        df["YearRemodAdd"]
        - df["YearBuilt"]
    )

    df["IsRemodeled"] = (
        df["YearRemodAdd"]
        != df["YearBuilt"]
    ).astype(int)

    # Garage
    df["HasGarage"] = (
        df["GarageArea"] > 0
    ).astype(int)

    # Fireplace
    df["HasFireplace"] = (
        df["Fireplaces"] > 0
    ).astype(int)

    # Pool
    df["HasPool"] = (
        df["PoolArea"] > 0
    ).astype(int)

    # Basement
    df["HasBasement"] = (
        df["TotalBsmtSF"] > 0
    ).astype(int)

    # Bathrooms
    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
    )

    # Garage space
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

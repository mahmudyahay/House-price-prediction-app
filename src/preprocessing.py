import numpy as np
import pandas as pd


def wrangle(df):
    """
    Clean the Ames Housing dataset and create additional features.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw Ames Housing dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned and feature-engineered dataset.
    """
    print('=' * 70)
    print('Start Cleaning')
    print('=' * 70)

    df = df.copy()


    # ====================================================
    # 1. Handling Missing Values for Categorical Features
    # ====================================================
    none_cols = ['PoolQC',
        'MiscFeature',
        'Alley',
        'Fence',
        'MasVnrType',
        'FireplaceQu',
        'GarageType',
        'GarageCond',
        'GarageQual',
        'GarageFinish',
        'BsmtFinType2',
        'BsmtExposure',
        'BsmtFinType1',
        'BsmtQual',
        'BsmtCond',
        ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')


    # ================================================
    # Filling 'Electrical' with Mode
    # due to only one missing value
    # ================================================
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    

    # ====================================================
    # Filling LotFrontage with Neighborhood group median
    # ====================================================
    df['LotFrontage'] = (
    df.groupby('Neighborhood')['LotFrontage']
    .transform(lambda x: x.fillna(x.median()))
    )

    # ===================================================================
    # Cleaning MasVnrArea but with style, 
    # any house that dont have MasVnrType should have 0 as MasVnrArea 
    # then fill real missing ones with median
    # ===================================================================

    df['MasVnrArea'] = np.where(df['MasVnrType']== 'None', 0, df['MasVnrArea'])

    df['MasVnrArea'] = df["MasVnrArea"].fillna(df['MasVnrArea'].median())

    print('=' * 70)
    print('Finished Cleaning')
    print('=' * 70)

    return df

def feat_gengeenering(clean_df):
    """
    A Function that takes in a clean dataframe and do some feature engineering 
    
    Parameters:
        dataframe (pd.DataFrame): A clean dataframe from our wrangle function.

    Returns:
        dataframe (pd.DataFrame): An engineered DataFrame

    Examples:
    >>> feat_gengeenering(clean_df)
    df

    """
    print('=' * 70)
    print('Start Engineering')
    print('=' * 70)

    df = clean_df.copy()

    # ==================================================================
    # Engineer HouseAge by Subtracting "YearBuilt" from max "YearBuilt" 
    # ===================================================================

    current_year = df["YearBuilt"].max()
    df['HouseAge'] = (
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
            df["GrLivArea"] + df["TotalBsmtSF"]
        )

    df["YearsSinceRemodel"] = (
            df["YearRemodAdd"] - df["YearBuilt"]
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

    # =================================================================
    # Log-transform SalePrice to be our target
    # ==================================================================

    if "SalePrice" in df.columns:

        df["LogSalePrice"] = np.log1p(
            df["SalePrice"]
        )
    print('=' * 70)
    print('Finished Engineering')
    print('=' * 70)
    return df


# ====================================================
# testing preprocessing 
# ====================================================

df = pd.read_csv('../data/train.csv')
df = feat_gengeenering(wrangle(df))



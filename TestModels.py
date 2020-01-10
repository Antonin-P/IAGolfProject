import pandas as pd

pd.options.display.max_columns = 100
pd.options.display.max_rows = 1000

from visualization import PlotUtils
import seaborn as sns

import numpy as np
import pandas_profiling as pp
import matplotlib
import matplotlib.pyplot as plt

DATASET = 'Data/DataGolfForecast.csv'


"""LOADING DATASET"""
df = pd.read_csv(DATASET, sep=";", decimal=",")
df = df[df["nbCov"].notnull()]
df.index = pd.to_datetime(df['Date'], format="%d/%m/%Y")
del df['Date']
print(df.head(10))

target_col = ["nbCov"]
numerical_feature_cols = ["tempMoy", "tempMax", "tempMin", "wind", "nbPlay"]
categorical_feature_cols = ["fer", "vac", "tourn", "rain", "open"]

plotUtils = PlotUtils(df)

"""PREPARATION DATA"""

def add_date_part(iDf, iNewCols):
    """
    Utils to add day of week, day of year, ...
    """
    dates = iDf.index

    if not np.issubdtype(dates.dtype, np.datetime64):
        iDf.index = dates = pd.to_datetime(dates)

    for col in iNewCols:
        iDf[col] = getattr(dates, col).astype('category')


def shift_cols(iDfOrigin, iDictLags, iDropNa=True, iKeepAllLags=True):
    """
    Utils to add the lagged columns to the original Dataframe
    iDictLags must be of type
        {'columnName':[numberOfLag, BooleanKeepAllIntermediateLags], ...}
        example : dict_lags = {'nbCov':[7, True]}
    """
    for col in iDictLags.keys():
        lDfLagTmp = plotUtils.create_lag_col(col,
                                             iDictLags[col][0],
                                             iDropNa=False,
                                             iKeepAllLags=iDictLags[col][0])
        del lDfLagTmp[col]
        iDfOrigin = pd.merge(iDfOrigin
                             , lDfLagTmp
                             , right_index=True
                             , left_index=True
                             , how="left"
                             )
    if iDropNa:
        iDfOrigin.dropna(inplace=True)

    return iDfOrigin


# Add lagged variables and lagged result
# True means you keep all intermediate lags
dict_lags = {'nbCov':[7, True]
             , 'vac':[2, True]
             , 'tempMoy':[4, True]
             , 'tempMax':[4, True]
             , 'tempMin':[4, True]
             , 'wind':[5, True]
             , 'rain':[5, True]
             , 'nbPlay':[3, True]
            }

lagged_df = shift_cols(df, dict_lags, True)

# Delete rows when the restaurant is closed

lagged_df = lagged_df[lagged_df['open']]
lagged_df.drop(columns=['open'], inplace=True)

# Add time-based variables
dict_dt_cols = {'week':"week_"
                , 'day':"day_"
                , 'dayofweek':"dow_"
                , 'dayofyear':"doy_"
               }

add_date_part(lagged_df, dict_dt_cols.keys())
lagged_df.head()
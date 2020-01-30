import pandas as pd
pd.options.display.max_columns = 100
pd.options.display.max_rows = 1000

import random as rd
import numpy as np
import warnings
import math
import time
warnings.filterwarnings('ignore')

import pandas_profiling as pp

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, SGDRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, max_error
import statsmodels.api as sm
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
#%matplotlib inline


#from pyecharts.charts.line import Line
from pyecharts.charts import Line

from tqdm import tqdm_notebook
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from visualization import PlotUtils

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
dict_lags = {'nbCov': [7, True]
    , 'vac': [2, True]
    , 'tempMoy': [4, True]
    , 'tempMax': [4, True]
    , 'tempMin': [4, True]
    , 'wind': [5, True]
    , 'rain': [5, True]
    , 'nbPlay': [3, True]
             }

lagged_df = shift_cols(df, dict_lags, True)

# Delete rows when the restaurant is closed

lagged_df = lagged_df[lagged_df['open']]
lagged_df.drop(columns=['open'], inplace=True)

# Add time-based variables
dict_dt_cols = {'week': "week_"
    , 'day': "day_"
    , 'dayofweek': "dow_"
    , 'dayofyear': "doy_"
                }

add_date_part(lagged_df, dict_dt_cols.keys())
"""lagged_df.head()"""


# Dummification : important for LSTM

def create_df_dummy(iDf, iColList):
    for col in iColList:
        df_tmp = pd.get_dummies(data=pd.Series(data=list(iDf[col])))
        df_tmp.columns = [dict_dt_cols[col] + str(dt_int) for dt_int in df_tmp]
        df_tmp.index = iDf.index
        iDf = iDf.merge(df_tmp, left_index=True, right_index=True, how="left")
        del iDf[col]
    return iDf


lagged_df_full = create_df_dummy(lagged_df, ['dayofweek'])
print("chill")
print(lagged_df_full.head())
print("up")
print(lagged_df_full.tail())
lagged_df_full.dropna(inplace=True)

"""SPLIT"""

# X = lagged_df_full[[col for col in lagged_df_full.columns if col!=target_col[0]]]
X = lagged_df_full[['tempMax', 'tourn', 'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6']]
# date_index = X.index
# X.reset_index(inplace=True)
y = lagged_df_full[target_col]
# Specific to LSTM
X = y.merge(X, how='left', right_index=True, left_index=True)


"""Specifique a un random Forest"""

"""numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

rf = Pipeline(steps=[('preprocessor', preprocessor),
                     ('classifier', RandomForestClassifier())])"""

list_features = [col for col in X.columns if "dow" not in col]
list_fixed_features = [col for col in X.columns if "dow" in col]
nb_features = len(list_features)
nb_fixed_features = len(list_fixed_features)

def test_df_is_safe_not_nan(df_):
    col_with_nan = []
    for col in df_.columns:
        if df_[col].isnull().any():
            col_with_nan.append(col)
    if len(col_with_nan) > 0:
        print("Your data contains NaN in columns: {}".format(col_with_nan))


test_df_is_safe_not_nan(lagged_df_full)

"""MACHINE LEARNING"""

"""METRICS"""


# Not present in Scikit metrics
def calculate_residuals(y_true, y_pred):
    return y_true - y_pred


def calculate_mean_residuals(y_true, y_pred):
    return np.mean(y_true - y_pred)


RATING_METRICS_INIT = {
    "MAXE": np.nan,
    "MSE": np.nan,
    "MeanAE": np.nan,
    "MedianAE": np.nan,
    "R2": np.nan,
    # "Residuals": [],
    # "MeanResiduals": np.nan
}


def evaluate_all_metrics(y_true, y_pred):
    return {
        "MAXE": max_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "MeanAE": mean_absolute_error(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        # "Residuals": calculate_residuals(y_true, y_pred),
        # "MeanResiduals": calculate_mean_residuals(y_true, y_pred)
    }


"""DUMB ALGO"""


class AverageModel:
    """
    Baseline model : get the average value
    """

    def __init__(self):
        self = self
        self.mean = 0

    def fit(self, X_train, y_train):
        self.mean = np.mean(y_train)
        return self.mean

    def predict(self, X_test):
        return [self.mean for x in range(len(X_test))]


class PreviousValueLagged:
    """
    Baseline model : get the previous coherent value
    """

    def __init__(self, lag):
        self.lag = lag
        self.lagged_values = [0]

    def fit(self, X_train, y_train):
        if isinstance(y_train, pd.DataFrame):
            self.lagged_values = y_train[target_col[0]].values.tolist()
        elif isinstance(y_train, list):
            self.lagged_values = y_train
        return self.lagged_values

    def predict(self, X_test, y_test):
        if isinstance(y_test, pd.DataFrame):
            l_test = y_test[target_col[0]].values.tolist()
        elif isinstance(y_test, list):
            l_test = y_test
        l_list = self.lagged_values + l_test
        return l_list[-self.lag - len(y_test):][:len(y_test)]


# Models to test

# Lag Model
lag_model = 7  # days
# Parameters LSTM
nb_features, nb_fixed_features, nb_step_input, nb_step_output = nb_features, nb_fixed_features, 14, 3

models = {
    # Basic models
    # "AverageModel" : AverageModel(),
    "LagModel": PreviousValueLagged(lag_model)
    ##Linear models
    # 'LinearRegression': LinearRegression(),
    # 'Ridge': Ridge(),
    # 'Huber': HuberRegressor(),
    # 'SGD': SGDRegressor(max_iter=1000, tol=1e-3),
    ##Not linear models
    # 'KNeighbors': KNeighborsRegressor(n_neighbors=7),
    # 'SVR': SVR(),
    # 'Bagging': BaggingRegressor(n_estimators=100),
    # 'RandomForest': RandomForestRegressor(n_estimators=100,n_jobs=-1),
    # 'ExtraTrees': ExtraTreesRegressor(n_estimators=100, n_jobs=-1),
    # 'GradientBoosting': GradientBoostingRegressor(n_estimators=100),
    # NeuralNet
    # 'LSTM': LstmModel(nb_features, nb_fixed_features, nb_step_input, nb_step_output
    #                  , prefix_fixed_features ="dow", nb_epochs=20, verbose=1)
}

models_predictions = {}
models_residuals = {}

"""TRAIN/TEST SPLIT"""
NB_SPLITS = 3
tscv = TimeSeriesSplit(n_splits=NB_SPLITS)



"""BENCHMARK"""


def generate_summary(algo, split, train_time, time_rating, rating_metrics):
    summary = {"Algo": algo, "Split": split, "Train time (s)": train_time, "Predicting time (s)": time_rating}
    if rating_metrics is None:
        rating_metrics = RATING_METRICS_INIT
    summary.update(rating_metrics)
    return summary


# For each data size and each algorithm, a recommender is evaluated.
cols = ["Algo", "Split", "Train time (s)", "Predicting time (s)",
        'MAXE', 'RMSE', 'MeanAE', 'MedianAE', 'R2'
        # , 'Residuals', 'MeanResiduals'
        ]
df_results = pd.DataFrame(columns=cols)

for m in models:
    print('\nModel: ', m)
    models_predictions[m] = {}
    models_residuals[m] = {}

    for idx, (train_index, test_index) in enumerate(tscv.split(X)):
        # Split Train Test
        print("TRAIN: ", len(train_index), "TEST: ", len(test_index))
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]
        # print(X_train)

        # Fit the model
        start_train_time = time.time()
        hist = models[m].fit(X_train, y_train)
        # print(hist)
        train_time = time.time() - start_train_time

        # Make a prediction
        start_predict_time = time.time()
        y_pred = models[m].predict(X_test, y_test)
        predict_time = time.time() - start_predict_time

        # Evaluate
        ratings = evaluate_all_metrics(y_test[:len(y_pred)], y_pred)  # LSTM reduce the size of the test set

        # Save predictions
        models_predictions[m][idx] = y_pred

        # Save residuals
        models_residuals[m][idx] = y_test[:len(y_pred)]["nbCov"].values - y_pred

        # Generate summary
        summary = generate_summary(m, idx, train_time, predict_time, ratings)

        df_results.loc[df_results.shape[0] + 1] = summary

print(df_results)

"""VISUALIZATION"""


def preparte_results_for_printing(model_name, X, models_predictions, split_numero):
    # Get indexes
    for idx, (train_index, test_index) in enumerate(tscv.split(X)):
        if idx == split_numero:
            train_idx = len(train_index)
            test_idx = len(test_index)

    # Get prediction
    pred = models_predictions[model_name][0]

    # Date indexing
    df_hist_tmp = X[target_col][:train_idx]
    df_real_tmp = X[target_col][train_idx:train_idx + test_idx]
    df_pred_tmp = pd.DataFrame({"prediction": pred}, index=X.index[train_idx:train_idx + test_idx])

    return df_hist_tmp, df_real_tmp, df_pred_tmp


def print_results_plt(df_hist_tmp, df_real_tmp, df_pred_tmp):
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_hist_tmp, color="darkblue")
    plt.plot(df_real_tmp, color="blue")
    plt.plot(df_pred_tmp, color="red", marker='o')
    plt.axvline(color='r', x=X.index[train_idx], linestyle='--')


def print_results_echarts(df_hist_tmp, df_real_tmp, df_pred_tmp):
    print(df_pred_tmp)

    line = Line("Model results")
    line.add("Historic", df_hist_tmp.index.date, df_hist_tmp[target_col[0]].values, mark_point=["average"])
    line.add("Prediction", df_pred_tmp.index.date, df_pred_tmp["prediction"], mark_line=["max", "average"])
    # line.show_config()
    return line

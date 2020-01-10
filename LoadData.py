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

"""pp.ProfileReport(df)"""

"""VIZUALISATION"""

plotUtils = PlotUtils(df)

"""Correlation"""
plotUtils.plot_correlation_heatmap(df[target_col+numerical_feature_cols])

"""Histogramme"""
plotUtils.plot_histogram_target(target_col)

"""Mustache box"""
plotUtils.plot_boxplots("nbCov", "M")

"""Graph all variable"""
plotUtils.plot_all_numerical_values(df.columns)

"""Correlation"""
plotUtils.plot_correlation_heatmap(df[target_col+numerical_feature_cols])

"""Correlation graph"""
"""sns.pairplot(df[numerical_feature_cols+target_col])"""






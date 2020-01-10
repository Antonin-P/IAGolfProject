import pandas as pd
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


class PlotUtils:
    def __init__(self, iDf):
        self.iDf = iDf

    def plot_histogram_target(self, iColTarget):
        self.iDf[iColTarget].hist(bins=40)
        plt.xlabel(iColTarget)
        plt.ylabel("Nombre Occurence")
        plt.show()

    def plot_all_numerical_values(self, iColsNumerical):
        # plot each column
        plt.figure(figsize=(15, 8))
        for idx, col in enumerate(iColsNumerical):
            plt.subplot(len(iColsNumerical), 1, idx + 1)
            plt.plot(self.iDf.loc[:, col])
            plt.title(col, y=0.5, loc='right')
        plt.show()

    def plot_boxplots(self, iColTarget, iStep):
        """
        """
        groups = self.iDf.loc[:, iColTarget].groupby(pd.Grouper(freq=iStep))
        steps = []
        data = []
        for name, group in groups:
            data.append(group.values)
            steps.append(name.date())

        plt.figure(figsize=(15, 8))
        plt.boxplot(data)
        plt.xticks(range(1, len(steps) + 1), steps, rotation='vertical')
        plt.show()

    def plot_correlation_heatmap(self, lDf):
        # Compute pairwise correlation of Dataframe's attributes
        corr = lDf.corr()
        # Plot
        fig, (ax) = plt.subplots(1, 1, figsize=(10, 6))

        hm = sns.heatmap(corr,
                         ax=ax,  # Axes in which to draw the plot, otherwise use the currently-active Axes.
                         cmap="coolwarm",  # Color Map.
                         # square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                         annot=True,
                         fmt='.2f',  # String formatting code to use when adding annotations.
                         # annot_kws={"size": 14},
                         linewidths=.05)

        fig.subplots_adjust(top=0.93)
        fig.suptitle('Correlation Heatmap',
                     fontsize=14,
                     fontweight='bold')

    def plot_autocorrelation(self, iTargetCol):
        # ACF
        sm.graphics.tsa.plot_acf(self.iDf[iTargetCol], lags=25)
        plt.show()

    def plot_lag(self, iTargetCol, iLag):
        # Lag plot
        pd.plotting.lag_plot(self.iDf[iTargetCol], lag=iLag)
        plt.show()

    def create_lag_col(self, iTargetCol, iLag, iDropNa=True, iKeepAllLags=True):
        values = pd.DataFrame(self.iDf[iTargetCol].values)
        list_values = [values]
        columns = [iTargetCol]
        # Do you want to keep all the intermediate lags
        if iKeepAllLags:
            for i in range(1, (iLag + 1)):
                list_values.append(values.shift(i))
                columns.append(iTargetCol + '_tplus' + str(i))
        else:
            list_values.append(values.shift(iLag))
            columns.append(iTargetCol + '_tplus' + str(iLag))

        df_lag = pd.concat(list_values, axis=1)
        df_lag.columns = columns
        df_lag.index=self.iDf.index

        if iDropNa:
            df_lag.dropna(inplace=True)

        return df_lag

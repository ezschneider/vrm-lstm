import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

class StatisticalAnalysis:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target

    def anderson_darling_test(self):
        result = stats.anderson(self.df.loc[:, self.target], dist='norm')

        # Prepare the results for tabulation
        headers = ["Significance Level (%)", "Critical Value"]
        table = [[sl, cv] for sl, cv in zip(result.significance_level, result.critical_values)]
        table.append(["Test Statistic", result.statistic])

        # Display the results
        print(tabulate(table, headers=headers, tablefmt="grid"))

        # Interpretation
        interpretation = "The data follows a normal distribution (fail to reject H0)" if result.statistic < result.critical_values[2] else "The data does NOT follow a normal distribution (reject H0)"
        print(interpretation)

    def descriptive_statistics(self):
        desc_stats = self.df[self.target].describe()
        headers = ["Statistic", "Value"]
        table = [[stat, value] for stat, value in desc_stats.items()]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def coefficient_of_variation(self):
        cv = np.std(self.df[self.target]) / np.mean(self.df[self.target])
        headers = ["Statistic", "Value"]
        table = [["Coefficient of Variation", cv]]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def skewness_and_kurtosis(self):
        skewness = self.df[self.target].skew()
        kurtosis = self.df[self.target].kurt()
        headers = ["Statistic", "Value"]
        table = [["Skewness", skewness], ["Kurtosis", kurtosis]]
        print(tabulate(table, headers=headers, tablefmt="grid"))

    def perform_analysis(self):
        self.descriptive_statistics()
        self.coefficient_of_variation()
        self.skewness_and_kurtosis()
        self.anderson_darling_test()

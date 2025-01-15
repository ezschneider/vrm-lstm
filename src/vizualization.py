import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class VibrationPlots:
    def __init__(self, df: pd.DataFrame):
        plt.rcParams.update({'font.size': 12})
        plt.figure(figsize=(12, 6))
        self.df = df.copy()

    def general(self):
        plt.plot(self.df.index, self.df['feature'], label='Vibration', color='blue')
        plt.title('Vibration ao Longo do Tempo')
        plt.xlabel('Data')
        plt.ylabel('Vibration')
        plt.legend()
        plt.show()

    def hist(self):
        sns.histplot(self.df['feature'], kde=True, color='green')
        plt.title('Distribuição das Vibrações')
        plt.xlabel('Vibration')
        plt.ylabel('Frequência')
        plt.show()

    def box(self):
        sns.boxplot(x=self.df['feature'], color='orange')
        plt.title('Boxplot das Vibrações')
        plt.xlabel('Vibration')
        plt.show()

    def corr(self):
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(self.df['feature'])
        plt.title('Correlograma das Vibrações')
        plt.show()

    def z_score(self, feature: str):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.df.index, self.df[feature], label='Vibration')
        ax.plot(self.df.index, self.df['z_score'], label='Z-score', color='r')
        ax.set_title(f'{feature} and Z-score')
        ax.set_xlabel('Time')
        ax.set_ylabel('Vibration')
        ax.legend()
        plt.show()

    def detail_z_score(self, feature: str):
        plt.plot(self.df.index, self.df[feature], label=feature, color='blue')
        plt.plot(self.df.index, self.df['z_score'], label='Z-score', color='red')

        outlier_points = self.df[self.df['z_score'].abs() > 3]
        plt.scatter(outlier_points.index, outlier_points[feature], color='black', label='Outliers')

        plt.xlabel('Time')
        plt.ylabel('Vibration')
        plt.title('Vibration and Z-score (with Outliers)')
        plt.legend()
        plt.show()

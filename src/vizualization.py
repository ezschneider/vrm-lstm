import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class VibrationPlots:
    def __init__(self, df: pd.DataFrame, target: str):
        sns.set_theme(style="whitegrid")  # Define um tema visual moderno
        self.df = df.copy()
        self.target = target

    def general(self):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        
        # General plot
        ax[0].plot(self.df.index, self.df[self.target], label='Vibration', color=sns.color_palette("tab10")[0])
        ax[0].set_title('Vibration along the Time')
        ax[0].set_xlabel('Data')
        ax[0].set_ylabel('Vibration')
        ax[0].legend()

        # Correlation plot
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(self.df[self.target], ax=ax[1])
        ax[1].set_title('Correlation between Vibration and Time')

        plt.show()

    def hist_and_box(self):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

        # Histogram
        sns.histplot(self.df[self.target], kde=True, ax=ax[0], color=sns.color_palette("tab10")[2])
        ax[0].set_title('Distribuição das Vibrações')
        ax[0].set_xlabel('Vibration')
        ax[0].set_ylabel('Frequência')

        # Boxplot
        sns.boxplot(x=self.df[self.target], ax=ax[1], color=sns.color_palette("tab10")[3])
        ax[1].set_title('Boxplot das Vibrações')
        ax[1].set_xlabel('Vibration')

        plt.show()

    def z_score(self, feature: str):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

        # Z-score and Vibration
        ax[0].plot(self.df.index, self.df[feature], label='Vibration', color=sns.color_palette("tab10")[0])
        ax[0].plot(self.df.index, self.df['z_score'], label='Z-score', color=sns.color_palette("tab10")[1])
        ax[0].set_title(f'{feature} and Z-score')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Vibration')
        ax[0].legend()

        # Z-score and Vibration (with Outliers)
        ax[1].plot(self.df.index, self.df[feature], label=feature, color=sns.color_palette("tab10")[0])
        ax[1].plot(self.df.index, self.df['z_score'], label='Z-score', color=sns.color_palette("tab10")[1])

        outlier_points = self.df[self.df['z_score'].abs() > 3]
        ax[1].scatter(outlier_points.index, outlier_points[feature], color='black', label='Outliers')

        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Vibration')
        ax[1].set_title('Vibration and Z-score (with Outliers)')
        ax[1].legend()

        plt.show()

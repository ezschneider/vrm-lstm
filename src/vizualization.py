import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ipywidgets as widgets


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

    def z_score(self, window_size=20):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

        # Z-score and Vibration
        ax[0].plot(self.df.index, self.df[self.target], label='Vibration', color=sns.color_palette("tab10")[0])
        ax[0].plot(self.df.index, self.df['z_score'], label='Z-score', color=sns.color_palette("tab10")[1])
        ax[0].set_title(f'{self.target} and Z-score')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Vibration')
        ax[0].legend()

        # Z-score and Vibration (with Outliers)
        ax[1].plot(self.df.index, self.df[self.target], label=self.target, color=sns.color_palette("tab10")[0])
        # ax[1].plot(self.df.index, self.df['z_score'], label='Z-score', color=sns.color_palette("tab10")[1])

        outlier_points = self.df[self.df['z_score'] > 3]
        ax[1].scatter(outlier_points.index, outlier_points[self.target], color='black', label='Outliers')

        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Vibration')
        ax[1].set_title('Vibration and Z-score (with Outliers)')
        ax[1].legend()

        plt.show()

        # Analyze windows around outliers and plot correlation heatmaps
        # windows = []
        # for outlier_index in outlier_points.index:
        #     start_idx = outlier_index - pd.Timedelta(window_size, unit='min')
        #     end_idx = outlier_index + pd.Timedelta(window_size, unit='min')
        #     
        #     window = self.df.loc[max(self.df.index.min(), start_idx):min(self.df.index.max(), end_idx)]
        #     windows.append(window)
        # 
        #     numeric_window = window.select_dtypes(include=[np.number])
        #     correlations = numeric_window.corr()
        # 
        #     fig, ax = plt.subplots(figsize=(10, 8))
        #     sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        #     ax.set_title(f"Correlation Heatmap around outlier at index {outlier_index}")
        #     plt.show()
        # 
        # return windows

    def seasonal_decompose(self):
        from statsmodels.tsa.seasonal import seasonal_decompose

        decomposition = seasonal_decompose(self.df[self.target], model='additive', period=1440)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12), constrained_layout=True)

        decomposition.observed.plot(ax=ax1, color=sns.color_palette("tab10")[0])
        ax1.set_ylabel('Observed')
        ax1.set_title('Seasonal Decomposition')

        decomposition.trend.plot(ax=ax2, color=sns.color_palette("tab10")[1])
        ax2.set_ylabel('Trend')

        decomposition.seasonal.plot(ax=ax3, color=sns.color_palette("tab10")[2])
        ax3.set_ylabel('Seasonal')

        decomposition.resid.plot(ax=ax4, color=sns.color_palette("tab10")[3])
        ax4.set_ylabel('Residual')
        
        fig.tight_layout()
        plt.show()
    
    def sma_ema(self):
        window_size = 10080  # 1 week of data (7 days * 24 hours * 60 minutes)
        self.df['SMA_7'] = self.df[self.target].rolling(window=window_size).mean()
        self.df['EMA_7'] = self.df[self.target].ewm(span=window_size, adjust=False).mean()

        fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
        ax.plot(self.df.index, self.df[self.target], label="Original", color=sns.color_palette("tab10")[0])
        ax.plot(self.df.index, self.df['SMA_7'], label="SMA (7 days)", linestyle="--", color=sns.color_palette("tab10")[1])
        ax.plot(self.df.index, self.df['EMA_7'], label="EMA (7 days)", linestyle=":", color=sns.color_palette("tab10")[2])
        
        ax.set_title('Simple Moving Average (SMA) and Exponential Moving Average (EMA)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Vibration')
        ax.legend()

        plt.show()
    
    def wavelet_transform(self):
        import pywt
        
        # Perform Continuous Wavelet Transform (CWT)
        scales = np.arange(1, 128)
        coefficients, frequencies = pywt.cwt(self.df[self.target], scales, 'cmor')

        fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
        ax.imshow(np.abs(coefficients), extent=[self.df.index.min(), self.df.index.max(), scales.min(), scales.max()],
                    cmap='PRGn', aspect='auto', vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
        ax.set_title('Continuous Wavelet Transform (CWT)')
        ax.set_ylabel('Scale')
        ax.set_xlabel('Time')
        plt.show()

def plot_outlier_window(df, target, outlier_time, window_size):
    """
    Plota uma janela em torno de um outlier identificado por Z-score maior que 3.
    
    Parâmetros:
        - df: DataFrame contendo os dados.
        - target: Nome da coluna alvo (eixo Y).
        - outlier_time: Índice de tempo do outlier (Timestamp).
        - window_size: Tamanho da janela (inteiro, número de pontos).
    """
    # Verificar se outlier_time está dentro do índice
    if outlier_time not in df.index:
        print("Erro: o índice de tempo fornecido não está no DataFrame.")
        return
    
    # Calcular início e fim da janela
    half_window = window_size // 2
    start_idx = outlier_time - pd.Timedelta(half_window, unit='min')
    end_idx = outlier_time + pd.Timedelta(half_window, unit='min')
    
    # Garantir que os índices estão dentro do intervalo do DataFrame
    start_idx = max(df.index.min(), start_idx)
    end_idx = min(df.index.max(), end_idx)
    
    # Selecionar dados da janela
    window = df.loc[start_idx:end_idx]
    
    # Plotar a janela
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(window.index, window[target], label="Vibration", color=sns.color_palette("tab10")[0])
    # Change the scale of Corrente motor to 0-1
    window["Corrente motor"] = (window["Corrente motor"] - window["Corrente motor"].min()) / (window["Corrente motor"].max() - window["Corrente motor"].min())
    ax.plot(window.index, window["Corrente motor"], label="Motor", color=sns.color_palette("tab10")[1])
    ax.axvline(outlier_time, color='red', linestyle='--', label='Outlier')
    ax.set_title(f'Outlier em {outlier_time} com Janela de {window_size}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Vibration')
    ax.legend()
    plt.show()

# Criar widget interativo
def interactive_outlier_plot(df, target):
    """
    Configura uma interface interativa para escolher o índice do outlier e o tamanho da janela.
    
    Parâmetros:
        - df: DataFrame contendo os dados.
        - target: Nome da coluna alvo (eixo Y).
    """
    # Filtrar os índices dos outliers (Z-score > 3)
    outliers = df[df['z_score'] > 3].index

    # Criar widgets interativos
    outlier_selector = widgets.Dropdown(
        options=[(str(idx), idx) for idx in outliers],
        description="Outlier:",
        disabled=False,
    )
    # Create a button to go to the next index
    button = widgets.Button(description='Next')
    
    # Define the button click event handler
    def on_button_click(b):
        current_index = outlier_selector.index
        next_index = (current_index + 1) % len(outliers)
        outlier_selector.index = next_index
    
    # Attach the event handler to the button
    button.on_click(on_button_click)
    
    # Display the button
    display(button)
    window_size_selector = widgets.IntSlider(
        value=20,
        min=5,
        max=100,
        step=5,
        description="Janela:",
        continuous_update=False,
    )
    
    # Configurar interação
    widgets.interact(
        lambda outlier_time, window_size: plot_outlier_window(df, target, outlier_time, window_size),
        outlier_time=outlier_selector,
        window_size=window_size_selector
    )
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml


class MultiStepTimeSeriesGenerator():
    """
    Copied and edited from https://www.tensorflow.org/tutorials/structured_data/time_series
    """
    def __init__(  # noqa: PLR0917, PLR0913
        self,
        input_width,
        label_width,
        shift,
        intervals,
        batch_size=32,
        target_df=37
    ):

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size = batch_size
        self.total_window_size = input_width + label_width
        self.input_slice = slice(0, input_width)
        self.labels_slice = slice(input_width, None)

        # Preprocess the raw data into datasets
        self.train_data = [
            valor
            for i, valor in enumerate(list(intervals.values())[:160])
            if i != target_df
        ]
        self.val_data = list(intervals.values())[160:]
        self.test_df = list(intervals.values())[target_df]

        self.train = self.concat_intervals(self.train_data)
        self.val = self.concat_intervals(self.val_data)
        self.test = self.concat_intervals(self.test_df)

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # Predict only the first column
        labels = tf.stack([labels[:, :, 0]], axis=-1)

        # Slicing doesn't preserve static shape information.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size
        )

        ds = ds.map(self.split_window)

        return ds

    def concat_intervals(self, intervals):
        if type(intervals) is list:
            datasets = [self.make_dataset(df) for df in intervals]
            concatenated_ds = datasets[0]
            for ds in datasets[1:]:
                concatenated_ds = concatenated_ds.concatenate(ds)
            return concatenated_ds
        return self.make_dataset(intervals)


def first_level_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform the first level processing on the given DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame to be processed.

    Returns:
    - processed_df (pandas.DataFrame): Processed DataFrame.
    """
    # Convert 'Time' column to datetime format if exists
    if 'Time' in df:
        df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M')

    # Set 'Time' column as index
    df.set_index('Time', inplace=True)

    # Reorder the columns: 'Vibration' first
    column_order = ['Vibração'] + [
        col
        for col in df.columns
        if col != 'Vibração'
    ]
    df = df[column_order]
    df = df.rename(columns={'Vibração': 'feature'})

    # Forward fill missing values in 'Blaine' and '#400' columns
    columns_to_fill = ['Blaine', '#400']
    df[columns_to_fill] = df[columns_to_fill].ffill()

    # Convert 'Umidade' column to float if it's not already
    if df['Umidade'].dtype != float:
        df['Umidade'] = df['Umidade'].str.replace('%', '').astype(float)

    # Convert 'Aditivo.1' column to float if it's not already
    if df['Aditivo.1'].dtype != float:
        df['Aditivo.1'] = df['Aditivo.1'].replace('', np.nan).astype(float)

    return df


def save_dataframe(df: pd.DataFrame, filename: str, folder="data"):
    """
    Save DataFrame to a specified folder within the Jupyter notebook environment.

    Parameters:
    - df (pandas.DataFrame): DataFrame to be saved.
    - filename (str): Name of the file to be saved.
    - folder (str, optional): Name of the folder to save the file in. Default is "data".
    """
    # Check if the specified folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Construct the full path to save the file
    filepath = os.path.join(folder, filename)

    # Save the DataFrame to a CSV file
    df.to_csv(filepath, sep=';', index=True)

    print(f"DataFrame saved successfully as '{filename}' in '{folder}' folder.")


def separate_valid_intervals(
    df: pd.DataFrame,
    min_interval=5,
    min_spacing=1
) -> dict[pd.Timestamp, pd.DataFrame]:
    # Drop rows where 'Vibration' column has NaN values
    df_cleaned = df.dropna(subset=['feature']).copy()

    # Calculate time difference between consecutive rows
    df_cleaned['TimeDiff'] = df_cleaned.index.to_series().diff().fillna(
        pd.Timedelta(seconds=0)
    )

    # Filter intervals with more than 5 hours of sequential data spaced every 1 minute
    min_interval = pd.Timedelta(hours=min_interval)
    min_spacing = pd.Timedelta(minutes=min_spacing)

    # Group consecutive rows by a common key when time difference exceeds min_spacing
    group_key = (df_cleaned['TimeDiff'] > min_spacing).cumsum()

    # Calculate the duration of each group
    group_duration = df_cleaned.groupby(group_key).size() * min_spacing
    # Use the index of the first row in each group
    group_duration.index = df_cleaned.groupby(group_key).head(1).index

    # Ensure the boolean Series has the same index as df_cleaned
    boolean_index = group_duration.reindex(df_cleaned.index)

    # Extract entire intervals based on group duration
    valid_intervals = {}
    for interval_id, duration in group_duration.items():
        if duration >= min_interval:
            # Get start and end index of the interval
            start_index = df_cleaned[boolean_index.index == interval_id].index[0]
            end_index = start_index + duration

            # Slice the original DataFrame to extract the entire interval data
            interval_data = df_cleaned.loc[start_index:end_index]
            interval_data.reset_index(inplace=True)
            interval_data.set_index('Time', inplace=True, drop=True)
            interval_data.drop(columns=['TimeDiff', 'Unnamed: 58'], inplace=True)
            # interval_data.ffill()
            interval_data.dropna(subset=interval_data.columns.values, inplace=True)

            valid_intervals[interval_id] = interval_data[20:len(interval_data) - 200]

    return valid_intervals


def preprocessing_pipeline():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Read the CSV file
    df = pd.read_csv(config['data']['raw_path'], sep=';')

    # Process the DataFrame
    processed_df = first_level_process(df)

    # Save the processed DataFrame
    save_dataframe(processed_df, config['data']['processed_filename'])

    # Separate valid intervals
    valid_intervals = separate_valid_intervals(
        processed_df,
        min_interval=config['data']['min_interval'],
        min_spacing=config['data']['min_spacing']
    )

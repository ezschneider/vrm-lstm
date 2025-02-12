# %%
import requests
from dotenv import load_dotenv
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.schema import ApiResponse

load_dotenv(".env")

API_URL = os.getenv("API_URL")
API_HEADERS = {"apikey": os.getenv("API_KEY")}
# %%
def get_data():
    web_id = "F1DP-7fYgsRTtUOa7V9NIwSujAts0EAAUElIQVZDXFBDLVoyTTAxWDE"
    # start_time = "*-13h"
    # end_time = "03-04-2024"
    start_time = "*-1d"
    end_time = "*"
    selected_fields = "Items.Name;Items.Items.Timestamp;Items.Items.Value.Value"

    response = requests.get(
        API_URL + f"webid={web_id}&startTime={start_time}&endTime={end_time}&selectedFields={selected_fields}",
        headers=API_HEADERS
    )
    return response.json()

# %%
def parse_response(data):
    return ApiResponse(**data)

response_data = get_data()
response_data
# parse_response(response_data)
# %%

# Extract the relevant data
items = response_data['Items'][0]['Items']
timestamps = [item['Timestamp'] for item in items]
values = [item['Value'] for item in items]

# Create a DataFrame
df = pd.DataFrame({'Timestamp': timestamps, 'Value': values})

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# %%
# exclude the rows with the 'Value' equal to '{'Value': 240}'
df = df[df['Value'] != {'Value': 240}]
df.head()
# %%

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Value'], marker='o')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.title('Timestamp vs Value')
plt.grid(True)
plt.show()
# %%

# plot 30min before and after the timestamp equal to 11:30
def plot_30min(df, timestamp):
    start_time = timestamp - pd.Timedelta(minutes=30)
    end_time = timestamp + pd.Timedelta(minutes=30)
    df_30min = df[(df.index >= start_time) & (df.index <= end_time)]

    plt.figure(figsize=(10, 5))
    plt.plot(df_30min.index, df_30min['Value'], marker='o')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.title('Timestamp vs Value')
    plt.grid(True)
    plt.show()

df.index = df.index.tz_localize(None)
plot_30min(df, pd.Timestamp('2024-03-04 11:30:00'))
# %%

# %%
import requests
from dotenv import load_dotenv
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.schema import ApiResponse

load_dotenv(".env")

API_URL = os.getenv("API_URL")
API_URL_WEB_ID = os.getenv("API_URL_WEB_ID")
API_HEADERS = {
    "apikey": os.getenv("API_KEY"),
    "User-Agent": "PostmanRuntime/7.43.0",
    "Date": "Wed, 26 Feb 2025 22:05:30 GMT",
    "Content-Type": "application/json; charset=utf-8",
    "Transfer-Encoding": "chunked",
    "Connection": "keep-alive",
    "cache-control": "no-cache",
    "cf-cache-status": "DYNAMIC",
    "Set-Cookie": "__cf_bm=Gi8_6cd52epI_dtLlx8AG3u51xcK7UOqxxRwCYYay0g-1740607530-1.0.1.1-kRbr2gjYcUQouiUDnjV94DAIcGSpwt02tXaWT7i.A95R3oR43Hip0FT7Od1bd9Kpix1iJeWEUkrCeWiKEH0mFA; path=/; expires=Wed, 26-Feb-25 22:35:30 GMT; domain=.votorantimcimentos.com; HttpOnly; Secure; SameSite=None",
    "Content-Security-Policy": "'default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Strict-Transport-Security": "max-age=15768000",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "X-XSS-Protection": "1; mode=block",
    "Server": "cloudflare",
    "CF-RAY": "91833ca8fce8f171-GRU",
    "Content-Encoding": "gzip"
}
# %%
def get_data():
    web_id = "F1DP-7fYgsRTtUOa7V9NIwSujAt80EAAUElIQVZDXFBDLVoyTTAxWDI"
    # web_id = "F1DP-7fYgsRTtUOa7V9NIwSujAKQwFAAUElIQVZDXFBDLVoyTTAzTTFYMl9QUF9GSUw"
    # web_id = "F1DP-7fYgsRTtUOa7V9NIwSujAts0EAAUElIQVZDXFBDLVoyTTAxWDE"
    start_time = "*-360d"
    end_time = "*-359d"
    # start_time = "*-1d"
    # end_time = "*"
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
    start_time = timestamp - pd.Timedelta(minutes=10)
    end_time = timestamp + pd.Timedelta(minutes=10)
    df_30min = df[(df.index >= start_time) & (df.index <= end_time)]

    plt.figure(figsize=(10, 5))
    plt.plot(df_30min.index, df_30min['Value'], marker='o')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.title('Timestamp vs Value')
    plt.grid(True)
    plt.show()

df.index = df.index.tz_localize(None)
plot_30min(df, pd.Timestamp('2024-12-05 4:00:00'))
# %%

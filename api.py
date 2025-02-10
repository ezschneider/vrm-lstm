# %%
import requests
from dotenv import load_dotenv
import os

from src.schema import ApiResponse

load_dotenv(".env")

API_URL = os.getenv("API_URL")
API_HEADERS = {"apikey": os.getenv("API_KEY")}
# %%
def get_data():
    web_id = "F1DP-7fYgsRTtUOa7V9NIwSujAYtYEAAUElIQVZDXFBDLVoyUFJPRFVDQU8"
    start_time = "*-1h"
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

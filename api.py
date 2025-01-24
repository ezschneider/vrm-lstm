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
    response = requests.get(
        API_URL + "F1DP-7fYgsRTtUOa7V9NIwSujAYtYEAAUElIQVZDXFBDLVoyUFJPRFVDQU8/recorded?startTime=2024-09-01&endTime=2024-09-09",
        headers=API_HEADERS
    )
    return response.json()

# %%
def parse_response(data):
    return ApiResponse(**data)

response_data = get_data()
parse_response(response_data)
# %%

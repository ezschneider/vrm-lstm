# %%
import requests
from dotenv import load_dotenv
import os

from .src.schema import ApiResponse

load_dotenv()

API_URL = os.getenv("API_URL")
API_HEADERS = {"apikey": os.getenv("API_KEY")}

# %%
def get_data():
    response = requests.get(API_URL, headers=API_HEADERS)
    return response.json()

# %%
get_data()
# %%
def parse_response(data):
    return ApiResponse(**data)

response_data = get_data()
parsed_data = parse_response(response_data)
print(parsed_data)
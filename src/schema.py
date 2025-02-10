from pydantic import BaseModel
from typing import List


class TimestampValue(BaseModel):
    Timestamp: str
    Value: float


class Item(BaseModel):
    Name: str
    Items: List[TimestampValue]


class ApiResponse(BaseModel):
    Items: List[Item]

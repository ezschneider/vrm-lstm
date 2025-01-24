from pydantic import BaseModel


class Item(BaseModel):
    Timestamp: str
    Value: float


class ApiResponse(BaseModel):
    Items: list[Item]

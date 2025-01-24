from pydantic import BaseModel

class Item(BaseModel):
    Timestamp: str
    Value: float
    UnitsAbbreviation: str
    Good: bool
    Questionable: bool
    Substituted: bool
    Annotated: bool

class ApiResponse(BaseModel):
    Links: dict
    Items: list[Item]
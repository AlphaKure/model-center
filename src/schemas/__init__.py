from pydantic import BaseModel

class BasicReturn(BaseModel):
    message: str
    detail: str
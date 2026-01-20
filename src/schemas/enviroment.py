from pydantic import BaseModel

class TorchInfomation(BaseModel):
    
    version: str
    isAvailable: bool
    deviceCount: int
    devices: list[str]
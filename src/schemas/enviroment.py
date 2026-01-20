from pydantic import BaseModel

class TorchInfomation(BaseModel):
    
    version: str
    is_available: bool
    device_count: int
    devices: list[str]
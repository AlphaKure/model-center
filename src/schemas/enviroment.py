from pydantic import BaseModel, Field

class TorchInformation(BaseModel):
    
    version: str = Field(examples=["2.9.0+cu129"], description= "Torch version")
    is_available: bool = Field(examples=[True], description= "Cuda available")
    device_count: int = Field(examples=[1], description= "How many cuda device")
    devices: list[str] = Field(examples=[["RTX 6090"]], description= "Cuda device name")
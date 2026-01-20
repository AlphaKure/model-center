from typing import Literal

from pydantic import BaseModel

class LoadModel(BaseModel):
    modelPath: str
    dtype: Literal["bfloat16", "auto"]
    offload: bool
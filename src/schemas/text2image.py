from typing import Literal

from pydantic import BaseModel

class LoadModel(BaseModel):
    modelPath: str
    dtype: Literal["bfloat16", "auto"]
    mode: Literal["balanced", "cuda"]
    #offload: bool
    outputPath: str

class Inference(BaseModel):

    prompt: str
    width: int
    height: int
    steps: int
    scale: float
    negativePrompt: str 

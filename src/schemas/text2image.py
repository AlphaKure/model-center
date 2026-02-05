from typing import Literal

from pydantic import BaseModel, Field

class LoadModel(BaseModel):
    modelPath: str = Field(examples= ["/path/to/model/folder"], description= "Point to model folder")
    dtype: Literal["bfloat16", "float16", "auto"] = Field(examples=["bfloat16"], description= "Model load dtype")
    mode: Literal["balanced", "cuda"] = Field(examples=["cuda"], description= "Use cuda mode for best performance")
    #offload: bool
    outputPath: str = Field(examples=["/path/to/output/folder"], description= "Point to output folder")

class Inference(BaseModel):

    prompt: str = Field(examples=["generate prompts"], description= "Generate prompt")
    width: int = Field(examples=[1024], description= "Generate width")
    height: int = Field(examples=[1024], description= "Generate height")
    steps: int = Field(examples=[1], description= "Generate steps")
    scale: float = Field(examples=[4.0], description= "Generate scale")
    negativePrompt: str = Field(examples=["generate negative prompt"], description= "Generate negative prompt")

class RecommendParams(BaseModel):

    scale: float = Field(examples=[4.0], description= "Recommend generate scale")
    steps: int = Field(examples=[1], description= "Recommend generate steps")
from typing import Literal, Tuple
import gc

from diffusers import AutoPipelineForText2Image
import torch

class TextToImageEngine:

    pipeline = None

    @classmethod 
    def load_model(
            cls, 
            modelPath: str,
            dtype: Literal["bfloat16", "auto"],
            offload: bool = False # offload to cpu
    )-> Tuple[int, str, str]: # code message detail

        if cls.pipeline is not None:
            return 400, "Already load model", "Model already loaded. Please unload it first" 

        if dtype == "bfloat16":
            dtype = torch.bfloat16

        try:
            cls.pipeline = AutoPipelineForText2Image.from_pretrained(
                modelPath,
                torch_dtype= dtype,
                trust_remote_code = True 
            )
            cls.pipeline.to("cuda")
        except Exception as error:
            return 500, "Load model error", str(error)
        
        if offload:
            cls.pipeline.enable_model_cpu_offload()

        return 200, "Success", ""
    
    @classmethod
    def unload_model(cls)-> Tuple[int, str, str]: # code message detail

        if cls.pipeline is None:
            return 400, "Model didn't load", "Model didn't load"

        del cls.pipeline
        cls.pipeline = None

        gc.collect()
        torch.cuda.empty_cache()
        return 200, "Success", ""

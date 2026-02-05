from typing import Literal, Tuple, Any
import gc
import asyncio
from functools import partial
from datetime import datetime
import os
import inspect
import json

from src.modules import _callback, Progress
from src.schemas.text2image import RecommendParams

from diffusers import AutoPipelineForText2Image, ZImagePipeline, Flux2KleinPipeline
import torch

class TextToImageEngine:

    pipeline = None
    isRunning: bool = False
    outputPath: str

    @classmethod 
    def load_model(
            cls, 
            modelPath: str,
            dtype: Literal["bfloat16", "float16", "auto"],
            mode: Literal["balanced", "cuda"],
            outputPath: str,
            #offload: bool = False, # offload to cpu
            
    )-> Tuple[int, str, str]: # code message detail

        if cls.pipeline is not None:
            return 400, "Already load model", "Model already loaded. Please unload it first" 

        if dtype == "bfloat16":
            dtype = torch.bfloat16
        
        if dtype == "float16": 
            dtype = torch.float16

        try:
            cls.pipeline = AutoPipelineForText2Image.from_pretrained(
                modelPath,
                torch_dtype= dtype,
                trust_remote_code = True,
                device_map = mode
            )
        except Exception as error:
            return 500, "Load model error", str(error)
        
        #if offload:
        #    cls.pipeline.enable_model_cpu_offload()

        if not os.path.isdir(outputPath):
            return 404, "Output path not exist", "Output path not exist"
        cls.outputPath = outputPath

        return 200, "Success", ""
    
    @classmethod
    def unload_model(cls)-> Tuple[int, str, str]: # code message detail

        if cls.pipeline is None:
            return 400, "Model didn't load", "Model didn't load"

        if cls.isRunning:
            return 400, "Another task running", "Another task running"

        del cls.pipeline
        cls.pipeline = None

        gc.collect()
        torch.cuda.empty_cache()
        return 200, "Success", ""

    @classmethod
    def get_recommand_parameter(cls)-> Tuple[int, str|dict[str, Any], str|None]: # code message detail

        if cls.pipeline is None:
            return 400, "Model didn't load", "Model didn't load"
        
        # This is not a good idea. Just for origin model.
        if isinstance(cls.pipeline, ZImagePipeline):
            return 200, RecommendParams(
                steps= 9,
                scale= 0.0
            ), None
        elif isinstance(cls.pipeline, Flux2KleinPipeline):
            return 200, RecommendParams(
                steps= 4,
                scale= 1.0
            ), None 
        else:
            return 404, "Unknown pipeline type ", "Unknown pipeline type"

    @classmethod
    def _inference(cls, loop:asyncio.AbstractEventLoop, queue:asyncio.Queue , prompt: str, width: int, height: int, steps: int, scale: float = 0.0, negativePrompt: str = ""):
        
        supportArgs = inspect.signature(cls.pipeline.__call__).parameters # Get pipeline generate parameter dictionary
        
        generateArgs = dict(
            prompt= prompt,
            width= width,
            height= height,
            num_inference_steps= steps,
            guidance_scale= scale,
            callback_on_step_end= partial(_callback, loop, queue, steps),
        )
        # Some of model not support negative_prompt
        if "negative_prompt" in supportArgs and negativePrompt:
            generateArgs["negative_prompt"] = negativePrompt

        try:
            newImage = cls.pipeline(
                **generateArgs                
            )
            currentTime = datetime.now().strftime('%Y%m%d_%H%M%S')
            newImage.images[0].save(os.path.join(cls.outputPath, f"{currentTime}.png"))
            loop.call_soon_threadsafe(queue.put_nowait, Progress(result= str(os.path.join(cls.outputPath, f"{currentTime}.png")), error= "", percentage= 100.0, statusCode= 200).dict())
        except Exception as error:
            loop.call_soon_threadsafe(queue.put_nowait, Progress(result= "", error= str(error), percentage= 0.0, statusCode= 500).dict())
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            cls.isRunning = False

    
    @classmethod
    async def interface(cls, prompt: str, width: int, height: int, steps: int, scale: float = 0.0, negativePrompt: str = ""):

        if cls.pipeline is None:
            return 400, "Model didn't load", "Model didn't load"
        
        if cls.isRunning:
            return 409, "Another task running", "Another task running"
        
        loop = asyncio.get_running_loop()
        tracker = asyncio.Queue()
        asyncio.create_task(asyncio.to_thread(cls._inference, loop, tracker, prompt, width, height, steps, scale, negativePrompt))

        async def get_process():
            while True:
                process = await tracker.get()
                if not process:
                    break                   
                yield f"data: {json.dumps(process)}\n\n"
                await asyncio.sleep(0.1)

        
        return 200, get_process(), None


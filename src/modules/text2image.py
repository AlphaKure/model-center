from typing import Literal, Tuple, Any
import gc
import asyncio
from functools import partial
from datetime import datetime
import os

from src.modules import _callback, Progress

from diffusers import AutoPipelineForText2Image, ZImagePipeline
import torch

class TextToImageEngine:

    pipeline = None
    isRunning: bool = False

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

    @classmethod
    def get_recommand_parameter(cls)-> Tuple[int, str|dict[str, Any], str|None]: # code message detail

        if cls.pipeline is None:
            return 400, "Model didn't load", "Model didn't load"
        
        if isinstance(cls.pipeline, ZImagePipeline):
            return 200, {
                "steps": 9,
                "scale": 0.0
            }, None
        else:
            return 404, "Unknown pipeline type ", "Unknown pipeline type"

    @classmethod
    def _inference(cls, loop:asyncio.AbstractEventLoop, queue:asyncio.Queue , prompt: str, width: int, height: int, steps: int, scale: float = 0.0, negativePrompt: str = ""):
        try:
            newImage = cls.pipeline(
                prompt= prompt,
                width= width,
                height= height,
                num_inference_steps= steps,
                guidance_scale= scale,
                negative_prompt= negativePrompt,
                callback_on_step_end= partial(_callback, loop, queue, steps)
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
            return 102, "Another task running", "Another task running"
        
        loop = asyncio.get_running_loop()
        tracker = asyncio.Queue()
        asyncio.create_task(asyncio.to_thread(cls._inference, loop, tracker, prompt, width, height, steps, scale, negativePrompt))

        async def get_process():
            while True:
                process = await tracker.get()
                if not process:
                    break                   
                yield f"{process}\n"
                await asyncio.sleep(0.1)

        
        return 200, get_process(), None


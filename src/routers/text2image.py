from src.modules.text2image import TextToImageEngine
from src.schemas.text2image import LoadModel, Inference, RecommendParams
from src.schemas import BasicReturn, Progress

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

route = APIRouter(
    prefix= "/api/t2i",
    tags= ["text2image"]
)

@route.put(
    "",
    responses= {
        200: {"description": "Success", "model": BasicReturn},
        404: {"description": "Output path not exist", "model": BasicReturn},
        500: {"description": "Exception error", "model": BasicReturn}
    }
)
def load_model(body: LoadModel):
    code, message, detail = TextToImageEngine.load_model(**body.dict())
    return JSONResponse(status_code= code, content=BasicReturn(message= message, detail= detail).dict())

@route.delete(
    "",
    responses= {
        200: {"description": "Success", "model": BasicReturn},
        400: {"description": "Model didn't load or another task running", "model": BasicReturn},
    }
)
def unload_model():
    code, message, detail = TextToImageEngine.unload_model()
    return JSONResponse(status_code= code, content=BasicReturn(message= message, detail= detail).dict())

@route.get(
    "",
    responses ={
        200: {"description": "Success", "model": RecommendParams},
        400: {"description": "Model didn't load", "model": BasicReturn},
        404: {"description": "Unknown Pipeline", "model": BasicReturn}
    }
)
def get_recommand_parameter():
    """Only working on origin models."""
    code, message, detail = TextToImageEngine.get_recommand_parameter()
    if code ==200:
        return message
    else:
        return JSONResponse(status_code= code, content=BasicReturn(message= message, detail= detail).dict())
    
@route.post(
    "",
    responses= {
        200: {"description": "Success. This is the Streaming Response", "model": Progress},
        409: {"description": "Another task running", "model": BasicReturn},
        400: {"description": "Model didn't load", "model": BasicReturn},
    }
)
async def inference(request:Inference):

    code, message, detail = await TextToImageEngine.interface(**request.model_dump())
    if code!= 200:
        return JSONResponse(status_code= code, content= BasicReturn(message= message, detail= detail).dict())
    return StreamingResponse(message, media_type="application/json")

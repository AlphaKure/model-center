from src.modules.text2image import TextToImageEngine
from src.schemas.text2image import LoadModel, Inference
from src.schemas import BasicReturn

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

route = APIRouter(
    prefix= "/api/t2i",
    tags= ["text2image"]
)

@route.put("")
def load_model(body: LoadModel):
    code, message, detail = TextToImageEngine.load_model(**body.dict())
    return JSONResponse(status_code= code, content=BasicReturn(message= message, detail= detail).dict())

@route.delete("")
def unload_model():
    code, message, detail = TextToImageEngine.unload_model()
    return JSONResponse(status_code= code, content=BasicReturn(message= message, detail= detail).dict())

@route.get("")
def get_recommand_parameter():
    code, message, detail = TextToImageEngine.get_recommand_parameter()
    if code ==200:
        return message
    else:
        return JSONResponse(status_code= code, content=BasicReturn(message= message, detail= detail).dict())
    
@route.post("")
async def inference(request:Inference):

    code, message, detail = await TextToImageEngine.interface(**request.model_dump())
    if code!= 200:
        return JSONResponse(status_code= code, content= BasicReturn(message= message, detail= detail).dict())
    return StreamingResponse(message)

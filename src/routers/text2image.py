from src.modules.text2image import TextToImageEngine
from src.schemas.text2image import LoadModel
from src.schemas import BasicReturn

from fastapi import APIRouter
from fastapi.responses import JSONResponse

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
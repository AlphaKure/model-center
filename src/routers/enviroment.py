from src.modules.enviroment import torch_test
from src.schemas.enviroment import TorchInformation

from fastapi import APIRouter

route= APIRouter(
    prefix="/api/env",
    tags= ["enviroment"]
)

@route.get(
    path= "/",
    response_model= TorchInformation,
    responses= 
        {
            200: {"description": "Success"}
        }
    )
def check_torch_version():
    """Check your PyTorch"""
    return torch_test()
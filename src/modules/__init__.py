import asyncio

from src.schemas import Progress

from pydantic import BaseModel

def _callback(loop: asyncio.AbstractEventLoop , queue: asyncio.Queue, totalSteps: int,  *args , ** kwargs):
    """for pipeline return progress"""
    
    if len(args) == 4:
        __ , step, __, callback_kwargs = args
    elif len(args) == 3:
        step, __, callback_kwargs = args
    if step is not None:
        percentage = round((int(step)+1)/totalSteps*100,2)
        loop.call_soon_threadsafe(queue.put_nowait,Progress(result= "", error= "", percentage= percentage, statusCode= 102).dict())
    return callback_kwargs if isinstance(callback_kwargs, dict) else {}

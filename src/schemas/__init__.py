from pydantic import BaseModel, Field

class BasicReturn(BaseModel):
    message: str = Field(examples= ["outline message"], description= "Outline message. Such as 'Success' or 'Request Error' ")
    detail: str = Field(examples= ["detail message"], description= "Detail message. use on error report. ")

class Progress(BaseModel):
    """Pipeline Progress"""
    result: str = Field(examples= ["/path/to/result"], description= "Finish will return here.")
    error: str = Field(examples=["error message"], description= "Error message will return here.")
    percentage: float = Field(examples=[97.5], description= "Progress")
    statusCode: int = Field(examples=[102,200,500], description= "Running status 102: running. 200: finish. 500: error")
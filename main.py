from src import routers

from fastapi import FastAPI
import uvicorn

app = FastAPI(
    version= "0.0.0",
    title= "model-center",
    docs_url= "/api"
)

app.include_router(routers.enviroment_route)
app.include_router(routers.t2i_route)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port= 8080)

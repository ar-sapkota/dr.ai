# from fastapi import FastAPI
# from app.api.chat import router as chat_router
# from app.api.image import router as image_router

# app = FastAPI()

# # prefix = '/api' adds /api in route. i.e domain.com/api/messages. it helps to create version of api.
# # tags=["chat"] will group all endpoints with same tags under a "Chat" heading
# app.include_router(chat_router, prefix="/api", tags=["Chat"])
# app.include_router(image_router, prefix="/api", tags=["Image"])

# @app.get("/")
# def home():
#     return {"message": "Dr AI Backend Running"}


from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.image import router as image_router


app = FastAPI()

app.include_router(chat_router)

app.include_router(image_router)


@app.get("/")
def home():

    return {"message": "Dr Sahab Backend Running"}
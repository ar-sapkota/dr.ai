from fastapi import FastAPI
from app.api.chat import router as chat_router

# OLD:
# from app.api.chat import router as chat_router
# from app.api.image import router as image_router
# app.include_router(image_router)  ← remove this

app = FastAPI()

app.include_router(chat_router)

@app.get("/")
def home():
    return {"message": "Dr Sahab Backend Running"}
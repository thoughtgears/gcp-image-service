from fastapi import FastAPI
from routes import images

app = FastAPI()

app.include_router(images.router, prefix="/images")


@app.get("/health")
async def health():
    return {"status": "ok"}

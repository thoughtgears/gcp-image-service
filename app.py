from fastapi import FastAPI
from routes import images
from routes import symantic

app = FastAPI()

app.include_router(images.router, prefix="/images")
app.include_router(symantic.router, prefix="/symantic")


@app.get("/health")
async def health():
    return {"status": "ok"}

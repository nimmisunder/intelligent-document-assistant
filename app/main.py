from fastapi import FastAPI
from app.routes import documents
from app.database import init_db
from app.routes import chat
from app.routes import evaluation
init_db()

app = FastAPI(
    title="Intelligent Document Assistant",
    description="AI-powered document processing",
    version="1.0.0"
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(evaluation.router)
@app.get("/")
def home():
    return {
        "message": "Intelligent Document Assistant API is running"
    }
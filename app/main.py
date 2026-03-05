from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import api
from app.db import engine, Base # Import engine và Base
from app.models import db_models # Import để SQLAlchemy nhận diện model

# Lệnh này sẽ tự động tạo bảng text_classifications trong Postgres nếu nó chưa tồn tại
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Text Classification API",
    description="API to classify text using Jina AI Embeddings v5",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(api.router)
from transformers import pipeline
import logging
import os
from dotenv import load_dotenv
from app.db import SessionLocal
from app.models.db_models import TextClassification

load_dotenv()

logger = logging.getLogger(__name__)

# Load model Jina AI
try:
    logger.info("Loading model distilbert-base-uncased-finetuned-sst-2-english...")
    classifier = pipeline(
        "text-classification", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    logger.info("Load model successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    classifier = None

#Function to get prediction
def get_prediction(text: str):
    if not classifier:
        raise RuntimeError("Model not initialized")
    
    # Inference process
    result = classifier(text)[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 4)
    }

def save_to_db(text: str, label: str, score: float) -> bool:
    db = SessionLocal() # Mở kết nối
    try:
        record = TextClassification(
            input_text=text,
            predicted_label=label,
            confidence_score=score
        )
        db.add(record) # Thêm vào session
        db.commit() # Lưu thay đổi xuống Postgres
        db.refresh(record)
        return record
    except Exception as e:
        logger.error(f"Error saving to DB: {e}")
        db.rollback() # Hoàn tác nếu có lỗi
        return False
    finally:
        db.close()


def get_recent_results(limit: int = 10):
    db = SessionLocal()

    try:
        records = (
            db.query(TextClassification)
            .order_by(TextClassification.created_at.desc())
            .limit(limit)
            .all()
        )
        # Chuyển đổi dữ liệu từ dạng object sang dictionary để trả về API
        return [
            {
                "id": r.id,
                "input_text": r.input_text,
                "predicted_label": r.predicted_label,
                "confidence_score": r.confidence_score,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
    except Exception as e:
        logger.error(f"Error fetching from Postgres: {e}")
        raise
    finally:
        db.close()
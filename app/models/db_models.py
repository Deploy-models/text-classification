from sqlalchemy import Column, Integer, String, Float, DateTime, func
from app.db import Base

class TextClassification(Base):
    __tablename__ = "text_classifications"
    
    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, nullable=False)
    predicted_label = Column(String(50), nullable=False)
    confidence_score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
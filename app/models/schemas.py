from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Text to classify")

class PredictResponse(BaseModel):
    label: str
    score: float
    message: str
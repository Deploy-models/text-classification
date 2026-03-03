from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import TextRequest, PredictResponse
from app.services.core import get_prediction, save_to_supabase, get_recent_results

router = APIRouter()
#Root endpoint
@router.get("/", tags=["System"])
async def root():
    return {
        "message": "Welcome to the Text Classification API. Use POST /predict or GET /predict?text=your_text."
    }

#Health check endpoint
@router.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "service": "text-classification-api"}

#Predict endpoint
@router.post("/predict", response_model=PredictResponse, tags=["AI API"])
async def predict_text(request: TextRequest):
    try:
        pred = get_prediction(request.text)
        
        is_saved = save_to_supabase(
            text=request.text,
            label=pred["label"],
            score=pred["score"]
        )
        if is_saved:
            message = "Prediction successful and saved to DB"
        else:
            message = "Prediction successful but failed to save to DB"

        return PredictResponse(
            label=pred["label"],
            score=pred["score"],
            message=message
        )
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")

#Get recent results endpoint
@router.get("/results", tags=["AI API"])
async def get_recent_results_endpoint(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Number of recent results to return",
    )
):
    try:
        results = get_recent_results(limit=limit)
        return {
            "limit": limit,
            "count": len(results),
            "results": results,
        }
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching results: {str(e)}")
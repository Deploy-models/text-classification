from transformers import pipeline
import logging
from supabase import create_client, Client
import os
from dotenv import load_dotenv

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

#connect to supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase_client: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Connected to Supabase successfully.")
    except Exception as e:
        logger.error(f"Error connecting to Supabase: {e}")
else:
    logger.warning("Missing SUPABASE_URL or SUPABASE_KEY.")

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

def save_to_supabase(text: str, label: str, score: float):
    if not supabase_client:
        logger.error("Cannot save, Supabase not configured.")
        return False
    
    try:
        response = supabase_client.table("text_classifications").insert({
            "input_text": text,
            "predicted_label": label,
            "confidence_score": score
        }).execute()
        
        if response.data:
            logger.info(f"Successfully saved record to Supabase: {label}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error saving to Supabase: {e}")
        return False


def get_recent_results(limit: int = 10):
    if not supabase_client:
        logger.error("Cannot fetch, Supabase not configured.")
        raise RuntimeError("Supabase client not initialized")

    try:
        response = (
            supabase_client
            .table("text_classifications")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as e:
        logger.error(f"Error fetching recent results from Supabase: {e}")
        raise
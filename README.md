## AI Text Classification API

A simple REST API for text classification built with FastAPI and Hugging Face `transformers`.  
The service classifies input text (sentiment using `distilbert-base-uncased-finetuned-sst-2-english`) and optionally stores predictions in a Supabase database for later inspection.

### Features

- **FastAPI backend** with CORS support (configured for `http://localhost:3000` by default).
- **Text classification** using Hugging Face `transformers` pipeline.
- **Supabase integration** to store predictions and retrieve recent results.
- **Health and system endpoints** for monitoring.
- **Pydantic models** for input validation and structured responses.

---

### Project Structure

- **`app/main.py`**: FastAPI app initialization, middleware, and router registration.
- **`app/routers/api.py`**: API endpoints (`/`, `/health`, `/predict`, `/results`).
- **`app/services/core.py`**: Core logic for:
  - Loading the Hugging Face classifier
  - Connecting to Supabase
  - Running predictions
  - Saving and fetching results from Supabase
- **`app/models/schemas.py`**: Pydantic request/response models.
- **`requirements.txt`**: Python dependencies.
- **`.env`**: Environment variables (Supabase configuration, etc.).
- **`docker/`**: Placeholder for Docker/Docker Compose setup (not yet implemented).

---

### Requirements

- **Python**: 3.9+ (recommended)
- **Dependencies** (from `requirements.txt`):
  - `fastapi`
  - `uvicorn`
  - `pydantic`
  - `transformers`
  - `torch`
  - `psycopg2-binary`
  - `sqlalchemy`
  - `supabase`
  - `dotenv` (used to load environment variables from `.env`)

---

### Environment Variables

Create a `.env` file in the project root with at least:

- **`SUPABASE_URL`**: Your Supabase project URL.
- **`SUPABASE_KEY`**: Your Supabase service role or anon key (with permission to insert/select from the `text_classifications` table).

Example `.env`:

```env
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
```

---

### Supabase Table Schema

The API expects a Supabase table named `text_classifications` with at least the following columns:

- **`id`**: primary key (e.g., `uuid` or `bigint`, auto-generated).
- **`input_text`**: text, the original input string.
- **`predicted_label`**: text, model output label (e.g., `POSITIVE` / `NEGATIVE`).
- **`confidence_score`**: numeric/float, model confidence.
- **`created_at`**: timestamp, default `now()` (used for ordering recent results).

---

### Running the Application

1. **Create and activate a virtual environment** (optional but recommended).
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Set up `.env`** with your Supabase credentials (see above).
4. **Run the FastAPI app with Uvicorn**:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Open API docs** in your browser:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

---

### API Endpoints

#### `GET /` (Root)

- **Description**: Welcome message and basic usage hint.
- **Response**:

```json
{
  "message": "Welcome to the Text Classification API. Use POST /predict or GET /predict?text=your_text."
}
```

#### `GET /health`

- **Description**: Health check endpoint.
- **Response**:

```json
{
  "status": "healthy",
  "service": "text-classification-api"
}
```

#### `POST /predict`

- **Description**: Classify a piece of text and attempt to save the result in Supabase.
- **Request body** (`TextRequest`):

```json
{
  "text": "I love using this API!"
}
```

- **Response** (`PredictResponse`):

```json
{
  "label": "POSITIVE",
  "score": 0.9987,
  "message": "Prediction successful and saved to DB"
}
```

- **Notes**:
  - If Supabase is not configured or saving fails, `message` will indicate that saving to DB failed, but the prediction will still be returned.
  - If the model is not initialized correctly, the API returns a `503` error.

#### `GET /results?limit={n}`

- **Description**: Fetch recent prediction results from Supabase.
- **Query parameters**:
  - **`limit`** (int, default `10`, min `1`, max `100`): number of records to return.
- **Response**:

```json
{
  "limit": 10,
  "count": 10,
  "results": [
    {
      "id": "...",
      "input_text": "I love using this API!",
      "predicted_label": "POSITIVE",
      "confidence_score": 0.9987,
      "created_at": "2026-03-03T10:00:00Z"
    }
  ]
}
```

- **Errors**:
  - If Supabase is not configured or querying fails, the endpoint returns a `503` or `500` with an error message.

---

### Model Details

- **Library**: `transformers` (Hugging Face)
- **Pipeline**: `"text-classification"`
- **Model**: `"distilbert-base-uncased-finetuned-sst-2-english"`
- **Output**:
  - `label`: sentiment label (e.g., `POSITIVE`, `NEGATIVE`)
  - `score`: confidence score (rounded to 4 decimal places)

The model is loaded at import time in `app/services/core.py`. If loading fails, the classifier is set to `None` and prediction calls will raise a `RuntimeError`, which is translated to a `503` response by the API.

---

### Notes and Future Work

- **Docker**: The `docker/` folder is present but the Dockerfile and `docker-compose.yml` are currently empty and can be filled in later for containerized deployment.
- **Frontend**: CORS is configured for `http://localhost:3000`, anticipating a frontend client (e.g., React) to consume this API.
- **Extensions**:
  - Add authentication and rate limiting.
  - Support more classification labels or different models.
  - Improve logging and monitoring (e.g., structured logs, metrics).


import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to find model_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_loader import ModelLoaderFactory  # Import the ModelLoaderFactory

app = FastAPI()
# Use an explicit path for the templates directory
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Initialize the model loader factory
factory = ModelLoaderFactory()
available_models = factory.get_available_models()

# Cache for loaded models and pipelines
model_cache = {}

# Simple test route to confirm the server is responding
@app.get("/test")
async def test():
    return {"message": "Server is running!"}

# Function to process input text through the selected model for summarization
def process_text(text: str, model_name: str, max_length: int = 128, min_length: int = 30) -> dict:
    try:
        # Load model and tokenizer from cache or initialize
        if model_name not in model_cache:
            loader = factory.get_loader(model_name)
            model, tokenizer = loader.load()
            # Create a summarization pipeline
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
            model_cache[model_name] = summarizer
        else:
            summarizer = model_cache[model_name]

        # Prepend "summarize: " for T5 models
        if "t5" in model_name.lower():
            text = f"summarize: {text}"
        logger.info(f"Model {model_name} - Input text: {text}")

        # Generate summary
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        summary_text = summary[0]["summary_text"]
        logger.info(f"Model {model_name} - Generated summary: {summary_text}")

        # Free up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"summary": summary_text}
    except Exception as e:
        logger.error(f"Error in process_text: {str(e)}")
        return {"error": f"Failed to process text: {str(e)}"}

# Route for the main page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "models": available_models, "summary": "", "input_text": "", "selected_model": ""}
    )

@app.post("/", response_class=HTMLResponse)
async def process_form(
    request: Request,
    text: str = Form(...),
    model: str = Form(...)
):
    if not text.strip():
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "models": available_models,
                "error": "Input text cannot be empty or whitespace.",
                "summary": "",
                "input_text": "",
                "selected_model": model
            }
        )

    # Process the text with the selected model
    result = process_text(text, model)

    if "error" in result:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "models": available_models,
                "error": result["error"],
                "summary": "",
                "input_text": text,
                "selected_model": model
            }
        )

    summary = result.get("summary", "")
    if not summary:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "models": available_models,
                "error": "No summary generated from the input text.",
                "summary": "",
                "input_text": text,
                "selected_model": model
            }
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": available_models,
            "summary": summary,
            "input_text": text,
            "selected_model": model
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
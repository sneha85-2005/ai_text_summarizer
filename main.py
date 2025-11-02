# main.py
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from transformers import pipeline
import re

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# Load summarizer model (may take a few minutes the first time)
# If memory is an issue, use a smaller model like "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def split_into_chunks(text: str, max_chars: int = 1200):
    """
    Split long text into chunks of ~max_chars, trying to split on sentence boundaries.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            # if single sentence is larger than max_chars, force-split
            if len(s) > max_chars:
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i:i+max_chars])
                current = ""
            else:
                current = s
    if current:
        chunks.append(current)
    return chunks

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/summarize", response_class=PlainTextResponse)
async def summarize(text: str = Form(...), format: str = Form("paragraph")):
    text = (text or "").strip()
    if not text:
        return PlainTextResponse("Please provide text to summarize.", status_code=400)

    words = len(text.split())

    if words < 40:
        # Short text: direct summarization
        min_len = max(10, words // 2)
        max_len = max(20, words + 10)
        try:
            out = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
            summary = out[0]["summary_text"].strip()
        except Exception as e:
            return PlainTextResponse(f"Error generating summary: {str(e)}", status_code=500)
    else:
        # Long text: chunk, summarize each, then combine
        chunks = split_into_chunks(text, max_chars=1200)
        partial_summaries = []
        try:
            for chunk in chunks:
                out = summarizer(
                    chunk,
                    max_length=200,
                    min_length=60,
                    do_sample=True,
                    temperature=0.8
                )
                partial_summaries.append(out[0]["summary_text"].strip())
        except Exception as e:
            return PlainTextResponse(f"Error summarizing chunk: {str(e)}", status_code=500)

        # Combine partial summaries and polish
        combined = " ".join(partial_summaries)
        min_len = max(30, len(combined.split()) // 6)
        max_len = max(80, len(combined.split()) // 3 + 40)
        try:
            final_out = summarizer(
                combined,
                max_length=250,
                min_length=80,
                do_sample=True,
                temperature=0.8
            )
            summary = final_out[0]["summary_text"].strip()
        except Exception as e:
            return PlainTextResponse(f"Error in final summarization: {str(e)}", status_code=500)

    # Format as bullets if requested
    if format == "bullets":
        sentences = re.split(r'(?<=[.!?]) +', summary)
        bullets = [s.strip() for s in sentences if s.strip()]
        summary = "\n".join("• " + s for s in bullets)

    return PlainTextResponse(summary)

    












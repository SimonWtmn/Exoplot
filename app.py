"""
Exoplot — FastAPI server
------------------------
HTTP entry point: static files (CSS / JS / images), landing page,
analysis single-page app, and JSON API routes living under
``routers/analysis.py``.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routers import analysis as analysis_router

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Exoplot",
    description="Visual exploration of exoplanet data",
    version="0.2.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Expose generated reports (DVR PDFs) as static assets so the browser
# can download them directly once the pipeline finishes.
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Landing page with the ``Launch Lightcurve Analysis`` entry."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"active": "home"},
    )


@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request) -> HTMLResponse:
    """Single-page analysis workspace (search → pipeline → results)."""
    return templates.TemplateResponse(
        request=request,
        name="analysis.html",
        context={"active": "analysis"},
    )


app.include_router(analysis_router.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

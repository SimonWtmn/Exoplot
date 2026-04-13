"""
Exoplot — serveur web FastAPI
-----------------------------
Point d'entrée HTTP : fichiers statiques (CSS/JS/images) et page d'accueil.
Les traitements scientifiques restent dans le package `modules` (catalogue,
courbes de lumière, MCMC, etc.) et seront branchés sur des routes API au fil
du développement.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(
    title="Exoplot",
    description="Visual exploration of exoplanet data",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Page d'accueil (landing)."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"active": "home"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
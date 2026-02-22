"""FastAPI application entry point."""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import load_config, get_models_ini_path

logger = logging.getLogger(__name__)

app = FastAPI(title="LlamaModel,", description="Manage GGUF models for llama.cpp")

_config: dict | None = None


def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
        logger.info(
            "Config loaded: port=%s models_dir=%s",
            _config["port"],
            _config["models_dir"],
        )
    return _config


# Templates and static files
ROOT = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))
_static_dir = ROOT / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
    logger.debug("Static files mounted from %s", _static_dir)


@app.get("/", response_class=RedirectResponse)
async def index(request: Request):
    """Redirect to My Models page natively."""
    logger.debug("GET / -> Redirecting to /models")
    return RedirectResponse(url="/models", status_code=303)


# Routes will be registered by including routers
def setup_routes():
    from app.routes import discover, models_ini, settings, api
    app.include_router(discover.router, tags=["discover"])
    app.include_router(models_ini.router, prefix="/models", tags=["models"])
    app.include_router(settings.router, prefix="/settings", tags=["settings"])
    app.include_router(api.router, prefix="/api", tags=["api"])
    logger.debug("All routers registered")


setup_routes()

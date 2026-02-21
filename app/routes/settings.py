"""Settings: show and edit port and models dir."""

import logging

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from app.main import templates, get_config
from app.config import save_config, load_config
import app.main as _app_main

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_class=HTMLResponse)
async def settings_page(request: Request, saved: str | None = None):
    """Show application settings (port, models directory)."""
    logger.debug("GET /settings saved=%r", saved)
    config = get_config()
    return templates.TemplateResponse(
        "settings.html",
        {"request": request, "config": config, "saved": saved == "1"},
    )


@router.post("", response_class=HTMLResponse)
async def save_settings(
    request: Request,
    port: int = Form(...),
    models_dir: str = Form(...),
):
    """Save port and models_dir to config.yaml and reload in-memory config."""
    logger.info("POST /settings: saving port=%s models_dir=%s", port, models_dir)
    save_config(port=port, models_dir=models_dir)
    # Reload the in-memory singleton so the running app picks up the new values
    # (env-var overrides still take precedence at runtime)
    _app_main._config = load_config()
    logger.info("POST /settings: in-memory config reloaded")
    return RedirectResponse(url="/settings?saved=1", status_code=303)

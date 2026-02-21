"""My Models: list and edit models.ini sections."""

import logging

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from app.main import templates, get_config
from app.config import get_models_ini_path
from app.services import ini_manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _ini_path():
    config = get_config()
    return get_models_ini_path(config["models_dir"])


@router.get("", response_class=HTMLResponse)
async def my_models_page(request: Request):
    """List models defined in models.ini."""
    logger.debug("GET /models")
    config = get_config()
    path = _ini_path()
    sections = ini_manager.list_sections(path)
    logger.debug("GET /models: %d sections", len(sections))
    return templates.TemplateResponse(
        "my_models.html",
        {"request": request, "sections": sections, "config": config},
    )


@router.get("/edit/{section_name}", response_class=HTMLResponse)
async def edit_model_page(request: Request, section_name: str):
    """Edit one model section."""
    logger.debug("GET /models/edit/%s", section_name)
    config = get_config()
    path = _ini_path()
    params = ini_manager.get_section(path, section_name)
    if params is None:
        logger.warning("GET /models/edit/%s: section not found", section_name)
        raise HTTPException(status_code=404, detail="Model not found")
    return templates.TemplateResponse(
        "model_edit.html",
        {"request": request, "section_name": section_name, "params": params, "config": config},
    )


@router.post("/edit/{section_name}")
async def save_model(section_name: str, request: Request):
    """Save model section from form. Form keys: param_<key> = value; optional new_param_key, new_param_value."""
    logger.debug("POST /models/edit/%s", section_name)
    path = _ini_path()
    form = await request.form()
    params = {}
    for k, v in form.items():
        if k.startswith("param_") and v is not None:
            arg_key = k[6:].strip()
            if arg_key:
                params[arg_key] = str(v).strip()
    new_key = (form.get("new_param_key") or "").strip()
    new_val = (form.get("new_param_value") or "").strip()
    if new_key:
        params[new_key] = new_val
    ini_manager.set_section(path, section_name, params)
    logger.info("POST /models/edit/%s: saved %d params", section_name, len(params))
    return RedirectResponse(url="/models", status_code=303)


@router.get("/delete/{section_name}")
async def delete_model(section_name: str):
    """Remove model section."""
    logger.debug("GET /models/delete/%s", section_name)
    path = _ini_path()
    ok = ini_manager.delete_section(path, section_name)
    if not ok:
        logger.warning("GET /models/delete/%s: section not found", section_name)
        raise HTTPException(status_code=404, detail="Model not found")
    logger.info("GET /models/delete/%s: section deleted", section_name)
    return RedirectResponse(url="/models", status_code=303)

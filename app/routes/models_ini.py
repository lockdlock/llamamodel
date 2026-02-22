"""My Models: list and edit models.ini sections."""

import logging
from pathlib import Path

from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from app.main import templates, get_config
from app.config import get_models_ini_path
from app.services import ini_manager
import os

logger = logging.getLogger(__name__)

router = APIRouter()


def _ini_path():
    config = get_config()
    return get_models_ini_path(config["models_dir"])


@router.get("", response_class=HTMLResponse)
async def my_models_page(request: Request):
    """List models defined in models.ini and dynamically scan for unconfigured .gguf files."""
    logger.debug("GET /models")
    config = get_config()
    path = _ini_path()
    
    # Check if there are top level generic parameters
    ini_parser = ini_manager.read_ini(path)
    # The general parameters should be under the [*] section explicitly now
    general_params = {}
    if ini_parser.has_section("*"):
         general_params = dict(ini_parser["*"])
    
    # We remove old logic keys if present because they're managed internally
    if "LLAMA_CONFIG_VERSION" in general_params:
         del general_params["LLAMA_CONFIG_VERSION"]
    if "version" in general_params:
         del general_params["version"]
         
    sections = ini_manager.list_sections(path)
    
    # Do not include '*' in the general sections list since it powers General Parameters explicitly
    sections = [s for s in sections if s["name"] != "*"]
    
    models_dir = Path(config["models_dir"])
    configured_paths = set()
    
    # We will enrich sections to have file size, card name, quant, file name
    from app.services.hf_service import _extract_quantization
    import os
    
    enriched_sections = []
    
    for s in sections:
        model_path_str = s["params"].get("model") or s["params"].get("LLAMA_ARG_MODEL")
        if model_path_str:
            # Strip enclosing single quotes if they exist before checking resolution
            if model_path_str.startswith("'") and model_path_str.endswith("'"):
                model_path_str = model_path_str[1:-1]
                
            resolved = Path(model_path_str).resolve()
            configured_paths.add(str(resolved))
            file_name = resolved.name
            size_mb = 0
            if resolved.exists():
                size_mb = os.path.getsize(resolved) / (1024 * 1024)
            size_str = f"{size_mb / 1024:.2f} GB" if size_mb > 1024 else f"{size_mb:.0f} MB"
            
            quant = _extract_quantization(file_name)
            # if we successfully extracted quant, remove it from the name to get card name
            card_name = file_name
            if quant:
                # Find the quant string and remove it (and preceding '-' or '.')
                import re
                esc_quant = re.escape(quant)
                # match dash or dot, then the quant (case insensitive), then .gguf
                # example: -Q4_K_M.gguf or .q4_k_m.gguf
                m = re.search(r"[-.]" + esc_quant + r"(?:\.gguf)?", file_name, flags=re.IGNORECASE)
                if m:
                    card_name = file_name[:m.start()]
                else:
                    card_name = file_name.replace(".gguf", "")
            else:
                card_name = file_name.replace(".gguf", "")
        else:
            file_name = "Unknown"
            size_str = "0 MB"
            card_name = s.name
            quant = "Unknown"
            
        s["display"] = {
            "card_name": card_name,
            "quantization": quant,
            "file_name": file_name,
            "file_size": size_str,
        }
        enriched_sections.append(s)

    unconfigured_files = []
    if models_dir.exists():
        for gf in models_dir.rglob("*.gguf"):
            rp = str(gf.resolve())
            if rp not in configured_paths:
                unconfigured_files.append({
                    "name": gf.name,
                    "path": rp,
                })
    unconfigured_files.sort(key=lambda x: x["name"])

    return templates.TemplateResponse(
        "my_models.html",
        {
            "request": request,
            "sections": enriched_sections,
            "general_params": general_params,
            "unconfigured_files": unconfigured_files,
            "config": config
        },
    )


@router.post("/add_local")
async def add_local_model(request: Request, path: str = Form(...)):
    """Add a local unconfigured GGUF to models.ini."""
    logger.debug("POST /models/add_local: %s", path)
    gguf_path = Path(path)
    if not gguf_path.exists():
        raise HTTPException(status_code=400, detail="File not found")
    
    # Use exact same parsing structure as download
    from app.services.hf_service import _extract_quantization
    import re
    
    filename = gguf_path.name
    quant = _extract_quantization(filename)
    if quant:
        esc_quant = re.escape(quant)
        m = re.search(r"[-.]" + esc_quant + r"(?:\.gguf)?", filename, flags=re.IGNORECASE)
        if m:
            card_name = filename[:m.start()]
        else:
            card_name = filename.replace(".gguf", "")
    else:
        card_name = filename.replace(".gguf", "")
        quant = "unknown"
        
    section_name = f"local/{card_name}:{quant}"
    
    ini_path = _ini_path()
    # The path to model file should be saved between single quotes in the models.ini
    params = {"model": f"'{str(gguf_path.resolve())}'"}
    param_descriptions = {"model": "model path to load"}
    ini_manager.add_or_update_section(ini_path, section_name, params, merge=True, param_descriptions=param_descriptions)
    return RedirectResponse(url="/models", status_code=303)


@router.get("/edit/{section_name:path}", response_class=HTMLResponse)
async def edit_model_page(request: Request, section_name: str):
    """Edit one model section."""
    logger.debug("GET /models/edit/%s", section_name)
    config = get_config()
    path = _ini_path()
    
    if section_name == "GENERAL_PARAMS":
        # special case to edit top level config
        ini_parser = ini_manager.read_ini(path)
        params = {}
        if ini_parser.has_section("*"):
             params = dict(ini_parser["*"])
        if "LLAMA_CONFIG_VERSION" in params:
             del params["LLAMA_CONFIG_VERSION"]
        if "version" in params:
             del params["version"]
    else:
        params = ini_manager.get_section(path, section_name)
        if params is None:
            logger.warning("GET /models/edit/%s: section not found", section_name)
            raise HTTPException(status_code=404, detail="Model not found")
            
    return templates.TemplateResponse(
        "model_edit.html",
        {"request": request, "section_name": section_name, "params": params, "config": config},
    )


@router.post("/edit/{section_name:path}")
async def save_model(section_name: str, request: Request, inline: int = 0):
    """Save model section from form. Form keys: param_<key> = value; optional new_param_key, new_param_value."""
    logger.debug("POST /models/edit/%s", section_name)
    path = _ini_path()
    form = await request.form()
    params = {}
    param_descriptions = {}
    for k, v in form.items():
        if k.startswith("param_") and v is not None:
            arg_key = k[6:].strip()
            if arg_key:
                params[arg_key] = str(v).strip()
                # Try to find corresponding description if it was submitted
                desc_key = f"desc_{arg_key}"
                if desc_key in form:
                    param_descriptions[arg_key] = form[desc_key]
    new_key = (form.get("new_param_key") or "").strip()
    new_val = (form.get("new_param_value") or "").strip()
    if new_key:
        params[new_key] = new_val
        
    if section_name == "GENERAL_PARAMS":
        # Write general configs cleanly exclusively to the [*] section standard
        
        # Build parameter descriptions
        for k in params.keys():
            if k not in param_descriptions:
                 param_descriptions[k] = "General parameter" 
             
        ini_manager.set_section(path, "*", params, param_descriptions)
        logger.info("POST /models/edit/%s: saved general config %d params", section_name, len(params))
        
        # Determine if this was an inline save or an edit page save
        if inline or "/models/edit/" not in request.headers.get("referer", ""):
            # It was saved from the inline form on the main models page
            return RedirectResponse(url="/models", status_code=303)
            
        return RedirectResponse(url="/models", status_code=303)

    # Standard model update
    for k in params.keys():
        if k not in param_descriptions:
             param_descriptions[k] = "Model-specific parameter"
         
    ini_manager.set_section(path, section_name, params, param_descriptions)
    logger.info("POST /models/edit/%s: saved %d params", section_name, len(params))
    
    # Check where the request came from so we redirect cleanly
    if inline or "/models/edit/" not in request.headers.get("referer", ""):
         # Came from the inline form, redirect straight back to /models page
         return RedirectResponse(url="/models", status_code=303)
         
    return RedirectResponse(url="/models", status_code=303)


@router.get("/delete/{section_name:path}")
async def delete_model(section_name: str):
    """Remove model section and associated file.
    Note: We map {section_name:path} to handle forward slashes naturally for our author/model convention."""
    logger.debug("GET /models/delete/%s", section_name)
    path = _ini_path()
    
    # Check if there is a file associated
    params = ini_manager.get_section(path, section_name)
    file_deleted = False
    
    if params and ("model" in params or "LLAMA_ARG_MODEL" in params):
        model_path_str = params.get("model") or params.get("LLAMA_ARG_MODEL")
        if model_path_str.startswith("'") and model_path_str.endswith("'"):
            model_path_str = model_path_str[1:-1]
            
        file_route = Path(model_path_str).resolve()
        if file_route.exists() and file_route.is_file():
            try:
                os.remove(file_route)
                logger.info("GET /models/delete/%s: deleted file %s", section_name, file_route)
                file_deleted = True
            except Exception as e:
                logger.error("Failed to delete file %s: %s", file_route, e)
    
    ok = ini_manager.delete_section(path, section_name)
    if not ok:
        logger.warning("GET /models/delete/%s: section not found", section_name)
        raise HTTPException(status_code=404, detail="Model not found")
    logger.info("GET /models/delete/%s: section deleted", section_name)
    return RedirectResponse(url="/models", status_code=303)

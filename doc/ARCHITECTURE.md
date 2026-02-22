# Architecture: LLM Manager

## Overview
The LLM Manager is a Python-based web application providing an LMStudio-like interface designed to manage GGUF models for `llama.cpp`. It interacts with the Hugging Face Hub to discover, inspect, and download models, while maintaining a `models.ini` configuration file that defines loading parameters for the `llama.cpp` server.

## Graphical Architecture Block Diagram

```mermaid
flowchart LR
    subgraph Client [Client / Browser]
        UI[Web UI (HTML + JS)]
        CSS[styles.css]
    end

    subgraph Server [Backend / FastAPI]
        Router[API & Page Routers]
        Config[App Config (config.py)]
        Params[Params Parser]
        
        subgraph Services [Core Services]
            HF[HF Service (hf_service.py)]
            INI[INI Manager (ini_manager.py)]
        end
    end

    subgraph Data [Filesystem]
        CfgFile[(config.yaml)]
        ModelsDir[(models_dir)]
        IniFile[(models.ini)]
        GitCache[(HF Hub Cache)]
    end
    
    subgraph External [External Services]
        HuggingFace[Hugging Face API]
    end

    UI --> Router
    Router --> HF
    Router --> INI
    Router --> Config
    Router --> Params
    
    HF --> HuggingFace
    HF --> GitCache
    INI --> IniFile
    Config --> CfgFile
    
    GitCache -. inside .- ModelsDir
    IniFile -. inside .- ModelsDir
```

## Directory & File Structure

- **`app/main.py`**: The FastAPI application entrypoint. Configures Jinja2 templates, static files, and mounts the various API/page routers.
- **`app/config.py`**: Handles configuration logic, resolving defaults, `config.yaml`, and environment variables (`LLAMAMODEL_PORT`, `LLAMAMODEL_MODELS_DIR`).
- **`app/routes/`**: Contains sub-routers.
  - `api.py`: Search backend, model details, file downloads, job statuses, and APIs for local `models.ini` reads.
  - `discover.py`: Renders the `/discover` frontend page.
  - `models_ini.py`: Renders pages for managing locally downloaded models.
  - `settings.py`: Renders and accepts POST edits for the app settings.
- **`app/services/`**: Core business logic modules.
  - `hf_service.py`: Wraps `huggingface_hub` for querying models, reading model cards, enumerating quantizations, parsing capability metrics, and handling background downloads.
  - `ini_manager.py`: Uses `configparser` and OS-level file locking to read/write `models.ini` with compatibility for `llama.cpp`.
  - `params_parser.py`: Extracts model constraints (like `n_ctx`) out of Hugging Face model cards.
- **`static/`**: Static JS (`app.js`) and CSS (`style.css`), driving frontend dynamic behaviors like search-as-you-type, sorting, filtering, and toasts (flash messages).
- **`templates/`**: Jinja2 templates standardizing the UI.
- **`doc/`**: Documentation including `PLAN.md` and this `ARCHITECTURE.md`.

## Data Flows

1. **Discovery & Search**: User actions (typing a query, clicking a tag) route to `/api/search` via AJAX. The backend accesses Hugging Face using `HfApi()`, caching filters logically and mapping visual characteristics (vision, thinking, tools calling) back to UI-friendly payloads.
2. **Model Download**: Triggers a background task calling `hf_hub_download` asynchronously. An in-memory status dictionary tracking the job gets polled by the client for updates.
3. **models.ini Management**: On a successful download, the parameter parser evaluates the model card text and dynamically structures an INI configuration block ensuring the `LLAMA_ARG_MODEL` maps precisely to the newly grabbed disk file. Users optionally overwrite these mapped defaults in the "My Models" tab.

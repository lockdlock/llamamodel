"""Hugging Face Hub integration: search GGUF models, list files, model card, download."""

import time as _time
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, ModelCard

logger = logging.getLogger(__name__)

# Cache for list_repo_files (quantizations) to avoid repeated API calls
_repo_files_cache: dict[str, tuple[float, list[str]]] = {}
# Cache for repo file info (with sizes): repo_id -> (timestamp, {filename: size_bytes})
_repo_file_info_cache: dict[str, tuple[float, dict[str, int]]] = {}
_CACHE_TTL_SEC = 300  # 5 minutes

# Retry / backoff settings for HF API calls
_MAX_RETRIES = 4
_BACKOFF_BASE_SEC = 2.0  # seconds; delay = base * 2^attempt


def _get_api() -> HfApi:
    return HfApi()


def _with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on rate-limit / transient errors.
    Retries up to _MAX_RETRIES times on HTTP 429 or 5xx-style exceptions.
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            # Retry on rate-limit (429) or server errors (5xx)
            is_retryable = (
                "429" in msg
                or "rate limit" in msg
                or "too many requests" in msg
                or "503" in msg
                or "502" in msg
                or "500" in msg
                or "connection" in msg
            )
            if not is_retryable or attempt == _MAX_RETRIES:
                raise
            delay = _BACKOFF_BASE_SEC * (2 ** attempt)
            logger.warning(
                "HF API call failed (attempt %d/%d): %s. Retrying in %.1fs…",
                attempt + 1, _MAX_RETRIES, exc, delay,
            )
            last_exc = exc
            _time.sleep(delay)
    raise last_exc  # unreachable but satisfies type checkers


def _infer_capabilities_from_text(text: str, model_id: str) -> dict[str, bool]:
    """Infer vision, tools, thinking from model id and/or card text."""
    s = (text or "").lower() + " " + (model_id or "").lower()
    return {
        "vision": "vision" in s or "visual" in s or "vlm" in s or "multimodal" in s or "image-text" in s,
        "tools": (
            "tool" in s or "function call" in s or "function_call" in s or "function-call" in s
            or "tool-calling" in s or "tool_calling" in s or "tool-use" in s or "tool_use" in s
            or "tool use" in s or "agent" in s
        ),
        "thinking": "thinking" in s or "reasoning" in s or "deepseek" in s or "r1" in s,
    }


def _infer_capabilities_from_tags(tags: list[str] | None) -> dict[str, bool]:
    """Infer vision, tools, thinking from Hugging Face model tags."""
    if not tags:
        return {"vision": False, "tools": False, "thinking": False}
    t = " ".join(t.lower() for t in tags)
    return {
        "vision": "vision" in t or "multimodal" in t or "image-text" in t or "vlm" in t,
        "tools": (
            "tool" in t or "function-call" in t or "function-calling" in t
            or "tool-calling" in t or "tool_calling" in t or "tool-use" in t or "tool_use" in t
            or "tool use" in t or "agent" in t
        ),
        "thinking": "thinking" in t or "reasoning" in t or "reasoner" in t,
    }


def _repo_name(model_id: str) -> str:
    """Return only the repository name (last segment of model_id)."""
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _parse_size_from_text(text: str) -> str:
    """
    Parse parameter count / model size from a string (repo name, tag, or description).
    Recognises patterns like: 7B, 0.5B, 70B, 1.5B, 7b, 7-B, 7_B, 7.0B, 7.0-B
    Also handles 'x' multiplier patterns like 8x7B (MoE).
    Returns a normalised string like '7B', '0.5B', '8x7B', or '' if not found.
    """
    import re
    # MoE pattern: 8x7B, 4x22B etc.
    m = re.search(r"(\d+)[xX](\d+\.?\d*)\s*[Bb]", text)
    if m:
        return f"{m.group(1)}x{m.group(2)}B"
    # Standard: 7B, 0.5B, 70B, 1.5B – must be followed by word boundary / separator
    m = re.search(r"(?<![.\d])(\d+\.?\d*)\s*[Bb](?:[_\-\s]|$)", text)
    if m:
        val = m.group(1)
        # Normalise: drop trailing .0
        if val.endswith(".0"):
            val = val[:-2]
        return val + "B"
    return ""


def _parse_size_from_tags(tags: list[str] | None) -> str:
    """
    Extract parameter count from Hugging Face model tags.
    Tags like '7b', '13b', '70b', '0.5b', '8x7b' are common.
    Returns normalised string like '7B' or '' if not found.
    """
    if not tags:
        return ""
    for tag in tags:
        # Tags are often just the size, e.g. "7b", "13b", "0.5b"
        result = _parse_size_from_text(tag)
        if result:
            return result
    return ""


def _parse_size_from_repo(repo_name: str) -> str:
    """Parse model size from repo name (e.g. 7B, 0.5B, 70B, 1.5B). Returns empty string if not found."""
    return _parse_size_from_text(repo_name)


def _extract_quantization(filename: str) -> str:
    """Extract quantization tag from GGUF filename. Formats: Q(n)_, FP(n), F(n), IQ(n)_, BF(n)."""
    import re
    base = filename.removesuffix(".gguf").removesuffix(".GGUF")
    parts = base.split("-")
    for p in reversed(parts):
        p = p.strip()
        if not p:
            continue
        if re.match(r"^Q\d", p, re.IGNORECASE):
            return p.upper()
        if re.match(r"^FP\d+$", p, re.IGNORECASE):
            return p.upper()
        if re.match(r"^F\d+$", p, re.IGNORECASE):
            return p.upper()
        if re.match(r"^IQ\d", p, re.IGNORECASE):
            return p.upper()
        if re.match(r"^BF\d+$", p, re.IGNORECASE):
            return p.upper()
    return ""


def _format_param_count(n: int) -> str:
    """Format a raw parameter count integer into a human-readable string like '7B', '405M'."""
    if n >= 1_000_000_000:
        val = n / 1_000_000_000
        s = f"{val:.1f}".rstrip("0").rstrip(".")
        return s + "B"
    if n >= 1_000_000:
        val = n / 1_000_000
        s = f"{val:.0f}"
        return s + "M"
    return str(n)


def _format_file_size(n: int) -> str:
    """Format bytes into human-readable string: GB, MB, KB."""
    if n >= 1_073_741_824:  # 1 GiB
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:  # 1 MiB
        return f"{n / 1_048_576:.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def get_repo_file_sizes(repo_id: str, use_cache: bool = True) -> dict[str, int]:
    """
    Return a dict of {filename: size_bytes} for all files in the repo.
    Cached for CACHE_TTL_SEC. Returns empty dict on failure.
    """
    now = _time.time()
    if use_cache and repo_id in _repo_file_info_cache:
        ts, info = _repo_file_info_cache[repo_id]
        if now - ts < _CACHE_TTL_SEC:
            return info
    api = _get_api()
    try:
        file_infos = list(_with_retry(api.list_repo_tree, repo_id, recursive=True))
        sizes: dict[str, int] = {}
        for fi in file_infos:
            # RepoFile objects have .path and .size; RepoFolder objects have no .size
            path = getattr(fi, "path", None)
            size = getattr(fi, "size", None)
            if path and size is not None:
                sizes[path] = int(size)
        _repo_file_info_cache[repo_id] = (now, sizes)
        return sizes
    except Exception as exc:
        logger.debug("get_repo_file_sizes(%s) failed: %s", repo_id, exc)
        return {}


def group_gguf_by_quantization(
    gguf_files: list[str],
    file_sizes: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Group GGUF files by quantization. Multifile models have -00001-of-00003 etc.
    Returns list of {quant, primary_file, all_files, total_size_bytes, total_size_display}.
    Primary is the single file or the first shard.
    file_sizes: optional dict of {filename: bytes} from get_repo_file_sizes().
    """
    by_quant: dict[str, list[str]] = {}
    for f in gguf_files:
        q = _extract_quantization(f)
        if not q:
            q = f.removesuffix(".gguf").removesuffix(".GGUF").split("-")[-1].upper() or f
        by_quant.setdefault(q, []).append(f)
    result = []
    for quant, files in by_quant.items():
        files = sorted(files)
        primary = None
        for f in files:
            if "-00001-of-" in f or "-0001-of-" in f:
                primary = f
                break
        if primary is None:
            primary = files[0]
        total_bytes: int | None = None
        if file_sizes:
            total = sum(file_sizes.get(f, 0) for f in files)
            if total > 0:
                total_bytes = total
        result.append({
            "quant": quant,
            "primary_file": primary,
            "all_files": files,
            "total_size_bytes": total_bytes,
            "total_size_display": _format_file_size(total_bytes) if total_bytes else "",
        })
    return result


def search_models(
    query: str | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "downloads",
) -> list[dict[str, Any]]:
    """Search for GGUF models. Returns list of {id, repo_name, author, downloads, capabilities, ...}."""
    api = _get_api()
    full_limit = offset + limit

    def _do_list_models(full: bool):
        return list(api.list_models(
            filter="gguf",
            search=query or "",
            sort=sort,
            limit=min(full_limit, 100),
            **({"full": True} if full else {}),
        ))

    try:
        raw = _with_retry(_do_list_models, True)
    except TypeError:
        raw = _with_retry(_do_list_models, False)

    items = []
    for i, m in enumerate(raw):
        if i < offset:
            continue
        if len(items) >= limit:
            break
        tags = getattr(m, "tags", None) or []
        if isinstance(tags, (list, tuple)):
            tags_list = list(tags)
            caps_from_tags = _infer_capabilities_from_tags(tags_list)
        else:
            tags_list = []
            caps_from_tags = {"vision": False, "tools": False, "thinking": False}
        caps_from_id = _infer_capabilities_from_text("", m.id)
        caps = {
            "vision": caps_from_tags["vision"] or caps_from_id["vision"],
            "tools": caps_from_tags["tools"] or caps_from_id["tools"],
            "thinking": caps_from_tags["thinking"] or caps_from_id["thinking"],
        }
        repo_name = _repo_name(m.id)

        # Parameter count: try tags first, then repo name, then safetensors metadata
        size_display = _parse_size_from_tags(tags_list)
        if not size_display:
            size_display = _parse_size_from_repo(repo_name)
        if not size_display:
            # Try safetensors total parameter count if available
            try:
                st = getattr(m, "safetensors", None)
                if st and hasattr(st, "total"):
                    total = st.total
                    if total:
                        size_display = _format_param_count(int(total))
            except Exception:
                pass
        if not size_display:
            # Last resort: search the full model id string
            size_display = _parse_size_from_text(m.id)

        items.append({
            "id": m.id,
            "repo_name": repo_name,
            "size_display": size_display,
            "author": getattr(m, "author", "") or (m.id.split("/")[0] if "/" in m.id else ""),
            "downloads": getattr(m, "downloads", None) or 0,
            "likes": getattr(m, "likes", None) or 0,
            "vision": caps["vision"],
            "tools": caps["tools"],
            "thinking": caps["thinking"],
        })
    return items


def list_gguf_files(repo_id: str, use_cache: bool = True) -> list[str]:
    """List .gguf filenames in the repo. Cached for CACHE_TTL_SEC."""
    now = _time.time()
    if use_cache and repo_id in _repo_files_cache:
        ts, files = _repo_files_cache[repo_id]
        if now - ts < _CACHE_TTL_SEC:
            return files
    api = _get_api()
    all_files = list(_with_retry(api.list_repo_files, repo_id))
    gguf_files = [f for f in all_files if f.endswith(".gguf")]
    _repo_files_cache[repo_id] = (now, gguf_files)
    return gguf_files


def get_model_card_content(repo_id: str) -> str:
    """Fetch model card markdown content. Returns empty string on failure."""
    try:
        card = _with_retry(ModelCard.load, repo_id)
        return card.content or ""
    except Exception:
        return ""


def get_model_capabilities(repo_id: str) -> dict[str, bool]:
    """Infer vision, tools, thinking from model card content and repo id."""
    content = get_model_card_content(repo_id)
    return _infer_capabilities_from_text(content, repo_id)


def download_model(
    repo_id: str,
    filename: str,
    models_dir: Path | str,
    local_dir_override: Path | str | None = None,
) -> Path:
    """
    Download a GGUF file into the models directory.
    If local_dir_override is set, download there; else use HF_HOME so file lands under models_dir/hub/.
    Returns path to the downloaded file.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    if local_dir_override is not None:
        local_dir = Path(local_dir_override)
        local_dir.mkdir(parents=True, exist_ok=True)
        path = _with_retry(
            hf_hub_download,
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        return Path(path)
    # Use HF_HOME so cache goes under models_dir.
    # HF_HOME must be set in the calling process environment before calling hf_hub_download;
    # the caller (api.py run_download) sets os.environ["HF_HOME"] before calling this function.
    path = _with_retry(
        hf_hub_download,
        repo_id=repo_id,
        filename=filename,
        local_dir=None,
        token=None,
    )
    # hf_hub_download with HF_HOME uses cache under HF_HOME/hub/; path returned is the resolved path
    return Path(path)

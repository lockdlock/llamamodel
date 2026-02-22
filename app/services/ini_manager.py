"""Read/write models.ini compatible with llama.cpp server."""

import fcntl
import logging
from configparser import ConfigParser
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LLAMA_CONFIG_VERSION = "1"
LLAMA_ARG_PREFIX = "LLAMA_ARG_"


def _ensure_models_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _lock_file(f, exclusive: bool = True) -> None:
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
    except (OSError, AttributeError):
        pass  # Windows or unsupported


def _unlock_file(f) -> None:
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (OSError, AttributeError):
        pass


def read_ini(ini_path: Path) -> ConfigParser:
    """Read models.ini; create minimal one if missing. Expects LLAMA_CONFIG_VERSION at top or [section] blocks."""
    parser = ConfigParser()
    parser.optionxform = str  # preserve key case (LLAMA_ARG_* must stay uppercase for llama.cpp)
    if ini_path.exists():
        with open(ini_path) as f:
            _lock_file(f, exclusive=False)
            try:
                content = f.read()
            finally:
                _unlock_file(f)
        # ConfigParser requires sections; prepend a dummy section if file starts with key=value
        if content.strip() and not content.strip().startswith("["):
            content = "[__top__]\n" + content
        parser.read_string(content)
        if parser.has_section("__top__"):
            parser.remove_section("__top__")
        logger.debug("Read models.ini from %s (%d sections)", ini_path, len(parser.sections()))
    else:
        logger.debug("models.ini not found at %s; returning empty parser", ini_path)
    return parser


def write_ini(ini_path: Path, parser: ConfigParser, param_descriptions: dict[str, str] = None) -> None:
    """Write models.ini with file lock. Writes version at top then model sections. Adds descriptions."""
    _ensure_models_dir(ini_path)
    sections = [s for s in parser.sections() if s != "DEFAULT"]
    
    # Optional dictionary to map config keys to their descriptions
    if param_descriptions is None:
        param_descriptions = {}
        
    with open(ini_path, "w") as f:
        _lock_file(f, exclusive=True)
        try:
            f.write(f"version = {LLAMA_CONFIG_VERSION}\n\n")
            
            # Write general parameters first explicitly if present
            if "*" in sections:
                f.write("[*]\n")
                for k, v in parser["*"].items():
                    desc = param_descriptions.get(k)
                    if desc:
                        f.write(f"; {desc}\n")
                    f.write(f"{k} = {v}\n")
                f.write("\n")
                
            # Write all other sections
            for section in sections:
                if section == "*":
                    continue
                f.write(f"[{section}]\n")
                for k, v in parser[section].items():
                    desc = param_descriptions.get(k)
                    if desc:
                        f.write(f"; {desc}\n")
                    f.write(f"{k} = {v}\n")
                f.write("\n")
        finally:
            _unlock_file(f)
    logger.debug("Wrote models.ini to %s (%d sections)", ini_path, len(sections))


def list_sections(ini_path: Path) -> list[dict[str, Any]]:
    """Return list of model sections (excluding DEFAULT). Each item: {name, params}."""
    parser = read_ini(ini_path)
    result = []
    for section in parser.sections():
        if section == "DEFAULT":
            continue
        params = dict(parser[section])
        result.append({"name": section, "params": params})
    logger.debug("list_sections: %d models in %s", len(result), ini_path)
    return result


def get_section(ini_path: Path, section_name: str) -> dict[str, str] | None:
    """Get one section's params or None if missing."""
    parser = read_ini(ini_path)
    if not parser.has_section(section_name):
        logger.debug("get_section: section '%s' not found in %s", section_name, ini_path)
        return None
    logger.debug("get_section: loaded '%s' from %s", section_name, ini_path)
    return dict(parser[section_name])


def set_section(ini_path: Path, section_name: str, params: dict[str, str], param_descriptions: dict[str, str] = None) -> None:
    """Set or replace one section. Keys should be LLAMA_ARG_* style."""
    parser = read_ini(ini_path)
    if parser.has_section(section_name):
        parser.remove_section(section_name)
    parser.add_section(section_name)
    for k, v in params.items():
        parser.set(section_name, k, str(v))
    write_ini(ini_path, parser, param_descriptions)
    logger.info("set_section: wrote section '%s' (%d keys) to %s", section_name, len(params), ini_path)


def add_or_update_section(
    ini_path: Path,
    section_name: str,
    params: dict[str, str],
    *,
    merge: bool = True,
    param_descriptions: dict[str, str] = None,
) -> None:
    """Add or update a section. If merge=True, existing keys are preserved if not in params."""
    parser = read_ini(ini_path)
    existed = parser.has_section(section_name)
    if existed and merge:
        existing = dict(parser[section_name])
        for k, v in params.items():
            existing[k] = v
        params = existing
    if parser.has_section(section_name):
        parser.remove_section(section_name)
    parser.add_section(section_name)
    for k, v in params.items():
        parser.set(section_name, k, str(v))
    write_ini(ini_path, parser, param_descriptions)
    action = "updated" if existed else "added"
    logger.info(
        "add_or_update_section: %s section '%s' (%d keys, merge=%s) in %s",
        action, section_name, len(params), merge, ini_path,
    )


def delete_section(ini_path: Path, section_name: str) -> bool:
    """Remove a section. Returns True if it existed."""
    parser = read_ini(ini_path)
    if not parser.has_section(section_name):
        logger.debug("delete_section: section '%s' not found in %s", section_name, ini_path)
        return False
    parser.remove_section(section_name)
    write_ini(ini_path, parser)
    logger.info("delete_section: removed section '%s' from %s", section_name, ini_path)
    return True

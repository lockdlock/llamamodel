"""Run the LLM Manager web app. Uses port and config from config.yaml / env."""

import logging
import uvicorn

from app.config import load_config

# Configure root logger: INFO level, timestamped format.
# Individual modules use getLogger(__name__) so their output is captured here.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
# Reduce noise from third-party libraries
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = load_config()
    port = config["port"]
    logger.info("Starting LLM Manager on port %d (models_dir=%s)", port, config["models_dir"])
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )

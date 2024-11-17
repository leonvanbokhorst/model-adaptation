from pathlib import Path
import shutil
import logging
from typing import Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def clear_embedding_cache(cache_dir: str = "cache/embeddings"):
    """Clear the embedding cache directory."""
    cache_path = Path(cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        logger.info(f"Cleared embedding cache at {cache_dir}")


def get_cache_size(cache_dir: str = "cache/embeddings") -> Tuple[int, int]:
    """Get the number of cached embeddings and total size in MB."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return 0, 0

    files = list(cache_path.glob("*.pt"))
    size_bytes = sum(f.stat().st_size for f in files)
    return len(files), size_bytes / (1024 * 1024)  # Convert to MB

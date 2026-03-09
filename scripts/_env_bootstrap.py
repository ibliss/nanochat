"""Load .env before any torch import so CUDA_VISIBLE_DEVICES is respected. Import this first in scripts that use CUDA."""
from pathlib import Path
from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parent.parent
load_dotenv(_repo_root / ".env", override=True)

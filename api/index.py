"""
Point d'entrée Serverless Vercel pour l'API FastAPI.
Exporte l'instance FastAPI `app`.
"""
import sys
from pathlib import Path

# Ajouter la racine et backend/ au sys.path afin d'importer l'application FastAPI
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

for p in (str(BACKEND_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.main import app  # noqa: E402, F401

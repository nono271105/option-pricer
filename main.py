"""
main.py : Point d'entrée de l'application Eel (React + Python).

Usage :
    # Mode production (charge le dossier dist/ compilé)
    python main.py

    # Mode développement (pointe vers le serveur Vite sur :5173)
    OPTION_DEV=1 python main.py
"""

import os
import sys
import logging
import eel

# Configure le logging avant tout import
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Importe toutes les fonctions @eel.expose du pont API
import api  # noqa: F401  : les décorateurs @eel.expose s'enregistrent à l'import


def main() -> None:
    """Démarre l'application Eel."""
    dev_mode = os.getenv("OPTION_DEV", "0") == "1"

    # Dossier racine du projet (là où main.py se trouve)
    project_root = os.path.dirname(os.path.abspath(__file__))
    frontend_dist = os.path.join(project_root, "frontend", "dist")

    if dev_mode:
        # En développement, Eel redirige vers le serveur Vite (hot-reload)
        eel.init(os.path.join(project_root, "frontend", "dist"))
        logging.info("Mode développement : ouverture via http://localhost:5173")
        eel.start(
            {"port": 5173},       # URL du serveur Vite
            mode="chrome",
            host="localhost",
            port=8888,             # port Eel (pour eel.js)
            block=True,
            size=(1440, 900),
            position=(50, 50),
        )
    else:
        # En production, Eel sert le dossier dist/ compilé
        if not os.path.isdir(frontend_dist):
            logging.error(
                "Le dossier frontend/dist/ est introuvable. "
                "Lancez 'npm run build' dans le dossier frontend/ d'abord."
            )
            sys.exit(1)

        eel.init(frontend_dist)
        logging.info("Mode production : ouverture depuis %s", frontend_dist)
        eel.start(
            "index.html",
            mode="chrome",
            host="localhost",
            port=8888,
            block=True,
            size=(1440, 900),
            position=(50, 50),
        )


if __name__ == "__main__":
    main()

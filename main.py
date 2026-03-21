import os
import sys
from typing import Optional

# IMPORTANT: Importer QtWebEngineWidgets AVANT QApplication
# Cela évite les erreurs de contexte OpenGL
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWebEngineWidgets  # noqa: F401 - Important pour QWebEngineView

from gui_app import OptionPricingApp

def main() -> None:
    """Point d'entrée principal de l'application."""
    # Configurer l'attribute OpenGL pour éviter les conflits
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    
    app = QApplication(sys.argv)
    window = OptionPricingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

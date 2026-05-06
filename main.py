import sys

# IMPORTANT: Importer QtWebEngineWidgets AVANT QApplication
# Cela évite les erreurs de contexte OpenGL
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gui_app import OptionPricingApp

def main() -> None:
    """Point d'entrée principal de l'application."""
    # Configurer l'attribute OpenGL pour éviter les conflits
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    
    app = QApplication(sys.argv)
    window = OptionPricingApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

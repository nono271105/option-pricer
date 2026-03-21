"""
Onglet pour la visualisation 3D de la surface de volatilité implicite avec Plotly.
"""

from typing import Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFormLayout, QGroupBox, QMessageBox, QProgressBar
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QDoubleValidator
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from datetime import datetime

from implied_volatility_surface import ImpliedVolatilitySurface

# Seuil max d'IV accepté — au-delà : options illiquides / données aberrantes
IV_MAX_PCT = 2.50 # 250% d'IV


class SurfaceCalculationThread(QThread):
    """Thread pour calculer la surface IV sans bloquer l'UI."""

    finished = pyqtSignal()
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, ticker_symbol: str, current_price: Optional[float] = None):
        super().__init__()
        self.ticker_symbol = ticker_symbol
        self.current_price = current_price
        self.surface_calculator = ImpliedVolatilitySurface()
        self.raw_data  = None
        self.grid_data = None

    def run(self):
        try:
            self.progress.emit(25)
            self.raw_data, self.grid_data = self.surface_calculator.get_surface_for_ticker(
                self.ticker_symbol, self.current_price
            )
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class VolatilitySurfaceTab(QWidget):
    """
    Onglet pour la visualisation 3D de la surface de volatilité implicite.

    Axes :
        X : Strike (prix d'exercice)
        Y : Time to Maturity (jours jusqu'à expiration)
        Z : Implied Volatility (%)
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.surface_calculator = ImpliedVolatilitySurface()
        self.current_price: Optional[float] = None
        self.raw_data  = None
        self.grid_data = None
        self.calculation_thread: Optional[SurfaceCalculationThread] = None

        self._init_ui()

    # =========================================================================
    # UI
    # =========================================================================

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # ── Panneau de contrôle ────────────────────────────────────────────
        control_group  = QGroupBox("Paramètres de la Surface IV")
        control_layout = QFormLayout()

        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setPlaceholderText("Ex: AAPL, MSFT, TSLA")
        control_layout.addRow("Ticker Symbole:", self.ticker_input)

        self.price_input = QLineEdit()
        self.price_input.setValidator(QDoubleValidator(0.01, 100000.0, 2))
        self.price_input.setPlaceholderText(
            "Optionnel — laisser vide pour récupérer automatiquement")
        control_layout.addRow("Prix Actuel (optionnel):", self.price_input)

        btn_layout = QHBoxLayout()
        self.calculate_button = QPushButton("Calculer la Surface IV")
        self.calculate_button.clicked.connect(self.calculate_surface)
        btn_layout.addWidget(self.calculate_button)

        self.save_button = QPushButton("Exporter (HTML)")
        self.save_button.clicked.connect(self.export_html)
        self.save_button.setEnabled(False)
        btn_layout.addWidget(self.save_button)
        control_layout.addRow(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addRow("Progression:", self.progress_bar)

        self.status_label = QLabel("En attente de calcul...")
        self.status_label.setStyleSheet("color: gray;")
        control_layout.addRow(self.status_label)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # ── Zone graphique — QWebEngineView créé UNE SEULE FOIS à l'init ───
        # Instancier QWebEngineView dynamiquement (dans plot_surface) provoque
        # une fenêtre blanche sur Windows et parfois sur Mac car le contexte
        # OpenGL n'est pas correctement hérité. Le créer ici, une seule fois,
        # rattaché à la hiérarchie de widgets dès le départ, résout ce problème.
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(500)
        self.web_view.setHtml(
            "<html><body style='display:flex;align-items:center;"
            "justify-content:center;height:100vh;font-family:Arial;"
            "color:#888;font-size:16px;'>"
            "Cliquez sur « Calculer la Surface IV » pour démarrer"
            "</body></html>"
        )
        main_layout.addWidget(self.web_view, 1)

    # =========================================================================
    # Calcul
    # =========================================================================

    def calculate_surface(self) -> None:
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer un ticker.")
            return

        current_price = None
        if self.price_input.text().strip():
            try:
                current_price = float(self.price_input.text())
            except ValueError:
                QMessageBox.warning(self, "Erreur", "Prix invalide.")
                return

        self.calculate_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("⏳ Récupération des données de marché...")
        self.status_label.setStyleSheet("color: orange;")

        self.calculation_thread = SurfaceCalculationThread(ticker, current_price)
        self.calculation_thread.progress.connect(self._on_progress)
        self.calculation_thread.finished.connect(self._on_finished)
        self.calculation_thread.error.connect(self._on_error)
        self.calculation_thread.start()

    def _on_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    def _on_finished(self) -> None:
        if self.calculation_thread:
            self.raw_data  = self.calculation_thread.raw_data
            self.grid_data = self.calculation_thread.grid_data

        # Filtrage des IV aberrantes (options très OTM illiquides)
        if self.raw_data is not None and not self.raw_data.empty:
            before = len(self.raw_data)
            self.raw_data = self.raw_data[
                self.raw_data['IV'] <= IV_MAX_PCT
            ].copy()
            filtered = before - len(self.raw_data)
            if filtered > 0:
                print(f"Surface IV : {filtered} points filtrés "
                      f"(IV > {IV_MAX_PCT*100:.0f}%)")

        if self.raw_data is not None and not self.raw_data.empty:
            self._display_figure()
            self.status_label.setText(
                f"✓ Surface IV calculée ({len(self.raw_data)} points)")
            self.status_label.setStyleSheet("color: green;")
            self.save_button.setEnabled(True)
        else:
            self.status_label.setText("✗ Erreur : aucune donnée valide")
            self.status_label.setStyleSheet("color: red;")

        self.calculate_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _on_error(self, error_msg: str) -> None:
        QMessageBox.critical(self, "Erreur de calcul", error_msg)
        self.status_label.setText(f"✗ Erreur : {error_msg}")
        self.status_label.setStyleSheet("color: red;")
        self.calculate_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    # =========================================================================
    # Construction de la figure — source unique partagée par app ET export
    # =========================================================================

    def _build_figure(self) -> go.Figure:
        """
        Construit la figure Plotly 3D.
        Appelée par _display_figure() ET export_html() pour garantir que
        l'affichage dans l'app et le fichier exporté sont strictement identiques
        (mêmes axes, même colorscale, même caméra, mêmes données filtrées).
        """
        ticker = self.ticker_input.text().strip().upper()
        fig    = go.Figure()

        # Points bruts masqués par défaut (visibles via clic sur légende)
        fig.add_trace(go.Scatter3d(
            x=self.raw_data['Strike'],
            y=self.raw_data['Days_to_Maturity'],
            z=self.raw_data['IV'] * 100,
            mode='markers',
            marker=dict(
                size=2,
                color=self.raw_data['IV'] * 100,
                colorscale='Plasma',
                showscale=False,
                opacity=0.4,
            ),
            name='Données marché',
            visible='legendonly',
            hovertemplate=(
                '<b>Strike :</b> $%{x:.2f}<br>'
                '<b>Maturité :</b> %{y} j<br>'
                '<b>IV :</b> %{z:.2f}%<extra></extra>'
            ),
        ))

        # Surface interpolée
        if self.grid_data is not None:
            X_grid, Y_grid, Z_grid = self.grid_data
            Z_pct = np.clip(Z_grid * 100, 0, IV_MAX_PCT * 100)
            fig.add_trace(go.Surface(
                x=X_grid[0],
                y=Y_grid[:, 0],
                z=Z_pct,
                colorscale='Plasma',
                opacity=0.85,
                name='Surface interpolée',
                showscale=True,
                colorbar=dict(title="IV (%)", thickness=15, len=0.7),
                hovertemplate=(
                    '<b>Strike :</b> %{x:.2f}<br>'
                    '<b>Maturité :</b> %{y} j<br>'
                    '<b>IV :</b> %{z:.2f}%<extra></extra>'
                ),
            ))

        fig.update_layout(
            title=dict(
                text=f"Surface de Volatilité Implicite — {ticker}",
                font=dict(size=16),
            ),
            scene=dict(
                xaxis=dict(
                    title='Strike (K)',
                    backgroundcolor="rgb(230,230,230)",
                    gridcolor="white",
                    showbackground=True,
                ),
                yaxis=dict(
                    title="Jours jusqu'à expiration",
                    backgroundcolor="rgb(230,230,230)",
                    gridcolor="white",
                    showbackground=True,
                ),
                zaxis=dict(
                    title='Volatilité Implicite (%) — σ',
                    backgroundcolor="rgb(230,230,230)",
                    gridcolor="white",
                    showbackground=True,
                ),
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.3),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1),
                ),
                aspectmode='cube',
            ),
            width=1200,
            height=700,
            hovermode='closest',
            showlegend=True,
        )
        return fig

    # =========================================================================
    # Affichage dans l'app
    # =========================================================================

    def _display_figure(self) -> None:
        """Écrit la figure dans un fichier temporaire et le charge via URL.
        
        setHtml() est limité à ~2MB — au-delà le contenu est silencieusement
        ignoré. Passer par un fichier temporaire + load(QUrl) contourne cette
        limite et fonctionne de manière identique sur Mac et Windows.
        """
        try:
            import tempfile, os
            fig = self._build_figure()

            # Fichier temporaire persistant le temps de la session
            if not hasattr(self, '_tmp_html_path'):
                tmp = tempfile.NamedTemporaryFile(
                    suffix='.html', delete=False, mode='w', encoding='utf-8'
                )
                self._tmp_html_path = tmp.name
                tmp.close()

            fig.write_html(
                self._tmp_html_path,
                include_plotlyjs='inline',
                full_html=True,
            )
            self.web_view.load(QUrl.fromLocalFile(self._tmp_html_path))

        except Exception as e:
            QMessageBox.critical(
                self, "Erreur d'affichage",
                f"Impossible d'afficher le graphique : {e}"
            )

    # =========================================================================
    # Export HTML
    # =========================================================================

    def export_html(self) -> None:
        if self.raw_data is None or self.raw_data.empty:
            QMessageBox.warning(self, "Erreur", "Pas de surface à exporter.")
            return
        try:
            ticker   = self.ticker_input.text().strip().upper()
            filename = (
                f"iv_surface_{ticker}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            fig = self._build_figure()   # même figure que dans l'app
            fig.write_html(filename, include_plotlyjs='inline')
            QMessageBox.information(
                self, "Succès", f"Fichier exporté : {filename}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Erreur", f"Erreur lors de l'export : {e}"
            )

    # =========================================================================
    # Synchronisation depuis gui_app
    # =========================================================================

    def update_financial_params(self, ticker: str, S: float) -> None:
        """Met à jour le ticker et le prix (appelé depuis l'app principale)."""
        if ticker:
            self.ticker_input.setText(ticker)
        if S is not None:
            self.price_input.setText(f"{S:.2f}")
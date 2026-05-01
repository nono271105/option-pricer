"""
Onglet Forecast TimesFM — Prévision du prix sous-jacent via TimesFM 2.5-200M
et repricing BSM jour par jour sur l'horizon de forecast.

Le modèle TimesFM (~800 MB) est importé DANS le QThread pour ne pas bloquer
le démarrage de l'application.
"""

import numpy as np
from datetime import date, datetime
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QFormLayout, QGroupBox,
    QMessageBox, QDateEdit, QSpinBox, QSplitter, QProgressBar
)
from PyQt5.QtCore import QDate, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QDoubleValidator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from option_models import OptionModels
from forecast_logic import ForecastLogic

# ---------------------------------------------------------------------------
# Worker QThread — exécute le forecast TimesFM en arrière-plan
# ---------------------------------------------------------------------------
class ForecastWorker(QThread):
    """
    Thread dédié au chargement de TimesFM et à l'inférence.

    Signals:
        finished(object, object, object):
            Émet (point_forecast, quantile_forecast, history_prices)
            - point_forecast  : np.ndarray shape (1, horizon)
            - quantile_forecast : np.ndarray shape (1, horizon, 10)
            - history_prices  : np.ndarray des 252 derniers cours de clôture
        error(str):
            Émet un message d'erreur en cas d'échec.
    """
    finished = pyqtSignal(object, object, object)
    error = pyqtSignal(str)

    def __init__(self, ticker: str, horizon: int, forecast_logic: ForecastLogic, parent=None):
        super().__init__(parent)
        self.ticker = ticker
        self.horizon = horizon
        self.forecast_logic = forecast_logic

    def run(self):
        try:
            point_forecast, quantile_forecast, prices = self.forecast_logic.run_forecast(
                self.ticker, self.horizon
            )
            self.finished.emit(point_forecast, quantile_forecast, prices)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Widget principal de l'onglet
# ---------------------------------------------------------------------------
class ForecastTimesFMTab(QWidget):
    """
    Onglet « Forecast TimesFM » intégré au QTabWidget principal.

    Fonctionnalités :
      - Panneau de contrôle (ticker, horizon, strike, maturité, type, bouton)
      - Prévision TimesFM exécutée dans un QThread séparé
      - Repricing BSM jour par jour sur le point_forecast
      - 3 subplots : (1) prix historique + forecast, (2) prix option, (3) delta
      - Synchronisation via update_financial_params()
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.option_models = OptionModels()
        self.forecast_logic = ForecastLogic(self.option_models)

        # Paramètres financiers synchronisés depuis l'app principale
        self.ticker_symbol: str = "AAPL"
        self.S: Optional[float] = None
        self.r: float = 0.05
        self.q: float = 0.0
        self.sigma: float = 0.20

        # Référence au thread en cours (pour éviter le garbage-collect)
        self._worker: Optional[ForecastWorker] = None

        self.init_ui()

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # === Panneau de contrôle (gauche) ===
        control_panel = QVBoxLayout()
        control_group = QGroupBox("Paramètres Forecast TimesFM")
        form = QFormLayout()

        # Ticker
        self.ticker_input = QLineEdit(self.ticker_symbol)
        self.ticker_input.setPlaceholderText("Ex : AAPL")
        form.addRow("Ticker :", self.ticker_input)

        # Horizon (5 – 63 jours)
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(5, 63)
        self.horizon_spin.setValue(21)
        self.horizon_spin.setSuffix(" jours")
        form.addRow("Horizon :", self.horizon_spin)

        # Strike K
        self.strike_input = QLineEdit("150.00")
        self.strike_input.setValidator(QDoubleValidator(0.01, 100000.0, 2))
        form.addRow("Strike (K) :", self.strike_input)

        # Date de maturité
        self.maturity_date_input = QDateEdit(QDate.currentDate().addMonths(3))
        self.maturity_date_input.setCalendarPopup(True)
        self.maturity_date_input.setDisplayFormat("dd/MM/yyyy")
        form.addRow("Maturité :", self.maturity_date_input)

        # Type call / put
        self.option_type_combo = QComboBox()
        self.option_type_combo.addItems(["call", "put"])
        form.addRow("Type :", self.option_type_combo)

        # Bouton lancer
        self.launch_button = QPushButton("Lancer le Forecast")
        self.launch_button.clicked.connect(self.on_launch)
        form.addRow(self.launch_button)

        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        form.addRow("Progression :", self.progress_bar)

        # Label de statut
        self.status_label = QLabel("En attente…")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        form.addRow("Statut :", self.status_label)

        # Données synchronisées (lecture seule)
        self.s_label = QLabel("N/A")
        self.r_label = QLabel("N/A")
        self.q_label = QLabel("N/A")
        self.sigma_label = QLabel("N/A")
        form.addRow("Prix Actuel (S) :", self.s_label)
        form.addRow("Taux sans risque (r) :", self.r_label)
        form.addRow("Dividende (q) :", self.q_label)
        form.addRow("Volatilité (σ) :", self.sigma_label)

        control_group.setLayout(form)
        control_panel.addWidget(control_group)
        control_panel.addStretch(1)

        main_layout.addLayout(control_panel, 1)

        # === Zone graphique (droite) ===
        plot_group = QGroupBox("Résultats du Forecast")
        plot_layout = QVBoxLayout()

        self.fig = Figure(figsize=(10, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)

        plot_group.setLayout(plot_layout)
        main_layout.addWidget(plot_group, 3)

    # --------------------------------------------------------- Synchronisation
    def update_financial_params(self, ticker, S, r, q, sigma):
        """
        Appelée par l'app principale pour synchroniser les paramètres
        financiers partagés entre onglets.
        """
        if ticker:
            self.ticker_symbol = ticker
            if self.ticker_input.text() == "" or self.ticker_input.text() != ticker:
                self.ticker_input.setText(ticker)

        if S is not None:
            self.S = S
            self.s_label.setText(f"{S:.2f}")
        if r is not None:
            self.r = r
            self.r_label.setText(f"{r*100:.2f} %")
        if q is not None:
            self.q = q
            self.q_label.setText(f"{q*100:.2f} %")
        if sigma is not None:
            self.sigma = sigma
            self.sigma_label.setText(f"{sigma*100:.2f} %")

    # --------------------------------------------------------- Lancement
    def on_launch(self):
        """Valide les inputs et lance le ForecastWorker."""
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Ticker manquant",
                                "Veuillez entrer un symbole de ticker.")
            return

        try:
            K = float(self.strike_input.text())
            if K <= 0:
                raise ValueError("K <= 0")
        except ValueError:
            QMessageBox.warning(self, "Strike invalide",
                                "Veuillez entrer un strike K valide (> 0).")
            return

        horizon = self.horizon_spin.value()

        # Extraire la date de maturité (compatible Python 3.8)
        qd = self.maturity_date_input.date()
        maturity = date(qd.year(), qd.month(), qd.day())
        today = date.today()
        T_total = (maturity - today).days / 365.0
        if T_total <= 0:
            QMessageBox.warning(self, "Maturité invalide",
                                "La date de maturité doit être dans le futur.")
            return

        # Désactiver le bouton et afficher le statut
        self.launch_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self._set_status("Chargement du modèle TimesFM…", "orange")

        # Lancer le worker
        self._worker = ForecastWorker(ticker, horizon, self.forecast_logic)
        self._worker.finished.connect(
            lambda pf, qf, hist: self._on_forecast_done(pf, qf, hist, K, T_total, horizon)
        )
        self._worker.error.connect(self._on_forecast_error)
        self._worker.start()

    # --------------------------------------------------------- Callbacks
    def _on_forecast_done(self, point_forecast, quantile_forecast,
                          history_prices, K, T_total, horizon):
        """Appelée quand le ForecastWorker a terminé avec succès."""
        try:
            self._set_status("Forecast terminé", "#2ecc71")

            option_type = self.option_type_combo.currentText()

            # --- Extraire les séries ---
            # point_forecast[0] → ndarray shape (horizon,)
            pf = point_forecast[0]
            # quantile_forecast[0] → ndarray shape (horizon, 10)
            qf = quantile_forecast[0]
            q10 = qf[:, 0]    # quantile 10 %
            q90 = qf[:, -1]   # quantile 90 %

            # --- Repricing BSM jour par jour via la logique métier ---
            pf, option_prices, deltas, hist_slice, hist_option_prices, hist_deltas, x_hist = \
                self.forecast_logic.process_forecast_results(
                    point_forecast, history_prices, horizon, K, T_total, 
                    self.r, self.sigma, self.q, option_type
                )
            
            x_fc = np.arange(0, horizon)

            # --- Tracé des 3 subplots ---
            self.fig.clear()
            gs = self.fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

            # Historique 60 derniers jours + forecast
            ax1 = self.fig.add_subplot(gs[0, :])

            ax1.plot(x_hist, hist_slice, color="#3498db", linewidth=1.5,
                     label="Historique (30j)")
            ax1.plot(x_fc, pf, color="#e74c3c", linewidth=2,
                     label="Forecast (point)")
            ax1.fill_between(x_fc, q10, q90, alpha=0.18, color="#e74c3c",
                             label="Intervalle q10–q90")
            # Ligne de jonction
            ax1.plot([x_hist[-1], x_fc[0]],
                     [hist_slice[-1], pf[0]],
                     color="#e74c3c", linewidth=1, linestyle="--")
            ax1.axvline(0, color="gray", linewidth=0.8, linestyle=":")
            ax1.set_title("Prix Sous-Jacent", fontsize=10, fontweight="bold")
            ax1.set_xlabel("Jours")
            ax1.set_ylabel("Prix ($)")
            ax1.legend(fontsize=7, loc="upper left")
            ax1.grid(True, alpha=0.3)

            # Prix de l'option reprojeté
            ax2 = self.fig.add_subplot(gs[1, 0])
            ax2.plot(x_hist, hist_option_prices, color="#3498db", linewidth=1.5, label="Historique (30j)")
            ax2.plot(x_fc, option_prices, color="#2ecc71", linewidth=2, label="Forecast")
            ax2.plot([x_hist[-1], x_fc[0]], [hist_option_prices[-1], option_prices[0]],
                     color="#2ecc71", linewidth=1, linestyle="--")
            ax2.axvline(0, color="gray", linewidth=0.8, linestyle=":")
            ax2.set_title(
                f"Prix {option_type.capitalize()} (K={K:.0f})",
                fontsize=10, fontweight="bold"
            )
            ax2.set_xlabel("Jours")
            ax2.set_ylabel("Prix Option ($)")
            ax2.legend(fontsize=7, loc="upper left")
            ax2.grid(True, alpha=0.3)

            # Delta Forecast
            ax3 = self.fig.add_subplot(gs[1, 1])
            ax3.plot(x_hist, hist_deltas, color="#3498db", linewidth=1.5, label="Historique (30j)")
            ax3.plot(x_fc, deltas, color="#9b59b6", linewidth=2, label="Forecast")
            ax3.plot([x_hist[-1], x_fc[0]], [hist_deltas[-1], deltas[0]],
                     color="#9b59b6", linewidth=1, linestyle="--")
            ax3.axvline(0, color="gray", linewidth=0.8, linestyle=":")
            ax3.set_title("Delta Forecast", fontsize=10, fontweight="bold")
            ax3.set_xlabel("Jours")
            ax3.set_ylabel("Delta")
            ax3.legend(fontsize=7, loc="upper left")
            ax3.grid(True, alpha=0.3)

            self.fig.suptitle(
                f"Forecast TimesFM — {self.ticker_input.text().upper()}  "
                f"(horizon {horizon}j)",
                fontsize=12, fontweight="bold", y=1.02
            )
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as exc:
            self._set_status(f"Erreur post-traitement : {exc}", "#e74c3c")
            QMessageBox.critical(
                self, "Erreur de Post-Traitement",
                f"Erreur lors du repricing / tracé :\n{exc}"
            )
        finally:
            self.launch_button.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _on_forecast_error(self, error_msg: str):
        """Appelée quand le ForecastWorker rencontre une erreur."""
        self.progress_bar.setVisible(False)
        self._set_status(f"Erreur : {error_msg}", "#e74c3c")
        self.launch_button.setEnabled(True)
        QMessageBox.critical(
            self, "Erreur Forecast TimesFM",
            f"Le forecast a échoué :\n\n{error_msg}"
        )

    # --------------------------------------------------------- Helpers
    def _set_status(self, text: str, color: str):
        """Met à jour le label de statut avec une couleur de fond."""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"font-weight: bold; padding: 6px; border-radius: 4px; "
            f"color: white; background-color: {color};"
        )

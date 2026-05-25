# Interface de prevision de volatilite implicite par TimesFM et repricing BSM

import numpy as np
from datetime import date, datetime
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QFormLayout, QGroupBox,
    QMessageBox, QDateEdit, QSpinBox, QProgressBar
)
from PySide6.QtCore import QDate, QThread, Signal, Qt
from PySide6.QtGui import QDoubleValidator

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data_fetcher import DataFetcher
from logic.bsm_logic import OptionModels
from logic.forecast_logic import ForecastLogic


# thread d'isolement pour la recuperation des donnees et l'inference TimesFM
class ForecastWorker(QThread):
    """
    Thread dedie a l'orchestration du pipeline IV forecast.

    Etapes :
      1. Recuperation des prix historiques via marketdata.app
      2. Inversion BSM pour recalculer l'IV historique
      3. Inference TimesFM sur la serie d'IV

    Signals:
        finished(object, object, object):
            Emet (point_forecast, quantile_forecast, iv_history)
        error(str):
            Emet un message d'erreur en cas d'echec.
    """
    finished = Signal(object, object, object)
    error = Signal(str)

    def __init__(
        self,
        ticker: str,
        horizon: int,
        expiration_date: str,
        strike: float,
        option_type: str,
        r: float,
        q: float,
        history_days: int,
        forecast_logic: ForecastLogic,
        data_fetcher: DataFetcher,
        parent=None,
    ):
        super().__init__(parent)
        self.ticker = ticker
        self.horizon = horizon
        self.expiration_date = expiration_date
        self.strike = strike
        self.option_type = option_type
        self.r = r
        self.q = q
        self.history_days = history_days
        self.forecast_logic = forecast_logic
        self.data_fetcher = data_fetcher

    def run(self):
        try:
            # etape 1 : recuperation des prix historiques du contrat
            hist_data = self.data_fetcher.get_option_history_marketdata(
                self.ticker,
                self.expiration_date,
                self.strike,
                self.option_type,
                self.history_days,
            )

            if hist_data is None:
                self.error.emit(
                    f"Impossible de recuperer l'historique pour "
                    f"{self.ticker} K={self.strike} exp={self.expiration_date}.\n"
                    f"Verifier le token MARKET_DATA_TOKEN et les parametres."
                )
                return

            if len(hist_data["mid"]) < 5:
                self.error.emit(
                    f"Historique insuffisant ({len(hist_data['mid'])} points). "
                    f"Minimum requis : 5."
                )
                return

            # etape 2 : inversion BSM pour recalculer l'IV
            iv_history = self.forecast_logic.compute_iv_from_prices(
                mid_prices=hist_data["mid"],
                underlying_prices=hist_data["underlyingPrice"],
                dtes=hist_data["dte"],
                strike=self.strike,
                r=self.r,
                q=self.q,
                option_type=self.option_type,
            )

            # etape 3 : inference TimesFM sur la serie d'IV
            point_forecast, quantile_forecast, iv_hist = self.forecast_logic.run_iv_forecast(
                iv_history, self.horizon
            )
            self.finished.emit(point_forecast, quantile_forecast, iv_hist)

        except Exception as exc:
            self.error.emit(str(exc))


# controleur de la vue de prevision IV
class ForecastTimesFMTab(QWidget):
    """
    Onglet Forecast TimesFM integre au QTabWidget principal.

    Fonctionnalites :
      - Panneau de controle (ticker, horizon, strike, maturite, type)
      - Recuperation de l'historique d'IV via marketdata.app
      - Prevision de l'IV avec TimesFM
      - Repricing BSM et recalcul du delta a IV predite
      - 3 subplots : (1) IV historique + forecast, (2) prix option, (3) delta
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.option_models = OptionModels()
        self.forecast_logic = ForecastLogic(self.option_models)
        self.data_fetcher = DataFetcher()

        # isolation locale de l'etat financier
        self.ticker_symbol: str = "AAPL"
        self.S: Optional[float] = None
        self.r: float = 0.05
        self.q: float = 0.0
        self.sigma: float = 0.20

        # maintien de la duree de vie de l'objet asynchrone
        self._worker: Optional[ForecastWorker] = None

        self.init_ui()

    # assemblage des composants graphiques
    def init_ui(self) -> None:
        """Initialise l'interface utilisateur de l'onglet Forecast."""
        main_layout = QHBoxLayout(self)

        # zone de saisie des parametres
        control_panel = QVBoxLayout()
        control_group = QGroupBox("Paramètres Forecast IV (TimesFM)")
        form = QFormLayout()

        # selection de l'actif
        self.ticker_input = QLineEdit(self.ticker_symbol)
        self.ticker_input.setPlaceholderText("Ex : AAPL")
        form.addRow("Ticker :", self.ticker_input)

        # libelle descriptif
        self.company_name_label = QLabel("N/A")
        form.addRow("Entreprise :", self.company_name_label)

        # profondeur de projection
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(5, 63)
        self.horizon_spin.setValue(21)
        self.horizon_spin.setSuffix(" jours")
        form.addRow("Horizon :", self.horizon_spin)

        # nombre de jours d'historique IV a recuperer
        self.history_days_spin = QSpinBox()
        self.history_days_spin.setRange(10, 90)
        self.history_days_spin.setValue(30)
        self.history_days_spin.setSuffix(" jours")
        form.addRow("Historique IV :", self.history_days_spin)

        # seuil d'exercice
        self.strike_input = QLineEdit("150.00")
        self.strike_input.setValidator(QDoubleValidator(0.01, 100000.0, 2))
        form.addRow("Strike (K) :", self.strike_input)

        # echeance du contrat
        from utils import get_default_maturity_date
        self.maturity_date_input = QDateEdit(get_default_maturity_date())
        self.maturity_date_input.setCalendarPopup(True)
        self.maturity_date_input.setDisplayFormat("dd/MM/yyyy")
        form.addRow("Maturité :", self.maturity_date_input)

        # sens de la transaction
        self.option_type_combo = QComboBox()
        self.option_type_combo.addItems(["call", "put"])
        form.addRow("Type :", self.option_type_combo)

        # declenchement de l'inference
        self.launch_button = QPushButton("Lancer le Forecast IV")
        self.launch_button.clicked.connect(self.on_launch)
        form.addRow(self.launch_button)

        # indicateur d'activite asynchrone
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        form.addRow("Progression :", self.progress_bar)

        # retour textuel sur l'etat du systeme
        self.status_label = QLabel("En attente…")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "font-weight: bold; padding: 6px; border-radius: 4px;"
        )
        form.addRow("Statut :", self.status_label)

        self.s_label = QLabel("N/A")
        self.r_label = QLabel("N/A")
        self.q_label = QLabel("N/A")
        form.addRow("Prix Actuel (S) :", self.s_label)
        form.addRow("Taux sans risque (r) :", self.r_label)
        form.addRow("Dividende (q) :", self.q_label)

        control_group.setLayout(form)
        control_panel.addWidget(control_group)
        control_panel.addStretch(1)

        main_layout.addLayout(control_panel, 1)

        # zone d'affichage des resultats
        plot_group = QGroupBox("Résultats du Forecast IV")
        plot_layout = QVBoxLayout()

        self.fig = Figure(figsize=(10, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)

        plot_group.setLayout(plot_layout)
        main_layout.addWidget(plot_group, 3)

    # ecoute active des changements d'etat global
    def update_financial_params(self, ticker, S, r, q):
        """
        Appelee par l'app principale pour synchroniser les parametres
        financiers partages entre onglets.
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

    def update_company_name(self, company_name: str) -> None:
        """Met a jour le label du nom de l'entreprise."""
        self.company_name_label.setText(company_name if company_name else "N/A")

    # execution du pipeline predictif
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

        if self.S is None:
            QMessageBox.warning(self, "Spot manquant",
                                "Le prix spot (S) n'est pas disponible. "
                                "Veuillez d'abord charger les données dans l'onglet BSM.")
            return

        horizon = self.horizon_spin.value()
        history_days = self.history_days_spin.value()

        # calcul de la duree de vie residuelle
        qd = self.maturity_date_input.date()
        maturity = date(qd.year(), qd.month(), qd.day())
        today = date.today()

        # construction de la date d'expiration au format API
        expiration_date = maturity.strftime("%Y-%m-%d")

        # Verification de l'existence de l'expiration et ajustement
        try:
            import yfinance as yf
            import datetime
            tkr = yf.Ticker(ticker)
            expirations = tkr.options
            if expirations and expiration_date not in expirations:
                # trouver la date la plus proche
                closest_date = min(expirations, 
                                   key=lambda x: abs(datetime.datetime.strptime(x, '%Y-%m-%d').date() - maturity))
                reply = QMessageBox.question(
                    self,
                    "Échéance introuvable",
                    f"L'échéance {expiration_date} n'existe pas sur le marché pour {ticker}.\n\n"
                    f"Voulez-vous utiliser l'échéance réelle la plus proche ({closest_date}) ?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Update UI et continue
                    closest_qdate = QDate.fromString(closest_date, "yyyy-MM-dd")
                    self.maturity_date_input.setDate(closest_qdate)
                    expiration_date = closest_date
                    maturity = datetime.datetime.strptime(closest_date, "%Y-%m-%d").date()
                else:
                    # annulation du lancement du forecast
                    return
        except Exception as e:
            # en cas d'erreur avec yfinance, on continue silencieusement
            pass

        T_total = (maturity - today).days / 365.0
        if T_total <= 0:
            QMessageBox.warning(self, "Maturité invalide",
                                "La date de maturité doit être dans le futur.")
            return

        option_type = self.option_type_combo.currentText()

        # verrouillage preventif de l'interface
        self.launch_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self._set_status("Récupération IV historique…", "orange")

        # instanciation du processus d'arriere-plan
        self._worker = ForecastWorker(
            ticker=ticker,
            horizon=horizon,
            expiration_date=expiration_date,
            strike=K,
            option_type=option_type,
            r=self.r,
            q=self.q,
            history_days=history_days,
            forecast_logic=self.forecast_logic,
            data_fetcher=self.data_fetcher,
        )
        self._worker.finished.connect(
            lambda pf, qf, iv_hist: self._on_forecast_done(
                pf, qf, iv_hist, K, T_total, horizon, option_type
            )
        )
        self._worker.error.connect(self._on_forecast_error)
        self._worker.start()

    # gestionnaires des evenements de fin de traitement
    def _on_forecast_done(self, point_forecast, quantile_forecast,
                          iv_history, K, T_total, horizon, option_type):
        """Appelee quand le ForecastWorker a termine avec succes."""
        try:
            self._set_status("Forecast terminé", "#2ecc71")

            S = self.S

            # extraction des bornes de l'intervalle de confiance (TimesFM renvoie [mean, q10, q20, ..., q90])
            qf = quantile_forecast[0]
            q10 = qf[:, 1]
            q90 = qf[:, -1]

            # repricing avec les IV predites
            iv_fc, option_prices, deltas, iv_hist_slice, hist_option_prices, hist_deltas, x_hist = \
                self.forecast_logic.process_iv_forecast_results(
                    point_forecast, iv_history, horizon, K, T_total,
                    S, self.r, self.q, option_type
                )

            x_fc = np.arange(0, horizon)

            # conversion en pourcentage pour l'affichage
            iv_hist_pct = iv_hist_slice * 100
            iv_fc_pct = iv_fc * 100
            q10_pct = q10 * 100
            q90_pct = q90 * 100

            # generation dynamique de la grille de graphiques
            self.fig.clear()
            gs = self.fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

            # subplot 1 : continuite temporelle de l'IV
            ax1 = self.fig.add_subplot(gs[0, :])
            ax1.plot(x_hist, iv_hist_pct, color="#3498db", linewidth=1.5,
                     label="IV Historique")
            ax1.plot(x_fc, iv_fc_pct, color="#e74c3c", linewidth=2,
                     label="IV Forecast (TimesFM)")
            ax1.fill_between(x_fc, q10_pct, q90_pct, alpha=0.18, color="#e74c3c",
                             label="Intervalle q10–q90")
            # raccordement visuel entre le reel et le predictif
            ax1.plot([x_hist[-1], x_fc[0]],
                     [iv_hist_pct[-1], iv_fc_pct[0]],
                     color="#e74c3c", linewidth=1, linestyle="--")
            ax1.axvline(0, color="gray", linewidth=0.8, linestyle=":")
            ax1.set_title("Volatilité Implicite — Historique & Forecast", fontsize=10, fontweight="bold")
            ax1.set_xlabel("Jours")
            ax1.set_ylabel("IV (%)")
            ax1.legend(fontsize=7, loc="upper left")
            ax1.grid(True, alpha=0.3)

            # subplot 2 : evolution du prix de l'option reprice a IV predite
            ax2 = self.fig.add_subplot(gs[1, 0])
            ax2.plot(x_hist, hist_option_prices, color="#3498db", linewidth=1.5, label="Historique")
            ax2.plot(x_fc, option_prices, color="#2ecc71", linewidth=2, label="Forecast")
            ax2.plot([x_hist[-1], x_fc[0]], [hist_option_prices[-1], option_prices[0]],
                     color="#2ecc71", linewidth=1, linestyle="--")
            ax2.axvline(0, color="gray", linewidth=0.8, linestyle=":")
            ax2.set_title(
                f"Prix {option_type.capitalize()} (K={K:.0f}, S={S:.0f})",
                fontsize=10, fontweight="bold"
            )
            ax2.set_xlabel("Jours")
            ax2.set_ylabel("Prix Option ($)")
            ax2.legend(fontsize=7, loc="upper left")
            ax2.grid(True, alpha=0.3)

            # subplot 3 : evolution de la sensibilite directionnelle
            ax3 = self.fig.add_subplot(gs[1, 1])
            ax3.plot(x_hist, hist_deltas, color="#3498db", linewidth=1.5, label="Historique")
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
                f"Forecast IV TimesFM — {self.ticker_input.text().upper()}  "
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
        """Appelee quand le ForecastWorker rencontre une erreur."""
        self.progress_bar.setVisible(False)
        self._set_status(f"Erreur : {error_msg}", "#e74c3c")
        self.launch_button.setEnabled(True)
        QMessageBox.critical(
            self, "Erreur Forecast IV",
            f"Le forecast a échoué :\n\n{error_msg}"
        )

    # fonctions utilitaires
    def _set_status(self, text: str, color: str):
        """Met a jour le label de statut avec une couleur de fond."""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"font-weight: bold; padding: 6px; border-radius: 4px; "
            f"color: white; background-color: {color};"
        )

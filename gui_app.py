from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QFormLayout, QGroupBox, QGridLayout,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QDateEdit,
    QTabWidget, QDialog
)
from PyQt5.QtCore import QDate, QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from datetime import date, datetime 
from scipy.interpolate import make_interp_spline
from concurrent.futures import ThreadPoolExecutor

from data_fetcher import DataFetcher
from option_models import OptionModels
from strategy_manager import StrategyManager
from simulation_tab import CallPriceSimulationTab
from volatility_smile_tab import VolatilitySmileTab
from volatility_surface_tab import VolatilitySurfaceTab
from exotic_options_tab import ExoticOptionsTab
from strategy_tab import StrategyTab
from forecast_tab import ForecastTimesFMTab

class PlottingDialog(QDialog):
    """
    Une petite fenêtre de dialogue pour afficher les graphiques Matplotlib.
    """
    def __init__(self, parent=None, title="Graphique"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(150, 150, 700, 500)

        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)


class CRRModelTab(QWidget):
    """
    Onglet pour le calcul du prix et des Grecs des options Américaines (CRR).
    """
    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Panneau de contrôle à gauche ---
        control_panel_layout = QVBoxLayout()
        control_panel_group = QGroupBox("Paramètres de l'option (CRR)")
        control_form_layout = QFormLayout()

        # Ticker (modifiable)
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ex: AAPL")
        control_form_layout.addRow("Ticker Symbole:", self.ticker_input)

        self.option_type_combo = QComboBox()
        self.option_type_combo.addItems(["call", "put"])
        control_form_layout.addRow("Type d'option:", self.option_type_combo)

        self.strike_input = QLineEdit("150.00")
        self.strike_input.setValidator(QDoubleValidator(0.0, 100000.0, 2))
        control_form_layout.addRow("Prix d'exercice (K):", self.strike_input)

        self.maturity_date_input = QDateEdit(QDate.currentDate().addMonths(3))
        self.maturity_date_input.setCalendarPopup(True)
        self.maturity_date_input.setDisplayFormat("dd/MM/yyyy")
        control_form_layout.addRow("Date d'échéance:", self.maturity_date_input)

        self.position_combo = QComboBox()
        self.position_combo.addItems(["long", "short"])
        control_form_layout.addRow("Position:", self.position_combo)
        
        # Nombre de Pas N
        self.steps_input = QLineEdit("100")
        self.steps_input.setValidator(QIntValidator(1, 10000))
        control_form_layout.addRow("Nombre de pas (N):", self.steps_input)
        
        # Le bouton d'appel doit appeler la nouvelle méthode fetch_data_for_tab
        self.fetch_data_button = QPushButton("Récupérer/Synchroniser les Données")
        self.fetch_data_button.clicked.connect(lambda: self.app.fetch_data_for_tab(self.ticker_input.text(), self))
        control_form_layout.addRow(self.fetch_data_button)

        self.calculate_button = QPushButton("Calculer Prix et Grecs (CRR)")
        self.calculate_button.clicked.connect(self.app.calculate_crr_metrics)
        control_form_layout.addRow(self.calculate_button)

        self.plot_payoff_button = QPushButton("Tracer le Payoff")
        self.plot_payoff_button.clicked.connect(self.app.plot_crr_payoff)
        control_form_layout.addRow(self.plot_payoff_button)

        control_panel_group.setLayout(control_form_layout)
        control_panel_layout.addWidget(control_panel_group)
        control_panel_layout.addStretch(1)

        main_layout.addLayout(control_panel_layout, 1)

        # --- Panneau d'affichage à droite ---
        display_panel_layout = QVBoxLayout()

        current_data_group = QGroupBox("Données Actuelles")
        current_data_layout = QFormLayout()
        
        self.live_price_label = QLabel("N/A")
        self.risk_free_rate_label = QLabel("N/A")
        self.dividend_yield_label = QLabel("N/A")
        self.vol_label = QLabel("N/A")
        self.crr_price_label = QLabel("N/A")

        current_data_layout.addRow("Prix Actuel (S):", self.live_price_label)
        current_data_layout.addRow("Taux Sans Risque SOFR (r):", self.risk_free_rate_label)
        current_data_layout.addRow("Rendement Dividende (q):", self.dividend_yield_label)
        current_data_layout.addRow("Volatilité Utilisée (σ):", self.vol_label)
        current_data_layout.addRow("Prix de l'option (CRR):", self.crr_price_label)
        current_data_group.setLayout(current_data_layout)
        display_panel_layout.addWidget(current_data_group)

        greeks_group = QGroupBox("Grecs (CRR)")
        greeks_table_layout = QGridLayout()
        self.greeks_table = QTableWidget(1, 5)
        self.greeks_table.setHorizontalHeaderLabels(["Delta", "Gamma", "Theta (par jour)", "Vega", "Rho"])
        self.greeks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.greeks_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.greeks_table.cellClicked.connect(lambda row, col: self.app.handle_crr_greek_click(row, col))

        for col in range(5):
            self.greeks_table.setItem(0, col, QTableWidgetItem("N/A"))

        greeks_table_layout.addWidget(self.greeks_table, 0, 0)
        greeks_group.setLayout(greeks_table_layout)
        display_panel_layout.addWidget(greeks_group)

        payoff_plot_group = QGroupBox("Payoff de l'option")
        payoff_plot_layout = QVBoxLayout()

        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        payoff_plot_layout.addWidget(self.canvas)
        payoff_plot_group.setLayout(payoff_plot_layout)
        display_panel_layout.addWidget(payoff_plot_group)

        main_layout.addLayout(display_panel_layout, 2)

    def update_financial_data(self, S, r, q, sigma_used, ticker, pricing_method=""):
        """Met à jour les labels d'information financière de l'onglet CRR."""
        if self.ticker_input.text() == "":
            self.ticker_input.setText(ticker)
            
        self.live_price_label.setText(f"{S:.2f}" if S is not None else "N/A")
        self.risk_free_rate_label.setText(f"{r*100:.2f}%" if r is not None else "N/A")
        self.dividend_yield_label.setText(f"{q*100:.2f}%" if q is not None else "N/A")
        
        if sigma_used is not None:
            self.vol_label.setText(f"{sigma_used*100:.2f}% ({pricing_method})")
        else:
            self.vol_label.setText("N/A")
            
class FetchDataWorker(QThread):
    data_ready = pyqtSignal(str, object, object, object, object)
    error = pyqtSignal(str)

    def __init__(self, data_fetcher, ticker_symbol):
        super().__init__()
        self.data_fetcher = data_fetcher
        self.ticker_symbol = ticker_symbol

    def run(self):
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                price_future = executor.submit(self.data_fetcher.get_live_price, self.ticker_symbol)
                sofr_future = executor.submit(self.data_fetcher.get_sofr_rate)
                dividend_future = executor.submit(self.data_fetcher.get_dividend_yield, self.ticker_symbol)
                vol_future = executor.submit(self.data_fetcher.get_historical_volatility, self.ticker_symbol, "1y")
                live_price = price_future.result(timeout=10)
                sofr = sofr_future.result(timeout=10)
                dividend = dividend_future.result(timeout=10)
                vol = vol_future.result(timeout=10)
            self.data_ready.emit(self.ticker_symbol, live_price, sofr, dividend, vol)
        except Exception as e:
            self.error.emit(str(e))


class OptionPricingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Option Pricer")
        self.setGeometry(100, 100, 1400, 900)

        self.data_fetcher = DataFetcher()
        self.option_models = OptionModels()
        self.strategy_manager = StrategyManager()

        self.S = None
        self.r = None
        self.q = None
        self.historical_vol = None
        self.current_ticker = None
        self.pricing_method = "N/A"
        self.K = None
        self.T = None
        self.option_type = None
        self.current_sigma = None 
        self._fetch_worker = None

        self.init_ui()

    def init_ui(self):
        self.tab_widget = QTabWidget()
        option_calculator_widget = QWidget()
        option_calculator_layout = QHBoxLayout(option_calculator_widget)

        # --- Panneau de contrôle (BSM) ---
        control_panel_layout = QVBoxLayout()
        control_panel_group = QGroupBox("Paramètres de l'option (BSM)") 
        control_form_layout = QFormLayout()

        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setPlaceholderText("Ex: AAPL")
        control_form_layout.addRow("Ticker Symbole:", self.ticker_input)

        self.option_type_combo = QComboBox()
        self.option_type_combo.addItems(["call", "put"])
        control_form_layout.addRow("Type d'option:", self.option_type_combo)

        self.strike_input = QLineEdit("150.00")
        self.strike_input.setValidator(QDoubleValidator(0.0, 100000.0, 2))
        control_form_layout.addRow("Prix d'exercice (K):", self.strike_input)

        self.maturity_date_input = QDateEdit(QDate.currentDate().addMonths(3))
        self.maturity_date_input.setCalendarPopup(True)
        self.maturity_date_input.setDisplayFormat("dd/MM/yyyy")
        control_form_layout.addRow("Date d'échéance:", self.maturity_date_input)

        self.position_combo = QComboBox()
        self.position_combo.addItems(["long", "short"])
        control_form_layout.addRow("Position:", self.position_combo)

        self.fetch_data_button = QPushButton("Récupérer les Données")
        self.fetch_data_button.clicked.connect(lambda: self.fetch_data_for_tab(self.ticker_input.text(), self))
        control_form_layout.addRow(self.fetch_data_button)

        self.calculate_button = QPushButton("Calculer Prix et Grecs (BSM)")
        self.calculate_button.clicked.connect(self.calculate_option_metrics)
        control_form_layout.addRow(self.calculate_button)

        self.plot_payoff_button = QPushButton("Tracer le Payoff")
        self.plot_payoff_button.clicked.connect(self.plot_option_payoff)
        control_form_layout.addRow(self.plot_payoff_button)

        control_panel_group.setLayout(control_form_layout)
        control_panel_layout.addWidget(control_panel_group)
        control_panel_layout.addStretch(1)

        option_calculator_layout.addLayout(control_panel_layout, 1)

        # --- Panneau d'affichage (BSM) ---
        display_panel_layout = QVBoxLayout()

        current_data_group = QGroupBox("Données Actuelles")
        current_data_layout = QFormLayout()
        self.live_price_label = QLabel("N/A")
        self.risk_free_rate_label = QLabel("N/A")
        self.dividend_yield_label = QLabel("N/A")
        self.historical_vol_label = QLabel("N/A")
        self.bs_price_label = QLabel("N/A")

        current_data_layout.addRow("Prix Actuel (S):", self.live_price_label)
        current_data_layout.addRow("Taux Sans Risque SOFR (r):", self.risk_free_rate_label)
        current_data_layout.addRow("Rendement Dividende (q):", self.dividend_yield_label)
        current_data_layout.addRow("Volatilité Utilisée (σ):", self.historical_vol_label)
        current_data_layout.addRow("Prix de l'option (BSM):", self.bs_price_label)
        current_data_group.setLayout(current_data_layout)
        display_panel_layout.addWidget(current_data_group)

        greeks_group = QGroupBox("Grecs (BSM)")
        greeks_table_layout = QGridLayout()
        self.greeks_table = QTableWidget(1, 5)
        self.greeks_table.setHorizontalHeaderLabels(["Delta", "Gamma", "Theta (par jour)", "Vega", "Rho"])
        self.greeks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.greeks_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.greeks_table.cellClicked.connect(self.handle_greek_click)

        for col in range(5):
            self.greeks_table.setItem(0, col, QTableWidgetItem("N/A"))

        greeks_table_layout.addWidget(self.greeks_table, 0, 0)
        greeks_group.setLayout(greeks_table_layout)
        display_panel_layout.addWidget(greeks_group)

        payoff_plot_group = QGroupBox("Payoff de l'option")
        payoff_plot_layout = QVBoxLayout()

        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        payoff_plot_layout.addWidget(self.canvas)
        payoff_plot_group.setLayout(payoff_plot_layout)
        display_panel_layout.addWidget(payoff_plot_group)

        option_calculator_layout.addLayout(display_panel_layout, 2)

        # Ajout des onglets dans l'ordre: BSM, CRR, Simulation, Smile
        self.tab_widget.addTab(option_calculator_widget, "Modèle BSM")

        self.crr_tab = CRRModelTab(self)
        self.tab_widget.addTab(self.crr_tab, "Modèle CRR")

        self.simulation_tab = CallPriceSimulationTab()
        self.tab_widget.addTab(self.simulation_tab, "Simulation")

        self.smile_tab = VolatilitySmileTab()
        self.tab_widget.addTab(self.smile_tab, "Smile de Volatilité")

        self.surface_tab = VolatilitySurfaceTab()
        self.tab_widget.addTab(self.surface_tab, "Surface IV")

        self.exotic_tab = ExoticOptionsTab(self)
        self.tab_widget.addTab(self.exotic_tab, "Exotiques")

        self.strategy_tab = StrategyTab(self)
        self.tab_widget.addTab(self.strategy_tab, "Stratégies")

        self.forecast_tab = ForecastTimesFMTab()
        self.tab_widget.addTab(self.forecast_tab, "Forecast TimesFM")

        main_window_layout = QVBoxLayout()
        main_window_layout.addWidget(self.tab_widget)
        self.setLayout(main_window_layout)

        # On appelle la méthode générique au démarrage
        self.fetch_data_for_tab(self.ticker_input.text(), self) 

        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def fetch_data_for_tab(self, ticker_symbol: str, source_tab: QWidget) -> None:
        """
        Récupère les données financières en parallèle avec threading.
        Optimisé pour éviter les freezes UI lors des appels API.
        """
        ticker_symbol = ticker_symbol.strip().upper()
        if not ticker_symbol:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer un symbole de ticker.")
            self.current_ticker = "N/A"
            self.S = None
            self.r = None
            self.q = None
            self.historical_vol = None
            self.update_all_tabs_financial_data()
            return
        
        self.current_ticker = ticker_symbol
        
        # Désactiver le bouton pendant le chargement
        if hasattr(source_tab, 'fetch_data_button'):
            source_tab.fetch_data_button.setEnabled(False)
            source_tab.fetch_data_button.setText("⏳ Chargement...")
        
        # Lancer un worker dans un QThread pour ne pas bloquer l'UI
        self._fetch_worker = FetchDataWorker(self.data_fetcher, ticker_symbol)
        self._fetch_worker.data_ready.connect(
            lambda tk, price, sofr, div, vol: self._on_fetch_done(tk, price, sofr, div, vol, source_tab)
        )
        self._fetch_worker.error.connect(lambda msg: self._on_fetch_error(msg, source_tab))
        self._fetch_worker.start()
        return

    def _on_fetch_done(self, ticker, live_price, sofr, dividend, vol, source_tab):
        # Traiter les résultats reçus depuis le thread worker (éviter 'or' sur des numériques)
        try:
            self.current_ticker = ticker

            self.S = live_price if live_price is not None else None

            r_result = sofr if sofr is not None else None
            self.r = r_result if r_result is not None else 0.01

            q_result = dividend if dividend is not None else None
            self.q = q_result if q_result is not None else 0.0

            vol_result = vol if vol is not None else None
            self.historical_vol = vol_result if vol_result is not None else 0.20

            if self.S is None:
                QMessageBox.warning(self, "Données Manquantes",
                                     f"Impossible de récupérer le prix de {ticker}. Les calculs suivants pourraient être inexacts.")

            if source_tab == self:
                self.ticker_input.setText(self.current_ticker)

            # Réactiver le bouton
            if hasattr(source_tab, 'fetch_data_button'):
                source_tab.fetch_data_button.setEnabled(True)
                source_tab.fetch_data_button.setText("Récupérer/Synchroniser les Données")

            # Nettoyage du worker
            self._fetch_worker = None

            self.update_all_tabs_financial_data(source_tab)
        except Exception as e:
            QMessageBox.critical(self, "Erreur Traitement Données", f"Erreur lors du traitement des données: {e}")

    def _on_fetch_error(self, msg, source_tab):
        # Afficher l'erreur et réactiver le bouton
        QMessageBox.critical(self, "Erreur de Récupération", f"Erreur lors de la récupération des données: {msg}")
        if hasattr(source_tab, 'fetch_data_button'):
            source_tab.fetch_data_button.setEnabled(True)
            source_tab.fetch_data_button.setText("Récupérer/Synchroniser les Données")
        self._fetch_worker = None

    def update_all_tabs_financial_data(self, source_tab=None):
        """Met à jour les labels d'information financière dans tous les onglets."""

        sigma_to_use = self.current_sigma if self.current_sigma is not None else (self.historical_vol if self.historical_vol is not None else 0.20)
        pricing_method_to_use = self.pricing_method if self.pricing_method != "N/A" else "Vol Historique"
        
        # 1. Onglet BSM (Calculateur d'Option)
        self.live_price_label.setText(f"{self.S:.2f}" if self.S is not None else "N/A")
        self.risk_free_rate_label.setText(f"{self.r*100:.2f}%" if self.r is not None else "N/A")
        self.dividend_yield_label.setText(f"{self.q*100:.2f}%" if self.q is not None else "N/A")
        self.historical_vol_label.setText(f"Historique: {(self.historical_vol or 0.0)*100:.2f}%")
        
        # 2. Onglet CRR (Modèle CRR)
        self.crr_tab.update_financial_data(self.S, self.r, self.q, sigma_to_use, self.current_ticker, pricing_method_to_use)
        
        # 3. Onglet Simulation
        self.simulation_tab.update_financial_data(self.current_ticker, self.S, self.r, self.q, sigma_to_use)
        
        # 4. Onglet Smile
        self.smile_tab.update_financial_params(self.r, self.q)
        self.smile_tab.update_S(self.S)
        if self.current_ticker and source_tab != self.smile_tab:
            self.smile_tab.ticker_input.setText(self.current_ticker)

        # 5. Onglet Options Exotiques
        self.exotic_tab.update_financial_data(
            self.current_ticker, self.S, self.r, self.q,
            sigma_to_use, pricing_method_to_use
        )

        # 6. Onglet Stratégies
        self.strategy_tab.update_financial_data(
            self.current_ticker, self.S, self.r, self.q,
            sigma_to_use, pricing_method_to_use
        )

        # 7. Onglet Forecast TimesFM
        if hasattr(self, 'forecast_tab'):
            self.forecast_tab.update_financial_params(
                self.current_ticker, self.S, self.r, self.q, sigma_to_use
            )
        

    def calculate_option_metrics(self):
        try:
            self.K = float(self.strike_input.text())
            self.option_type = self.option_type_combo.currentText() 

            maturity_qdate = self.maturity_date_input.date()
            maturity_datetime = datetime(maturity_qdate.year(), maturity_qdate.month(), maturity_qdate.day())
            
            # --- Vérifications préalables ---
            if self.S is None or self.r is None or self.q is None or self.current_ticker is None:
                QMessageBox.warning(self, "Données Manquantes", "Veuillez d'abord récupérer toutes les données de l'actif sous-jacent (S, r, q).")
                return
            if self.K <= 0:
                QMessageBox.warning(self, "Erreur de Strike", "Le prix d'exercice doit être supérieur à 0.")
                return

            # --- RÉCUPÉRATION DE L'IV ET DU PRIX MARCHÉ ---
            fetched_iv, market_price, closest_date = self.data_fetcher.get_implied_volatility_and_price(
                self.current_ticker, self.K, maturity_datetime, self.option_type 
            )
            
            # Mise à jour du temps T
            if closest_date:
                closest_date_obj = datetime.strptime(closest_date, '%Y-%m-%d').date()
                today = date.today()
                time_difference = closest_date_obj - today
                self.T = time_difference.days / 365.0
                if self.T < 0: self.T = 1e-6 
            else:
                # Calcul basé sur la date du calendrier si pas de chaîne d'options trouvée
                today = date.today()
                time_difference = maturity_datetime.date() - today
                self.T = time_difference.days / 365.0
                if self.T <= 0:
                    QMessageBox.warning(self, "Erreur de Maturité", "La date d'échéance doit être dans le futur.")
                    return

            # --- Logique de choix de la Volatilité (IV vs Historique) ---
            if fetched_iv is not None and fetched_iv > 0.001 and market_price is not None:
                sigma = fetched_iv
                self.pricing_method = "IV Marché"
                bs_price = market_price
                
            else:
                sigma = self.historical_vol if self.historical_vol is not None and self.historical_vol > 0 else 0.20
                self.pricing_method = "Vol Historique (Fallback)"
                
                bs_price = self.option_models.black_scholes_price(self.S, self.K, self.T, self.r, sigma, self.q, self.option_type)
                
                if self.historical_vol is None or self.historical_vol <= 0 or fetched_iv is None:
                     QMessageBox.information(self, "Volatilité",
                                         f"L'IV du marché n'est pas disponible. "
                                         f"Utilisation d'une volatilité ({sigma*100:.2f}%) pour les calculs.")

            self.current_sigma = sigma 
            
            # --- Mise à jour de l'affichage ---
            self.historical_vol_label.setText(f"Utilisée ({self.pricing_method}): {self.current_sigma*100:.2f}%")
            self.bs_price_label.setText(f"{bs_price:.4f} $")

            # Calcul des Grecs (BSM)
            greeks = self.option_models.calculate_greeks(
                self.S, self.K, self.T, self.r, self.current_sigma, self.q, self.option_type
            )

            # Mise à jour de la table des Grecs
            self.greeks_table.setItem(0, 0, QTableWidgetItem(f"{greeks.get('delta', 0):.4f}"))
            self.greeks_table.setItem(0, 1, QTableWidgetItem(f"{greeks.get('gamma', 0):.4f}"))
            self.greeks_table.setItem(0, 2, QTableWidgetItem(f"{greeks.get('theta', 0):.4f}"))
            self.greeks_table.setItem(0, 3, QTableWidgetItem(f"{greeks.get('vega', 0)/100:.4f}")) 
            self.greeks_table.setItem(0, 4, QTableWidgetItem(f"{greeks.get('rho', 0):.4f}"))

            # Mettre à jour les données pour les autres onglets
            # On utilise le `self.pricing_method` mis à jour pour les autres onglets
            self.simulation_tab.update_financial_data(self.current_ticker, self.S, self.r, self.q, self.current_sigma)
            self.crr_tab.update_financial_data(self.S, self.r, self.q, self.current_sigma, self.current_ticker, self.pricing_method) 
        
        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie", "Veuillez entrer des valeurs numériques valides pour K.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur de Calcul", f"Une erreur inattendue est survenue: {e}")

    def calculate_crr_metrics(self):
        try:
            # Récupération des inputs depuis l'onglet CRR
            K = float(self.crr_tab.strike_input.text())
            option_type = self.crr_tab.option_type_combo.currentText() 
            N = int(self.crr_tab.steps_input.text())

            maturity_qdate = self.crr_tab.maturity_date_input.date()
            maturity_datetime = datetime(maturity_qdate.year(), maturity_qdate.month(), maturity_qdate.day())

            # Vérifications des données
            if self.S is None or self.r is None or self.q is None or self.current_ticker is None:
                QMessageBox.warning(self, "Données Manquantes", "Veuillez Récupérer/Synchroniser les données de l'actif (S, r, q) d'abord.")
                return
            if K <= 0 or N <= 0 or N > 10000:
                QMessageBox.warning(self, "Erreur de Paramètres", "K doit être > 0, et le nombre de pas (N) doit être entre 1 et 10000.")
                return

            # Calcul du Temps T (basé sur la date du CRR)
            today = date.today()
            time_difference = maturity_datetime.date() - today
            T = time_difference.days / 365.0
            if T <= 0:
                QMessageBox.warning(self, "Erreur de Maturité", "La date d'échéance doit être dans le futur.")
                return
            
            # --- RÉCUPÉRATION DE L'IV ET DU PRIX MARCHÉ POUR LES PARAMÈTRES CRR ---
            fetched_iv, market_price, closest_date = self.data_fetcher.get_implied_volatility_and_price(
                self.current_ticker, K, maturity_datetime, option_type
            )
            
            # --- Logique de choix de la Volatilité (IV vs Historique) ---
            if fetched_iv is not None and fetched_iv > 0.001 and market_price is not None:
                sigma = fetched_iv
                pricing_method_used = "IV Marché"
            else:
                sigma = self.historical_vol if self.historical_vol is not None and self.historical_vol > 0 else 0.20
                pricing_method_used = "Vol Historique (Fallback)"
            
            # Mise à jour explicite de la volatilité utilisée dans l'affichage CRR
            self.crr_tab.vol_label.setText(f"{sigma*100:.2f}% ({pricing_method_used})")

            # Calcul du prix CRR
            crr_price = self.option_models.cox_ross_rubinstein_price(
                self.S, K, T, self.r, self.q, sigma, N, option_type
            )
            self.crr_tab.crr_price_label.setText(f"{crr_price:.4f} $")

            # Calcul des Grecs CRR
            greeks = self.option_models.calculate_greeks_crr(
                self.S, K, T, self.r, self.q, sigma, N, option_type
            )

            # Mise à jour de la table des Grecs CRR
            self.crr_tab.greeks_table.setItem(0, 0, QTableWidgetItem(f"{greeks.get('delta', 0):.4f}"))
            self.crr_tab.greeks_table.setItem(0, 1, QTableWidgetItem(f"{greeks.get('gamma', 0):.4f}"))
            self.crr_tab.greeks_table.setItem(0, 2, QTableWidgetItem(f"{greeks.get('theta', 0):.4f}"))
            self.crr_tab.greeks_table.setItem(0, 3, QTableWidgetItem(f"{greeks.get('vega', 0)/100:.4f}"))
            self.crr_tab.greeks_table.setItem(0, 4, QTableWidgetItem(f"{greeks.get('rho', 0):.4f}"))

        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie", "Veuillez entrer des valeurs numériques/entières valides pour K et N.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur de Calcul CRR", f"Une erreur inattendue est survenue: {e}")

    def _draw_payoff(self, ax, K, premium, option_type, position):
        S_min = max(0, K * 0.7)
        S_max = K * 1.3
        S_range = np.linspace(S_min, S_max, 200)
        payoff = self.strategy_manager.calculate_single_option_payoff(S_range, K, premium, option_type, position)
        ax.plot(S_range, payoff, label=f'{position.capitalize()} {option_type.capitalize()} (K={K})')
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(K, color='grey', linestyle=':', linewidth=0.8, label=f'Strike K={K}')
        ax.set_xlabel("Prix de l'actif sous-jacent à l'échéance (S)")
        ax.set_ylabel("Profit/Perte")
        ax.grid(True)
        ax.legend()

    def plot_crr_payoff(self):
        try:
            K = float(self.crr_tab.strike_input.text())
            option_type = self.crr_tab.option_type_combo.currentText()
            position = self.crr_tab.position_combo.currentText()

            price_str = self.crr_tab.crr_price_label.text()
            if "N/A" in price_str:
                QMessageBox.warning(self, "Prix CRR Manquant",
                                    "Veuillez d'abord calculer le prix CRR avant de tracer le payoff.")
                return

            premium = float(price_str.replace('$', '').strip())
            
            if K <= 0:
                QMessageBox.warning(self, "Erreur de Strike", "Le prix d'exercice doit être supérieur à 0.")
                return

            if option_type == "call":
                breakeven = K + premium
            elif option_type == "put":
                breakeven = K - premium
            
            self.crr_tab.fig.clear()
            ax = self.crr_tab.fig.add_subplot(111)

            self._draw_payoff(ax, K, premium, option_type, position)

            title_text = f"Payoff de l'Option Américaine {position.capitalize()} {option_type.capitalize()} (K={K:.2f}, Premium={premium:.4f})"
            title_text += f"\nBreakeven = {breakeven:.2f}"
            
            ax.set_title(title_text)
            self.crr_tab.canvas.draw()

        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie", "Veuillez entrer des valeurs numériques valides.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur de Tracé", f"Une erreur est survenue lors du tracé du payoff: {e}")




    def handle_greek_click(self, row, column):
        greek_names = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
        if column < len(greek_names):
            greek_name = greek_names[column]
            self.plot_greek_evolution(greek_name)
    
    def plot_greek_evolution(self, greek_name):
        if self.S is None or self.K is None or self.T is None or \
           self.r is None or self.q is None or self.current_sigma is None or \
           self.option_type is None:
            QMessageBox.warning(self, "Données Manquantes",
                                 f"Veuillez d'abord calculer les métriques de l'option (Prix, Grecs) pour pouvoir tracer l'évolution du {greek_name}.")
            return

        S_range = np.linspace(self.S * 0.7, self.S * 1.3, 100)
        greek_values = []
        for s_val in S_range:
            greeks = self.option_models.calculate_greeks(
                S=float(s_val), K=self.K, T=self.T, r=self.r, sigma=self.current_sigma, q=self.q, option_type=self.option_type
            )
            
            if greek_name == "Delta":
                value = greeks.get('delta', 0)
            elif greek_name == "Gamma":
                value = greeks.get('gamma', 0)
            elif greek_name == "Theta":
                value = greeks.get('theta', 0)
            elif greek_name == "Vega":
                value = greeks.get('vega', 0) / 100 
            elif greek_name == "Rho":
                value = greeks.get('rho', 0)
            else:
                value = 0
            
            greek_values.append(value)

        dialog = PlottingDialog(self, title=f"Évolution du {greek_name} (BSM)")
        ax = dialog.fig.add_subplot(111)
        ax.plot(S_range, greek_values)
        ax.axvline(self.S, color='r', linestyle='--', label=f'S₀ Actuel: {self.S:.2f}')
        ax.set_title(f'Évolution du {greek_name} en fonction du prix du sous-jacent S₀')
        ax.set_xlabel('Prix du Sous-jacent (S₀)')
        ax.set_ylabel(greek_name)
        ax.grid(True)
        ax.legend()
        dialog.canvas.draw()
        dialog.exec_()


    def plot_option_payoff(self):
        try:
            K = float(self.strike_input.text())
            option_type = self.option_type_combo.currentText()
            position = self.position_combo.currentText()

            bs_price_str = self.bs_price_label.text()
            if "N/A" in bs_price_str:
                QMessageBox.warning(self, "Prix BSM Manquant",
                                    "Veuillez d'abord calculer le prix Black-Scholes avant de tracer le payoff. ")
                return

            premium = float(bs_price_str.replace('$', '').strip())

            if premium <= 0 and position == 'long':
                 QMessageBox.information(self, "Premium Nul/Négatif",
                                     "Le prix Black-Scholes calculé (premium) est nul ou négatif pour un achat. ")

            if K <= 0:
                QMessageBox.warning(self, "Erreur de Strike", "Le prix d'exercice doit être supérieur à 0.")
                return

            breakeven = 0.0
            if option_type == "call":
                if position == "long":
                    breakeven = K + premium
                elif position == "short":
                    breakeven = K - premium
            elif option_type == "put":
                if position == "long":
                    breakeven = K - premium
                elif position == "short":
                    breakeven = K + premium 
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            self._draw_payoff(ax, K, premium, option_type, position)

            title_text = f"Payoff de l'Option Européenne {position.capitalize()} {option_type.capitalize()} (K={K:.2f}, Premium={premium:.4f})"
            if breakeven is not None:
                title_text += f"\nBreakeven = {breakeven:.2f}"
            
            ax.set_title(title_text)
            self.canvas.draw()

        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie", "Veuillez entrer des valeurs numériques valides pour K.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur de Tracé", f"Une erreur est survenue lors du tracé du payoff: {e}")

    def handle_crr_greek_click(self, row: int, column: int) -> None:
        """Gère le click sur un Grec du tableau CRR pour afficher son évolution."""
        greek_names = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
        if column < len(greek_names):
            greek_name = greek_names[column]
            self.plot_crr_greek_evolution(greek_name)

    def plot_crr_greek_evolution(self, greek_name: str) -> None:
        """Trace l'évolution d'un Grec du modèle CRR en fonction du prix de l'actif."""
        try:
            if self.S is None or self.r is None or self.q is None or self.current_sigma is None:
                QMessageBox.warning(self, "Données Manquantes", "Veuillez d'abord calculer les Grecs CRR.")
                return

            K = float(self.crr_tab.strike_input.text())
            option_type = self.crr_tab.option_type_combo.currentText()
            N = int(self.crr_tab.steps_input.text())

            maturity_qdate = self.crr_tab.maturity_date_input.date()
            maturity_datetime = datetime(maturity_qdate.year(), maturity_qdate.month(), maturity_qdate.day())
            today = date.today()
            T = (maturity_datetime.date() - today).days / 365.0

            # Plage de prix autour de S actuel
            S_range = np.linspace(self.S * 0.7, self.S * 1.3, 50)
            greek_values = []

            for S in S_range:
                greeks = self.option_models.calculate_greeks_crr(
                    S, K, T, self.r, self.q, self.current_sigma, N, option_type
                )
                
                if greek_name == "Delta":
                    greek_values.append(greeks['delta'])
                elif greek_name == "Gamma":
                    greek_values.append(greeks['gamma'])
                elif greek_name == "Theta":
                    greek_values.append(greeks['theta'])
                elif greek_name == "Vega":
                    greek_values.append(greeks['vega'] / 100)
                elif greek_name == "Rho":
                    greek_values.append(greeks['rho'])

            dialog = PlottingDialog(self, f"Évolution du {greek_name} (CRR)")
            ax = dialog.fig.add_subplot(111)
            ax.plot(S_range, greek_values, linewidth=2, color='steelblue')
            ax.axvline(self.S, color='red', linestyle='--', label=f'Prix actuel S={self.S:.2f}')
            ax.set_xlabel("Prix de l'actif sous-jacent (S)")
            ax.set_ylabel(greek_name)
            ax.set_title(f"Évolution du {greek_name} - Modèle CRR")
            ax.grid(True, alpha=0.3)
            ax.legend()
            dialog.fig.tight_layout()
            dialog.canvas.draw()
            dialog.exec_()

        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie", "Veuillez entrer des valeurs numériques valides.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Une erreur est survenue: {e}")

    def on_tab_changed(self, index: int) -> None:
        """Synchronisation des données entre l'onglet BSM et les autres onglets."""
        # Synchronisation des données entre l'onglet BSM et les autres onglets
        if self.S is not None: # Ne synchronise que si les données ont été chargées au moins une fois
            sigma_to_use = self.current_sigma if self.current_sigma is not None else (self.historical_vol if self.historical_vol is not None else 0.20)
            pricing_method_to_use = self.pricing_method if self.pricing_method != "N/A" else "Vol Historique"

            if index == self.tab_widget.indexOf(self.crr_tab):
                self.crr_tab.update_financial_data(self.S, self.r, self.q, sigma_to_use, self.current_ticker, pricing_method_to_use)
            
            elif index == self.tab_widget.indexOf(self.simulation_tab):
                self.simulation_tab.update_financial_data(self.current_ticker, self.S, self.r, self.q, sigma_to_use)
            
            elif index == self.tab_widget.indexOf(self.smile_tab):
                self.smile_tab.ticker_input.setText(self.current_ticker)
            
            elif index == self.tab_widget.indexOf(self.exotic_tab):
                self.exotic_tab.update_financial_data(
                    self.current_ticker, self.S, self.r, self.q,
                    sigma_to_use, pricing_method_to_use
                )

            elif index == self.tab_widget.indexOf(self.strategy_tab):
                self.strategy_tab.update_financial_data(
                    self.current_ticker, self.S, self.r, self.q,
                    sigma_to_use, pricing_method_to_use
                )

            elif index == self.tab_widget.indexOf(getattr(self, 'forecast_tab', None)):
                self.forecast_tab.update_financial_params(
                    self.current_ticker, self.S, self.r, self.q,
                    sigma_to_use
                )
            
        else:
            QMessageBox.information(self, "Données Manquantes",
                                    "Veuillez récupérer les données financières (Ticker, Prix Actuel, Taux Sans Risque, Dividende, Volatilité) dans l'onglet 'Calculateur d'Option' d'abord.")
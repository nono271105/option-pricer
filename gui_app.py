"""
gui_app.py — Orchestrateur pur.
Instancie le MarketDataStore, les 8 tabs UI, et le QTabWidget.
Toute la logique métier est dans logic/, toute l'UI dans UI/.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QMessageBox,
)
from PySide6.QtCore import QThread, Signal

from data_fetcher import DataFetcher
from market_data_store import MarketDataStore

from UI.bsm_ui import BSMTab
from UI.crr_ui import CRRModelTab
from UI.simulation_ui import CallPriceSimulationTab
from UI.volatility_smile_ui import VolatilitySmileTab
from UI.volatility_surface_ui import VolatilitySurfaceTab
from UI.exotic_options_ui import ExoticOptionsTab
from UI.strategy_ui import StrategyTab
from UI.forecast_ui import ForecastTimesFMTab


class FetchDataWorker(QThread):
    data_ready = Signal(str, object, object, object, object, str)
    error = Signal(str)

    def __init__(self, data_fetcher, ticker_symbol):
        super().__init__()
        self.data_fetcher = data_fetcher
        self.ticker_symbol = ticker_symbol

    def run(self) -> None:
        """Exécute la récupération des données en arrière-plan."""
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=5) as executor:
                price_future = executor.submit(self.data_fetcher.get_live_price, self.ticker_symbol)
                sofr_future = executor.submit(self.data_fetcher.get_sofr_rate)
                dividend_future = executor.submit(self.data_fetcher.get_dividend_yield, self.ticker_symbol)
                volatility_future = executor.submit(self.data_fetcher.get_historical_volatility, self.ticker_symbol, "1y")
                company_name_future = executor.submit(self.data_fetcher.get_company_name, self.ticker_symbol)
                live_price = price_future.result(timeout=10)
                sofr = sofr_future.result(timeout=10)
                dividend = dividend_future.result(timeout=10)
                volatility = volatility_future.result(timeout=10)
                company_name = company_name_future.result(timeout=10)
            self.data_ready.emit(self.ticker_symbol, live_price, sofr, dividend, volatility, company_name or self.ticker_symbol)
        except Exception as e:
            self.error.emit(str(e))


class OptionPricingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Option Pricer")
        self.setGeometry(100, 100, 1400, 900)

        self.data_fetcher = DataFetcher()
        self.store = MarketDataStore()
        self._fetch_worker = None

        self.init_ui()

    def init_ui(self) -> None:
        """Initialise l'interface utilisateur principale."""
        self.tab_widget = QTabWidget()

        # Instancier les 8 tabs avec le store et la callback fetch
        self.bsm_tab = BSMTab(self.store, fetch_fn=self.fetch_data_for_tab)
        self.crr_tab = CRRModelTab(self.store, fetch_fn=self.fetch_data_for_tab)
        self.simulation_tab = CallPriceSimulationTab()
        self.smile_tab = VolatilitySmileTab()
        self.surface_tab = VolatilitySurfaceTab()
        self.exotic_tab = ExoticOptionsTab(self.store, fetch_fn=self.fetch_data_for_tab)
        self.strategy_tab = StrategyTab(self.store, fetch_fn=self.fetch_data_for_tab)
        self.forecast_tab = ForecastTimesFMTab()

        # Ajouter les tabs au widget
        self.tab_widget.addTab(self.bsm_tab, "Modèle BSM")
        self.tab_widget.addTab(self.crr_tab, "Modèle CRR")
        self.tab_widget.addTab(self.simulation_tab, "Simulation")
        self.tab_widget.addTab(self.smile_tab, "Smile de Volatilité")
        self.tab_widget.addTab(self.surface_tab, "Surface IV")
        self.tab_widget.addTab(self.exotic_tab, "Exotiques")
        self.tab_widget.addTab(self.strategy_tab, "Stratégies")
        self.tab_widget.addTab(self.forecast_tab, "Forecast TimesFM")

        main_window_layout = QVBoxLayout()
        main_window_layout.addWidget(self.tab_widget)
        self.setLayout(main_window_layout)

        # Fetch initial
        self.fetch_data_for_tab("AAPL", self.bsm_tab)

    def fetch_data_for_tab(self, ticker_symbol: str, source_tab: QWidget) -> None:
        """
        Récupère les données financières en parallèle avec threading.
        """
        ticker_symbol = ticker_symbol.strip().upper()
        if not ticker_symbol:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer un symbole de ticker.")
            return

        # Désactiver le bouton pendant le chargement
        if hasattr(source_tab, 'fetch_data_button'):
            source_tab.fetch_data_button.setEnabled(False)
            source_tab.fetch_data_button.setText("⏳ Chargement...")

        # Lancer un worker dans un QThread
        self._fetch_worker = FetchDataWorker(self.data_fetcher, ticker_symbol)
        self._fetch_worker.data_ready.connect(
            lambda tk, price, sofr, div, vol, company_name: self._on_fetch_done(
                tk, price, sofr, div, vol, company_name, source_tab)
        )
        self._fetch_worker.error.connect(lambda msg: self._on_fetch_error(msg, source_tab))
        self._fetch_worker.start()

    def _on_fetch_done(self, ticker: str, live_price, sofr, dividend, volatility, company_name, source_tab) -> None:
        """Callback — met à jour le store, qui notifie automatiquement tous les tabs."""
        try:
            S = live_price if live_price is not None else None
            r = sofr if sofr is not None else 0.01
            q = dividend if dividend is not None else 0.0
            hist_vol = volatility if volatility is not None else 0.20

            if S is None:
                QMessageBox.warning(self, "Données Manquantes",
                    f"Impossible de récupérer le prix de {ticker}.")

            # Le store notifie automatiquement les 8 tabs
            self.store.update(
                ticker=ticker,
                S=S,
                r=r,
                q=q,
                historical_vol=hist_vol,
                sigma=hist_vol,
                company_name=company_name,
                pricing_method="Vol Historique",
            )

            # Les tabs qui ont update_financial_data/update_financial_params
            # mais pas encore le store (simulation, smile, surface, forecast)
            sigma_to_use = hist_vol
            pricing_method = "Vol Historique"

            self.simulation_tab.update_financial_data(ticker, S, r, q, sigma_to_use)
            self.simulation_tab.update_company_name(company_name or ticker)

            self.smile_tab.update_financial_params(r, q)
            self.smile_tab.update_S(S)
            self.smile_tab.update_company_name(company_name or ticker)
            if ticker:
                self.smile_tab.ticker_input.setText(ticker)

            self.surface_tab.update_financial_params(ticker, S, r, q)
            self.surface_tab.update_company_name(company_name or ticker)

            if hasattr(self, 'forecast_tab'):
                self.forecast_tab.update_financial_params(ticker, S, r, q, sigma_to_use)
                self.forecast_tab.update_company_name(company_name or ticker)

            # Réactiver le bouton du tab source
            if hasattr(source_tab, 'fetch_data_button'):
                source_tab.fetch_data_button.setEnabled(True)
                source_tab.fetch_data_button.setText("Récupérer/Synchroniser les Données")

            self._fetch_worker = None

        except Exception as e:
            QMessageBox.critical(self, "Erreur Traitement Données",
                f"Erreur lors du traitement des données: {e}")

    def _on_fetch_error(self, msg, source_tab):
        """Affiche l'erreur et réactive le bouton."""
        QMessageBox.critical(self, "Erreur de Récupération",
            f"Erreur lors de la récupération des données: {msg}")
        if hasattr(source_tab, 'fetch_data_button'):
            source_tab.fetch_data_button.setEnabled(True)
            source_tab.fetch_data_button.setText("Récupérer/Synchroniser les Données")
        self._fetch_worker = None
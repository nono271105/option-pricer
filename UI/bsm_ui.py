"""
UI/bsm_ui.py — Onglet BSM (Black-Scholes-Merton)
Extrait de gui_app.py : panneau contrôle BSM + panneau affichage BSM
+ calculate_option_metrics, plot_option_payoff, handle_greek_click, plot_greek_evolution
"""

from __future__ import annotations

import numpy as np
from datetime import date, datetime
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QFormLayout, QGroupBox, QGridLayout,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QDateEdit,
    QDialog, QAbstractItemView,
)
from PySide6.QtCore import QDate
from PySide6.QtGui import QDoubleValidator

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data_fetcher import DataFetcher
from logic.bsm_logic import OptionModels
from logic.strategy_logic import StrategyManager


class PlottingDialog(QDialog):
    """Fenêtre de dialogue pour les graphiques Matplotlib."""
    def __init__(self, parent=None, title="Graphique"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(150, 150, 700, 500)
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)


class BSMTab(QWidget):
    """
    Onglet Black-Scholes-Merton.
    Panneau gauche  : paramètres (ticker, K, T, position, boutons)
    Panneau droit   : données actuelles + grecs + payoff Matplotlib
    """

    def __init__(self, store, fetch_fn, parent=None):
        super().__init__(parent)
        self.store = store
        self._fetch_fn = fetch_fn
        self.data_fetcher = DataFetcher()
        self.option_models = OptionModels()
        self.strategy_manager = StrategyManager()

        # État local pour les calculs
        self.S = None
        self.r = None
        self.q = None
        self.K = None
        self.T = None
        self.option_type = None
        self.current_sigma = None
        self.historical_vol = None
        self.current_ticker = None
        self.pricing_method = "N/A"

        self._build_ui()
        store.subscribe(self.on_market_update)

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Panneau de contrôle gauche ---
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

        from utils import get_default_maturity_date
        self.maturity_date_input = QDateEdit(get_default_maturity_date())
        self.maturity_date_input.setCalendarPopup(True)
        self.maturity_date_input.setDisplayFormat("dd/MM/yyyy")
        control_form_layout.addRow("Date d'échéance:", self.maturity_date_input)

        self.position_combo = QComboBox()
        self.position_combo.addItems(["long", "short"])
        control_form_layout.addRow("Position:", self.position_combo)

        self.fetch_data_button = QPushButton("Récupérer les Données")
        self.fetch_data_button.clicked.connect(self._fetch_data)
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
        main_layout.addLayout(control_panel_layout, 1)

        # --- Panneau d'affichage droite ---
        display_panel_layout = QVBoxLayout()

        current_data_group = QGroupBox("Données Actuelles")
        current_data_layout = QFormLayout()
        self.company_name_label = QLabel("N/A")
        current_data_layout.addRow("Entreprise:", self.company_name_label)
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
        self.greeks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.greeks_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
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

        main_layout.addLayout(display_panel_layout, 2)

    # =========================================================================
    # Fetch data
    # =========================================================================

    def _fetch_data(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer un symbole de ticker.")
            return
        self._fetch_fn(ticker, self)

    # =========================================================================
    # MarketDataStore — pub/sub
    # =========================================================================

    def on_market_update(self, store) -> None:
        """Appelé automatiquement quand le store est mis à jour."""
        self.S = store.S
        self.r = store.r
        self.q = store.q
        self.historical_vol = store.historical_vol
        self.current_ticker = store.ticker

        self.company_name_label.setText(store.company_name or store.ticker or "N/A")
        self.live_price_label.setText(f"{store.S:.2f}" if store.S is not None else "N/A")
        self.risk_free_rate_label.setText(f"{store.r*100:.2f}%" if store.r is not None else "N/A")
        self.dividend_yield_label.setText(f"{store.q*100:.2f}%" if store.q is not None else "N/A")

        sigma_to_use = store.sigma if store.sigma is not None else (store.historical_vol if store.historical_vol is not None else 0.20)
        pm = store.pricing_method if store.pricing_method != "N/A" else "Vol Historique"
        self.historical_vol_label.setText(f"{pm}: {sigma_to_use*100:.2f}%")

        if store.ticker:
            self.ticker_input.setText(store.ticker)

        self.fetch_data_button.setEnabled(True)
        self.fetch_data_button.setText("Récupérer les Données")

    # =========================================================================
    # calculate_option_metrics — inchangé algorithmiquement
    # =========================================================================

    def calculate_option_metrics(self):
        try:
            self.K = float(self.strike_input.text())
            self.option_type = self.option_type_combo.currentText()

            maturity_qdate = self.maturity_date_input.date()
            maturity_datetime = datetime(maturity_qdate.year(), maturity_qdate.month(), maturity_qdate.day())

            if self.S is None or self.r is None or self.q is None or self.current_ticker is None:
                QMessageBox.warning(self, "Données Manquantes",
                    "Veuillez d'abord récupérer toutes les données de l'actif sous-jacent (S, r, q).")
                return
            if self.K <= 0:
                QMessageBox.warning(self, "Erreur de Strike", "Le prix d'exercice doit être supérieur à 0.")
                return

            fetched_iv, market_price, closest_date = self.data_fetcher.get_implied_volatility_and_price(
                self.current_ticker, self.K, maturity_datetime, self.option_type
            )

            if closest_date:
                closest_date_obj = datetime.strptime(closest_date, '%Y-%m-%d').date()
                today = date.today()
                time_difference = closest_date_obj - today
                self.T = time_difference.days / 365.0
                if self.T < 0:
                    self.T = 1e-6
            else:
                today = date.today()
                time_difference = maturity_datetime.date() - today
                self.T = time_difference.days / 365.0
                if self.T <= 0:
                    QMessageBox.warning(self, "Erreur de Maturité", "La date d'échéance doit être dans le futur.")
                    return

            if fetched_iv is not None and fetched_iv > 0.001 and market_price is not None:
                sigma = fetched_iv
                self.pricing_method = "IV Marché"
            else:
                sigma = self.historical_vol if self.historical_vol is not None and self.historical_vol > 0 else 0.20
                self.pricing_method = "Vol Historique (Fallback)"
                if self.historical_vol is None or self.historical_vol <= 0 or fetched_iv is None:
                    QMessageBox.information(self, "Volatilité",
                        f"L'IV du marché n'est pas disponible. "
                        f"Utilisation d'une volatilité ({sigma*100:.2f}%) pour les calculs.")

            bs_price = self.option_models.black_scholes_price(
                self.S, self.K, self.T, self.r, sigma, self.q, self.option_type
            )

            self.current_sigma = sigma

            self.historical_vol_label.setText(f"Utilisée ({self.pricing_method}): {self.current_sigma*100:.2f}%")
            self.bs_price_label.setText(f"{bs_price:.4f} $")

            greeks = self.option_models.calculate_greeks(
                self.S, self.K, self.T, self.r, self.current_sigma, self.q, self.option_type
            )

            self.greeks_table.setItem(0, 0, QTableWidgetItem(f"{greeks.get('delta', 0):.4f}"))
            self.greeks_table.setItem(0, 1, QTableWidgetItem(f"{greeks.get('gamma', 0):.4f}"))
            self.greeks_table.setItem(0, 2, QTableWidgetItem(f"{greeks.get('theta', 0):.4f}"))
            self.greeks_table.setItem(0, 3, QTableWidgetItem(f"{greeks.get('vega', 0)/100:.4f}"))
            self.greeks_table.setItem(0, 4, QTableWidgetItem(f"{greeks.get('rho', 0):.4f}"))

            # Mettre à jour le store avec la nouvelle volatilité
            self.store.update(sigma=sigma, pricing_method=self.pricing_method)

        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie", "Veuillez entrer des valeurs numériques valides pour K.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur de Calcul", f"Une erreur inattendue est survenue: {e}")

    # =========================================================================
    # _draw_payoff — utilitaire partagé (copié dans CRRModelTab aussi)
    # =========================================================================

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

    # =========================================================================
    # plot_option_payoff
    # =========================================================================

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
            title_text += f"\nBreakeven = {breakeven:.2f}"
            ax.set_title(title_text)
            self.canvas.draw()

        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie", "Veuillez entrer des valeurs numériques valides pour K.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur de Tracé", f"Une erreur est survenue lors du tracé du payoff: {e}")

    # =========================================================================
    # Greek evolution
    # =========================================================================

    def handle_greek_click(self, row, column):
        greek_names = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
        if column < len(greek_names):
            self.plot_greek_evolution(greek_names[column])

    def plot_greek_evolution(self, greek_name):
        if self.S is None or self.K is None or self.T is None or \
           self.r is None or self.q is None or self.current_sigma is None or \
           self.option_type is None:
            QMessageBox.warning(self, "Données Manquantes",
                f"Veuillez d'abord calculer les métriques de l'option pour tracer l'évolution du {greek_name}.")
            return

        S_range = np.linspace(self.S * 0.7, self.S * 1.3, 100)
        greek_values = []
        for s_val in S_range:
            greeks = self.option_models.calculate_greeks(
                S=float(s_val), K=self.K, T=self.T, r=self.r,
                sigma=self.current_sigma, q=self.q, option_type=self.option_type
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
        dialog.exec()

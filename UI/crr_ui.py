# Interface de valorisation des options américaines par arbre binomial

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
from PySide6.QtGui import QDoubleValidator, QIntValidator

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data_fetcher import DataFetcher
from logic.crr_logic import CRRModels
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


class CRRModelTab(QWidget):
    """
    Onglet pour le calcul du prix et des Grecs des options Américaines (CRR).
    """

    def __init__(self, store, fetch_fn, parent=None):
        super().__init__(parent)
        self.store = store
        self._fetch_fn = fetch_fn
        self.data_fetcher = DataFetcher()
        self.crr_models = CRRModels()
        self.strategy_manager = StrategyManager()

        # état interne du composant pour la persistance des paramètres
        self.S = None
        self.r = None
        self.q = None
        self.current_sigma = None
        self.historical_vol = None
        self.current_ticker = None

        self._build_ui()
        store.subscribe(self.on_market_update)

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # construction de la barre d'outils latérale
        control_panel_layout = QVBoxLayout()
        control_panel_group = QGroupBox("Paramètres de l'option (CRR)")
        control_form_layout = QFormLayout()

        self.ticker_input = QLineEdit()
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
        self.maturity_date_input.setToolTip("Date théorique libre")
        control_form_layout.addRow("Date d'échéance:", self.maturity_date_input)

        self.position_combo = QComboBox()
        self.position_combo.addItems(["long", "short"])
        control_form_layout.addRow("Position:", self.position_combo)

        self.steps_input = QLineEdit("100")
        self.steps_input.setValidator(QIntValidator(1, 10000))
        control_form_layout.addRow("Nombre de pas (N):", self.steps_input)

        self.fetch_data_button = QPushButton("Récupérer/Synchroniser les Données")
        self.fetch_data_button.clicked.connect(self._fetch_data)
        control_form_layout.addRow(self.fetch_data_button)

        self.calculate_button = QPushButton("Calculer Prix et Grecs (CRR)")
        self.calculate_button.clicked.connect(self.calculate_crr_metrics)
        control_form_layout.addRow(self.calculate_button)

        self.plot_payoff_button = QPushButton("Tracer le Payoff")
        self.plot_payoff_button.clicked.connect(self.plot_crr_payoff)
        control_form_layout.addRow(self.plot_payoff_button)

        control_panel_group.setLayout(control_form_layout)
        control_panel_layout.addWidget(control_panel_group)
        control_panel_layout.addStretch(1)
        main_layout.addLayout(control_panel_layout, 1)

        # construction de la zone de visualisation principale
        display_panel_layout = QVBoxLayout()

        current_data_group = QGroupBox("Données Actuelles")
        current_data_layout = QFormLayout()
        self.company_name_label = QLabel("N/A")
        self.live_price_label = QLabel("N/A")
        self.risk_free_rate_label = QLabel("N/A")
        self.dividend_yield_label = QLabel("N/A")
        self.volatility_label = QLabel("N/A")
        self.crr_price_label = QLabel("N/A")

        current_data_layout.addRow("Entreprise:", self.company_name_label)
        current_data_layout.addRow("Prix Actuel (S):", self.live_price_label)
        current_data_layout.addRow("Taux Sans Risque SOFR (r):", self.risk_free_rate_label)
        current_data_layout.addRow("Rendement Dividende (q):", self.dividend_yield_label)
        current_data_layout.addRow("Volatilité (σ):", self.volatility_label)
        current_data_layout.addRow("Prix de l'option (CRR):", self.crr_price_label)
        current_data_group.setLayout(current_data_layout)
        display_panel_layout.addWidget(current_data_group)

        greeks_group = QGroupBox("Grecs (CRR)")
        greeks_table_layout = QGridLayout()
        self.greeks_table = QTableWidget(1, 5)
        self.greeks_table.setHorizontalHeaderLabels(["Delta", "Gamma", "Theta (par jour)", "Vega", "Rho"])
        self.greeks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.greeks_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.greeks_table.cellClicked.connect(self.handle_crr_greek_click)
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

    # récupération asynchrone des conditions de marché

    def _fetch_data(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer un symbole de ticker.")
            return
        self._fetch_fn(ticker, self)

    # synchronisation réactive avec le bus de données

    def on_market_update(self, store) -> None:
        """Appelé automatiquement quand le store est mis à jour."""
        self.S = store.S
        self.r = store.r
        self.q = store.q
        self.historical_vol = store.historical_vol
        self.current_ticker = store.ticker
        self.current_sigma = store.sigma

        if self.ticker_input.text() == "":
            self.ticker_input.setText(store.ticker or "")

        self.company_name_label.setText(store.company_name or store.ticker or "N/A")
        self.live_price_label.setText(f"{store.S:.2f}" if store.S is not None else "N/A")
        self.risk_free_rate_label.setText(f"{store.r*100:.2f}%" if store.r is not None else "N/A")
        self.dividend_yield_label.setText(f"{store.q*100:.2f}%" if store.q is not None else "N/A")

        if store.sigma is not None:
            suffix = " (IV)" if "IV" in (store.pricing_method or "") else " (historique)"
            self.volatility_label.setText(f"{store.sigma*100:.2f}%{suffix}")
        else:
            self.volatility_label.setText("NC")

        self.fetch_data_button.setEnabled(True)
        self.fetch_data_button.setText("Récupérer/Synchroniser les Données")

    # évaluation de la prime américaine et extraction des grecs discrets

    def calculate_crr_metrics(self):
        try:
            K = float(self.strike_input.text())
            option_type = self.option_type_combo.currentText()
            N = int(self.steps_input.text())

            maturity_qdate = self.maturity_date_input.date()
            maturity_datetime = datetime(maturity_qdate.year(), maturity_qdate.month(), maturity_qdate.day())

            if self.S is None or self.r is None or self.q is None or self.current_ticker is None:
                QMessageBox.warning(self, "Données Manquantes",
                    "Veuillez Récupérer/Synchroniser les données de l'actif (S, r, q) d'abord.")
                return
            if K <= 0 or N < 3 or N > 10000:
                QMessageBox.warning(self, "Erreur de Paramètres",
                    "K doit être > 0, et le nombre de pas (N) doit être entre 3 et 10000.")
                return

            today = date.today()
            time_difference = maturity_datetime.date() - today
            T = time_difference.days / 365.0
            if T <= 0:
                QMessageBox.warning(self, "Erreur de Maturité", "La date d'échéance doit être dans le futur.")
                return

            fetched_iv, market_price, closest_date = self.data_fetcher.get_implied_volatility_and_price(
                self.current_ticker, K, maturity_datetime, option_type
            )

            if closest_date:
                closest_date_obj = datetime.strptime(closest_date, '%Y-%m-%d').date()
                time_difference = closest_date_obj - today
                T = time_difference.days / 365.0
                if T <= 0:
                    T = 1e-6

            if fetched_iv is not None and fetched_iv > 0.001 and market_price is not None:
                sigma = fetched_iv
                pricing_method_used = "IV Marché"
            else:
                sigma = self.historical_vol if self.historical_vol is not None and self.historical_vol > 0 else 0.20
                pricing_method_used = "Vol Historique (Fallback)"

            suffix = " (IV)" if "IV" in pricing_method_used else " (historique)"
            self.volatility_label.setText(f"{sigma*100:.2f}%{suffix}")

            crr_price = self.crr_models.cox_ross_rubinstein_price(
                self.S, K, T, self.r, self.q, sigma, N, option_type
            )
            self.crr_price_label.setText(f"{crr_price:.4f} $")

            greeks = self.crr_models.calculate_greeks_crr(
                self.S, K, T, self.r, self.q, sigma, N, option_type
            )

            self.greeks_table.setItem(0, 0, QTableWidgetItem(f"{greeks.get('delta', 0):.4f}"))
            self.greeks_table.setItem(0, 1, QTableWidgetItem(f"{greeks.get('gamma', 0):.4f}"))
            self.greeks_table.setItem(0, 2, QTableWidgetItem(f"{greeks.get('theta', 0):.4f}"))
            self.greeks_table.setItem(0, 3, QTableWidgetItem(f"{greeks.get('vega', 0)/100:.4f}"))
            self.greeks_table.setItem(0, 4, QTableWidgetItem(f"{greeks.get('rho', 0):.4f}"))

            self.current_sigma = sigma
            self.store.update(sigma=sigma, pricing_method=pricing_method_used)

        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie", "Veuillez entrer des valeurs numériques/entières valides pour K et N.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur de Calcul CRR", f"Une erreur inattendue est survenue: {e}")

    # routine de tracé du profil de rentabilité à échéance

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

    # génération de la visualisation des points morts

    def plot_crr_payoff(self):
        try:
            K = float(self.strike_input.text())
            option_type = self.option_type_combo.currentText()
            position = self.position_combo.currentText()

            price_str = self.crr_price_label.text()
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

            self.fig.clear()
            ax = self.fig.add_subplot(111)
            self._draw_payoff(ax, K, premium, option_type, position)

            title_text = f"Payoff de l'Option Américaine {position.capitalize()} {option_type.capitalize()} (K={K:.2f}, Premium={premium:.4f})"
            title_text += f"\nBreakeven = {breakeven:.2f}"
            ax.set_title(title_text)
            self.canvas.draw()

        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie", "Veuillez entrer des valeurs numériques valides.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur de Tracé", f"Une erreur est survenue lors du tracé du payoff: {e}")

    # analyse de la dynamique des sensibilités par rapport au spot

    def handle_crr_greek_click(self, row: int, column: int) -> None:
        greek_names = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
        if column < len(greek_names):
            self.plot_crr_greek_evolution(greek_names[column])

    def plot_crr_greek_evolution(self, greek_name: str) -> None:
        try:
            if self.S is None or self.r is None or self.q is None or self.current_sigma is None:
                QMessageBox.warning(self, "Données Manquantes", "Veuillez d'abord calculer les Grecs CRR.")
                return

            K = float(self.strike_input.text())
            option_type = self.option_type_combo.currentText()
            N = int(self.steps_input.text())

            maturity_qdate = self.maturity_date_input.date()
            maturity_datetime = datetime(maturity_qdate.year(), maturity_qdate.month(), maturity_qdate.day())
            today = date.today()
            T = (maturity_datetime.date() - today).days / 365.0
            if T <= 0:
                QMessageBox.warning(self, "Erreur de Maturité", "La date d'échéance doit être dans le futur.")
                return
            if N < 3:
                QMessageBox.warning(self, "Erreur de Paramètres",
                    "Le nombre de pas (N) doit être entre 3 et 10000.")
                return

            fetched_iv, _, closest_date = self.data_fetcher.get_implied_volatility_and_price(
                self.current_ticker, K, maturity_datetime, option_type
            )
            if closest_date:
                closest_date_obj = datetime.strptime(closest_date, '%Y-%m-%d').date()
                T = (closest_date_obj - today).days / 365.0
                if T <= 0:
                    T = 1e-6

            S_range = np.linspace(self.S * 0.7, self.S * 1.3, 50)
            greek_values = []

            for S in S_range:
                greeks = self.crr_models.calculate_greeks_crr(
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
            dialog.exec()

        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie", "Veuillez entrer des valeurs numériques valides.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Une erreur est survenue: {e}")

import numpy as np
from typing import Optional, Dict, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QFormLayout, QGroupBox, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QDateEdit
)
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QColor
from PyQt5.QtCore import QDate, Qt
from datetime import date

from option_models import OptionModels

class CallPriceSimulationTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.option_models: OptionModels = OptionModels()

        self.ticker_symbol: str = "N/A"
        self.S_current: Optional[float] = None
        self.r_current: Optional[float] = None
        self.q_current: Optional[float] = None
        self.historical_vol_current: Optional[float] = None

        self.init_ui()

    def init_ui(self) -> None:
        """Initialise l'interface utilisateur."""
        main_layout = QVBoxLayout()

        params_group = QGroupBox("Paramètres de la simulation Call Price")
        params_layout = QFormLayout()

        self.ticker_display_label = QLabel("N/A")
        params_layout.addRow("Ticker Symbole:", self.ticker_display_label)
        self.S_display_label = QLabel("N/A")
        params_layout.addRow("Prix Actuel (S):", self.S_display_label)

        self.strike_input = QLineEdit("100.00")
        self.strike_input.setValidator(QDoubleValidator(0.0, 100000.0, 2))
        params_layout.addRow("Prix d'exercice (K):", self.strike_input)

        self.maturity_date_input = QDateEdit(QDate.currentDate().addMonths(3))
        self.maturity_date_input.setCalendarPopup(True)
        self.maturity_date_input.setDisplayFormat("dd/MM/yyyy")
        params_layout.addRow("Date d'échéance:", self.maturity_date_input)

        self.vol_min_display = QLineEdit()
        self.vol_min_display.setReadOnly(True)
        params_layout.addRow("Volatilité -15 bps (%):", self.vol_min_display)

        self.vol_max_display = QLineEdit()
        self.vol_max_display.setReadOnly(True)
        params_layout.addRow("Volatilité +15 bps (%):", self.vol_max_display)

        self.vol_step_input = QLineEdit("1")
        self.vol_step_input.setValidator(QIntValidator(1, 10))
        params_layout.addRow("Pas Volatilité (%):", self.vol_step_input)

        self.underlying_min_display = QLineEdit()
        self.underlying_min_display.setReadOnly(True)
        params_layout.addRow("Prix Sous-jacent -10%:", self.underlying_min_display)

        self.underlying_max_display = QLineEdit()
        self.underlying_max_display.setReadOnly(True)
        params_layout.addRow("Prix Sous-jacent +10%:", self.underlying_max_display)

        self.underlying_step_input = QLineEdit("5")
        self.underlying_step_input.setValidator(QIntValidator(1, 1000))
        params_layout.addRow("Pas Prix Sous-jacent:", self.underlying_step_input)

        self.simulate_button = QPushButton("Lancer la Simulation")
        self.simulate_button.clicked.connect(self.run_simulation)
        params_layout.addRow(self.simulate_button)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        results_group = QGroupBox("Résultats de la Simulation (Volatility(%) Vs Underlying Price)")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        self.setLayout(main_layout)

    def update_financial_data(self, ticker, S, r, q, historical_vol):
        self.ticker_symbol = ticker
        self.S_current = S
        self.r_current = r
        self.q_current = q
        self.historical_vol_current = historical_vol

        self.ticker_display_label.setText(self.ticker_symbol if self.ticker_symbol else "N/A")
        self.S_display_label.setText(f"{self.S_current:.2f}" if self.S_current is not None else "N/A")
        self.update_simulation_ranges()

    def update_simulation_ranges(self):
        if self.historical_vol_current is not None and self.historical_vol_current > 0:
            vol_hist_percent = self.historical_vol_current * 100.0
            
            sim_vol_min = int(round(vol_hist_percent - 15))
            sim_vol_max = int(round(vol_hist_percent + 15))
            
            sim_vol_min = max(1, sim_vol_min)
            sim_vol_max = min(100, sim_vol_max)

            self.vol_min_display.setText(str(sim_vol_min))
            self.vol_max_display.setText(str(sim_vol_max))
        else:
            self.vol_min_display.setText("N/A")
            self.vol_max_display.setText("N/A")

        if self.S_current is not None and self.S_current > 0:
            sim_underlying_min = int(round(self.S_current * 0.9))
            sim_underlying_max = int(round(self.S_current * 1.1))
            
            sim_underlying_min = max(1, sim_underlying_min)

            self.underlying_min_display.setText(str(sim_underlying_min))
            self.underlying_max_display.setText(str(sim_underlying_max))
        else:
            self.underlying_min_display.setText("N/A")
            self.underlying_max_display.setText("N/A")

    def get_color_for_value(self, value, min_val, max_val):
        if max_val == min_val:
            return QColor(128, 128, 0) 
        
        normalized_value = (value - min_val) / (max_val - min_val)
        hue = 120 * (1 - normalized_value)
        color = QColor.fromHsv(int(hue), 255, 200)
        return color

    def run_simulation(self):
        if not self.ticker_symbol or self.ticker_symbol == "N/A" or self.S_current is None or self.r_current is None or self.q_current is None or self.historical_vol_current is None:
            QMessageBox.warning(self, "Données Manquantes",
                                 "Veuillez d'abord récupérer les données financières (Ticker, S, r, q, Volatilité) dans l'onglet 'Calculateur d'Option'.")
            return

        try:
            K = float(self.strike_input.text())

            today = date.today()
            maturity_qdate = self.maturity_date_input.date()
            maturity_date = date(maturity_qdate.year(), maturity_qdate.month(), maturity_qdate.day())
            T_days = (maturity_date - today).days
            T = T_days / 365.0

            if T <= 0:
                QMessageBox.warning(self, "Erreur de Maturité", "La date d'échéance doit être dans le futur.")
                return

            vol_min = int(self.vol_min_display.text())
            vol_max = int(self.vol_max_display.text())
            vol_step = int(self.vol_step_input.text())

            underlying_min = int(self.underlying_min_display.text())
            underlying_max = int(self.underlying_max_display.text())
            underlying_step = int(self.underlying_step_input.text())

            if not (vol_min <= vol_max and vol_step >= 1):
                QMessageBox.warning(self, "Erreur Volatilité", "Vérifiez les paramètres de volatilité (Min <= Max, Pas >= 1).")
                return
            if not (underlying_min <= underlying_max and underlying_step >= 1):
                QMessageBox.warning(self, "Erreur Prix Sous-jacent", "Vérifiez les paramètres du prix sous-jacent (Min <= Max, Pas >= 1).")
                return
            if K <= 0:
                QMessageBox.warning(self, "Erreur de Strike", "Le prix d'exercice doit être supérieur à 0.")
                return

            volatilities_percent = np.arange(vol_min, vol_max + vol_step, vol_step)
            underlying_prices = np.arange(underlying_min, underlying_max + underlying_step, underlying_step)

            if len(volatilities_percent) == 0 or len(underlying_prices) == 0:
                QMessageBox.warning(self, "Plages Vides", "Les plages de volatilité ou de prix sous-jacent générées sont vides. Ajustez les pas ou vérifiez les bornes calculées.")
                return

            all_prices = []
            results_matrix = np.zeros((len(volatilities_percent), len(underlying_prices)))

            for i, vol_percent in enumerate(volatilities_percent):
                sigma = vol_percent / 100.0
                for j, S in enumerate(underlying_prices):
                    price = self.option_models.black_scholes_price(
                        S=float(S), K=K, T=T, r=self.r_current, sigma=sigma, q=self.q_current, option_type='call'
                    )
                    results_matrix[i, j] = price
                    all_prices.append(price)

            if not all_prices:
                QMessageBox.warning(self, "Aucun Résultat", "Aucun prix de Call n'a pu être calculé pour la simulation.")
                self.results_table.setRowCount(0)
                self.results_table.setColumnCount(0)
                return

            min_price = min(all_prices)
            max_price = max(all_prices)

            self.results_table.setRowCount(len(volatilities_percent))
            self.results_table.setColumnCount(len(underlying_prices))

            self.results_table.setHorizontalHeaderLabels([str(s) for s in underlying_prices])
            self.results_table.setVerticalHeaderLabels([f"{v}%" for v in volatilities_percent])

            for i in range(len(volatilities_percent)):
                for j in range(len(underlying_prices)):
                    price = results_matrix[i, j]
                    item = QTableWidgetItem(f"{price:.3f}")
                    
                    color = self.get_color_for_value(price, min_price, max_price)
                    item.setBackground(color)
                    item.setTextAlignment(Qt.AlignCenter) 
                    
                    self.results_table.setItem(i, j, item)

        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie", "Veuillez entrer des valeurs numériques valides pour tous les paramètres.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur de Simulation", f"Une erreur inattendue est survenue: {e}")
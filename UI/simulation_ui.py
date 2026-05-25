# Interface de simulation de portefeuille par variation matricielle des paramètres

from typing import Optional
from datetime import date

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QFormLayout, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QDateEdit, QAbstractItemView,
)
from PySide6.QtGui import QDoubleValidator, QIntValidator, QColor
from PySide6.QtCore import Qt

from logic.bsm_logic import OptionModels
from logic.simulation_logic import SimulationLogic


# définition du thème visuel pour les champs immuables
_READONLY_BG = "background-color: #2b2b2b; color: #888888;"


class CallPriceSimulationTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.option_models = OptionModels()
        self.simulation_logic = SimulationLogic(self.option_models)

        self.ticker_symbol: str = "N/A"
        self.S_current: Optional[float] = None
        self.r_current: Optional[float] = None
        self.q_current: Optional[float] = None
        self.historical_vol_current: Optional[float] = None

        self._build_ui()

    
    # assemblage des composants graphiques
    
    def _build_ui(self) -> None:
        # disposition structurelle de la fenêtre principale
        root_layout = QVBoxLayout(self)

        # segmentation du bandeau supérieur
        top_row = QHBoxLayout()

        # zone de saisie des conditions initiales
        left_layout = QVBoxLayout()

        params_group = QGroupBox("Paramètres de la simulation")
        params_form = QFormLayout()

        # affichage des données financières de référence
        self.company_name_label = QLabel("N/A")
        params_form.addRow("Entreprise:", self.company_name_label)

        self.ticker_display_label = QLabel("N/A")
        params_form.addRow("Ticker Symbole:", self.ticker_display_label)

        self.S_display_label = QLabel("N/A")
        params_form.addRow("Prix Actuel (S):", self.S_display_label)

        # sens de la position
        self.option_type_combo = QComboBox()
        self.option_type_combo.addItems(["call", "put"])
        params_form.addRow("Type d'option:", self.option_type_combo)

        # niveau de déclenchement
        self.strike_input = QLineEdit("100.00")
        self.strike_input.setValidator(QDoubleValidator(0.0, 100000.0, 2))
        params_form.addRow("Prix d'exercice (K):", self.strike_input)

        # horizon temporel de la simulation
        from utils import get_default_maturity_date
        self.maturity_date_input = QDateEdit(get_default_maturity_date())
        self.maturity_date_input.setCalendarPopup(True)
        self.maturity_date_input.setDisplayFormat("dd/MM/yyyy")
        self.maturity_date_input.setToolTip("Date théorique libre")
        params_form.addRow("Date d'échéance:", self.maturity_date_input)

        # déclenchement de la matrice de calcul
        self.simulate_button = QPushButton("Lancer la Simulation")
        self.simulate_button.clicked.connect(self.run_simulation)
        params_form.addRow(self.simulate_button)

        params_group.setLayout(params_form)
        left_layout.addWidget(params_group)
        left_layout.addStretch(1)
        top_row.addLayout(left_layout, 1)

        # zone de définition des scénarios de stress
        right_layout = QVBoxLayout()

        ranges_group = QGroupBox("Plages de la simulation")
        ranges_form = QFormLayout()

        # borne inférieure du choc de volatilité
        self.vol_min_display = QLineEdit()
        self.vol_min_display.setReadOnly(True)
        self.vol_min_display.setStyleSheet(_READONLY_BG)
        ranges_form.addRow("Volatilité −15 bps (%):", self.vol_min_display)

        # borne supérieure du choc de volatilité
        self.vol_max_display = QLineEdit()
        self.vol_max_display.setReadOnly(True)
        self.vol_max_display.setStyleSheet(_READONLY_BG)
        ranges_form.addRow("Volatilité +15 bps (%):", self.vol_max_display)

        # résolution de la grille de volatilité
        self.vol_step_input = QLineEdit("1")
        self.vol_step_input.setValidator(QIntValidator(1, 10))
        ranges_form.addRow("Pas Volatilité (%):", self.vol_step_input)

        # borne inférieure du choc directionnel
        self.underlying_min_display = QLineEdit()
        self.underlying_min_display.setReadOnly(True)
        self.underlying_min_display.setStyleSheet(_READONLY_BG)
        ranges_form.addRow("Prix Sous-jacent −10%:", self.underlying_min_display)

        # borne supérieure du choc directionnel
        self.underlying_max_display = QLineEdit()
        self.underlying_max_display.setReadOnly(True)
        self.underlying_max_display.setStyleSheet(_READONLY_BG)
        ranges_form.addRow("Prix Sous-jacent +10%:", self.underlying_max_display)

        # résolution de la grille directionnelle
        self.underlying_step_input = QLineEdit("5")
        self.underlying_step_input.setValidator(QIntValidator(1, 1000))
        ranges_form.addRow("Pas Prix Sous-jacent:", self.underlying_step_input)

        ranges_group.setLayout(ranges_form)
        right_layout.addWidget(ranges_group)
        right_layout.addStretch(1)
        top_row.addLayout(right_layout, 1)

        root_layout.addLayout(top_row)

        # restitution spatiale de la matrice de rentabilité
        results_group = QGroupBox("Résultats de la Simulation (Volatilité (%) Vs Prix Sous-jacent)")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        root_layout.addWidget(results_group, 1)

    
    # synchronisation de l'état avec le modèle de données global
    
    def update_financial_data(self, ticker, S, r, q, historical_vol) -> None:
        self.ticker_symbol = ticker
        self.S_current = S
        self.r_current = r
        self.q_current = q
        self.historical_vol_current = historical_vol

        self.ticker_display_label.setText(self.ticker_symbol if self.ticker_symbol else "N/A")
        self.S_display_label.setText(f"{self.S_current:.2f}" if self.S_current is not None else "N/A")
        self._update_simulation_ranges()

    def update_company_name(self, company_name: str) -> None:
        self.company_name_label.setText(company_name if company_name else "N/A")

    def _update_simulation_ranges(self) -> None:
        if self.historical_vol_current is not None and self.historical_vol_current > 0:
            vol_pct = self.historical_vol_current * 100.0
            vol_min = max(1, int(round(vol_pct - 15)))
            vol_max = min(100, int(round(vol_pct + 15)))
            self.vol_min_display.setText(str(vol_min))
            self.vol_max_display.setText(str(vol_max))
        else:
            self.vol_min_display.setText("N/A")
            self.vol_max_display.setText("N/A")

        if self.S_current is not None and self.S_current > 0:
            s_min = max(1, int(round(self.S_current * 0.9)))
            s_max = int(round(self.S_current * 1.1))
            self.underlying_min_display.setText(str(s_min))
            self.underlying_max_display.setText(str(s_max))
        else:
            self.underlying_min_display.setText("N/A")
            self.underlying_max_display.setText("N/A")

    
    # interpolation chromatique pour la matrice de résultats
    
    def _color_for_value(self, value: float, min_val: float, max_val: float) -> QColor:
        if max_val == min_val:
            return QColor(128, 128, 0)
        normalized = (value - min_val) / (max_val - min_val)
        hue = int(120 * (1 - normalized))
        return QColor.fromHsv(hue, 255, 200)

    
    # exécution du pipeline d'évaluation stochastique
    
    def run_simulation(self) -> None:
        if (not self.ticker_symbol or self.ticker_symbol == "N/A"
                or self.S_current is None or self.r_current is None
                or self.q_current is None or self.historical_vol_current is None):
            QMessageBox.warning(
                self, "Données Manquantes",
                "Veuillez d'abord récupérer les données financières (Ticker, S, r, q, Volatilité) "
                "dans l'onglet 'Calculateur d'Option'."
            )
            return

        try:
            K = float(self.strike_input.text())
            if K <= 0:
                QMessageBox.warning(self, "Erreur de Strike",
                                    "Le prix d'exercice doit être supérieur à 0.")
                return

            today = date.today()
            mq = self.maturity_date_input.date()
            maturity = date(mq.year(), mq.month(), mq.day())
            T = (maturity - today).days / 365.0
            if T <= 0:
                QMessageBox.warning(self, "Erreur de Maturité",
                                    "La date d'échéance doit être dans le futur.")
                return

            option_type = self.option_type_combo.currentText()

            vol_min_txt = self.vol_min_display.text()
            vol_max_txt = self.vol_max_display.text()
            s_min_txt = self.underlying_min_display.text()
            s_max_txt = self.underlying_max_display.text()

            if "N/A" in [vol_min_txt, vol_max_txt, s_min_txt, s_max_txt]:
                QMessageBox.warning(self, "Données Manquantes",
                                    "Certains paramètres de plage sont à 'N/A'.")
                return

            vol_min = int(vol_min_txt)
            vol_max = int(vol_max_txt)
            vol_step = int(self.vol_step_input.text())
            s_min = int(s_min_txt)
            s_max = int(s_max_txt)
            s_step = int(self.underlying_step_input.text())

            if not (vol_min <= vol_max and vol_step >= 1):
                QMessageBox.warning(self, "Erreur Volatilité",
                                    "Vérifiez les paramètres de volatilité (Min ≤ Max, Pas ≥ 1).")
                return
            if not (s_min <= s_max and s_step >= 1):
                QMessageBox.warning(self, "Erreur Prix Sous-jacent",
                                    "Vérifiez les paramètres du prix sous-jacent (Min ≤ Max, Pas ≥ 1).")
                return

            vols, prices, matrix, all_prices = self.simulation_logic.run_simulation(
                K=K, T=T, r=self.r_current, q=self.q_current,
                vol_min=vol_min, vol_max=vol_max, vol_step=vol_step,
                underlying_min=s_min, underlying_max=s_max, underlying_step=s_step,
                option_type=option_type,
            )

            if len(vols) == 0 or len(prices) == 0:
                QMessageBox.warning(self, "Plages Vides",
                                    "Les plages générées sont vides. Ajustez les pas.")
                return

            if not all_prices:
                QMessageBox.warning(self, "Aucun Résultat",
                                    "Aucun prix n'a pu être calculé.")
                self.results_table.setRowCount(0)
                self.results_table.setColumnCount(0)
                return

            min_p, max_p = min(all_prices), max(all_prices)

            self.results_table.setRowCount(len(vols))
            self.results_table.setColumnCount(len(prices))
            self.results_table.setHorizontalHeaderLabels([str(s) for s in prices])
            self.results_table.setVerticalHeaderLabels([f"{v}%" for v in vols])

            for i in range(len(vols)):
                for j in range(len(prices)):
                    price = matrix[i, j]
                    item = QTableWidgetItem(f"{price:.3f}")
                    item.setBackground(self._color_for_value(price, min_p, max_p))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.results_table.setItem(i, j, item)

        except ValueError:
            QMessageBox.warning(self, "Erreur de Saisie",
                                "Veuillez entrer des valeurs numériques valides.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur de Simulation",
                                 f"Une erreur inattendue est survenue : {e}")
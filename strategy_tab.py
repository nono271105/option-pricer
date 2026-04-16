"""
strategy_tab.py
---------------
Onglet "Stratégies"
Panneau gauche  : famille → stratégie → ticker → date → fetch → legs auto-générés
Panneau droit   : 5 métriques + 2 graphiques (payoff maturité + valeur today) + grecs agrégés
Nouvelles méthodes utilisées depuis strategy_manager.py :
    build_legs()          — construit les legs selon la stratégie choisie
    fetch_leg_premiums()  — récupère les primes via yfinance (fallback BSM)
    compute_metrics()     — coût, gain max, perte max, breakevens
    compute_payoff()      — P&L à maturité
    compute_value_today() — valeur actuelle avec valeur temps
    compute_greeks()      — grecs agrégés de tous les legs
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict
from datetime import date, datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QFormLayout, QGroupBox, QGridLayout,
    QMessageBox, QDateEdit, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from data_fetcher import DataFetcher
from option_models import OptionModels
from strategy_manager import StrategyManager


# =============================================================================
# Worker QThread — récupération des primes + calculs en arrière-plan
# =============================================================================

class StrategyWorker(QThread):
    """Récupère les primes de marché et calcule toutes les métriques."""
    result_ready   = pyqtSignal(object)   # dict de résultats
    error_occurred = pyqtSignal(str)

    def __init__(self, params: dict, manager: StrategyManager,
                 option_models: OptionModels, data_fetcher: DataFetcher):
        super().__init__()
        self.params       = params
        self.manager      = manager
        self.option_models = option_models
        self.data_fetcher  = data_fetcher

    def run(self):
        try:
            p = self.params
            legs = self.manager.build_legs(
                p["strategy_name"], p["S"], p["T"],
                p["r"], p["sigma"], p["q"],
                p["maturity_datetime"], p["ticker"],
                self.data_fetcher, self.option_models
            )

            S_range = np.linspace(p["S"] * 0.60, p["S"] * 1.40, 400)

            payoff       = self.manager.compute_payoff(legs, S_range)
            value_today  = self.manager.compute_value_today(
                legs, S_range, p["S"], p["T"], p["r"], p["sigma"], p["q"],
                self.option_models)
            metrics      = self.manager.compute_metrics(legs, S_range, payoff)
            greeks       = self.manager.compute_greeks(
                legs, p["S"], p["T"], p["r"], p["sigma"], p["q"],
                self.option_models)

            self.result_ready.emit({
                "legs":        legs,
                "S_range":     S_range,
                "payoff":      payoff,
                "value_today": value_today,
                "metrics":     metrics,
                "greeks":      greeks,
                "S":           p["S"],
            })
        except Exception as exc:
            self.error_occurred.emit(str(exc))


# =============================================================================
# Canvas Matplotlib : payoff à maturité + valeur aujourd'hui
# =============================================================================

class StrategyCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 6))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._build_axes()

    def _build_axes(self):
        gs = gridspec.GridSpec(2, 1, figure=self.fig,
                               hspace=0.45,
                               left=0.07, right=0.97,
                               top=0.93, bottom=0.07)
        self.ax_payoff = self.fig.add_subplot(gs[0])
        self.ax_value  = self.fig.add_subplot(gs[1])
        for ax in (self.ax_payoff, self.ax_value):
            ax.grid(True, linestyle=":", alpha=0.5)

    def clear_all(self):
        for ax in (self.ax_payoff, self.ax_value):
            ax.cla()
            ax.grid(True, linestyle=":", alpha=0.5)

    def plot(self, result: dict, strategy_name: str):
        self.clear_all()
        S_range     = result["S_range"]
        payoff      = result["payoff"]
        value_today = result["value_today"]
        S           = result["S"]
        metrics     = result["metrics"]

        # 1. Payoff à maturité
        ax = self.ax_payoff
        ax.plot(S_range, payoff, color="#1f77b4", lw=2, label="Payoff à maturité")
        ax.fill_between(S_range, payoff, 0,
                        where=(payoff >= 0), alpha=0.15, color="green")
        ax.fill_between(S_range, payoff, 0,
                        where=(payoff < 0),  alpha=0.15, color="red")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(S, color="red", lw=1.2, ls="--", label=f"S₀ = {S:.2f}")

        # Breakevens
        for be in metrics.get("breakevens", []):
            ax.axvline(be, color="orange", lw=1.0, ls=":",
                       label=f"BE = {be:.2f}")

        ax.set_title(f"{strategy_name} — Payoff à maturité", fontsize=9)
        ax.set_xlabel("Prix sous-jacent ($)", fontsize=8)
        ax.set_ylabel("P&L ($)", fontsize=8)
        ax.legend(fontsize=7, loc="best")

        # 2. Valeur actuelle avec valeur temps
        ax = self.ax_value
        ax.plot(S_range, value_today, color="#2ca02c", lw=2,
                label="Valeur actuelle (avec valeur temps)")
        ax.plot(S_range, payoff, color="#1f77b4", lw=1,
                ls="--", alpha=0.5, label="Payoff à maturité")
        ax.fill_between(S_range, value_today, 0,
                        where=(value_today >= 0), alpha=0.12, color="green")
        ax.fill_between(S_range, value_today, 0,
                        where=(value_today < 0),  alpha=0.12, color="red")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(S, color="red", lw=1.2, ls="--", label=f"S₀ = {S:.2f}")

        ax.set_title("Valeur actuelle de la stratégie", fontsize=9)
        ax.set_xlabel("Prix sous-jacent ($)", fontsize=8)
        ax.set_ylabel("Valeur ($)", fontsize=8)
        ax.legend(fontsize=7, loc="best")

        self.fig.canvas.draw_idle()


# =============================================================================
# Onglet principal
# =============================================================================

class StrategyTab(QWidget):
    """
    Onglet Stratégies — structure identique à CRRModelTab.
    """

    # Familles et stratégies disponibles
    FAMILIES: Dict[str, List[str]] = {
        "Positions de base": [
            "Long Call", "Short Call", "Long Put", "Short Put",
        ],
        "Spreads directionnels": [
            "Bull Call Spread", "Bear Call Spread",
            "Bull Put Spread",  "Bear Put Spread",
        ],
        "Volatilité": [
            "Long Straddle", "Short Straddle",
            "Long Strangle",  "Short Strangle",
        ],
        "Butterflies": [
            "Long Call Butterfly",  "Short Call Butterfly",
            "Long Put Butterfly",   "Short Put Butterfly",
            "Long Iron Butterfly",  "Short Iron Butterfly",
        ],
        "Condors": [
            "Long Call Condor",  "Short Call Condor",
            "Long Put Condor",   "Short Put Condor",
            "Long Iron Condor",  "Short Iron Condor",
        ],
    }

    def __init__(self, app_instance, parent=None):
        super().__init__(parent)
        self.app           = app_instance
        self.data_fetcher  = DataFetcher()
        self.option_models = OptionModels()
        self.manager       = StrategyManager()
        self._worker: Optional[StrategyWorker] = None

        # Données marché
        self._S:      Optional[float] = None
        self._r:      Optional[float] = None
        self._q:      Optional[float] = None
        self._sigma:  Optional[float] = None
        self._ticker: Optional[str]   = None

        self._build_ui()

    # =========================================================================
    # Construction de l'UI
    # =========================================================================

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # Panneau gauche
        left_layout = QVBoxLayout()

        # Groupe Stratégie 
        strat_group = QGroupBox("Paramètres de la stratégie")
        strat_form  = QFormLayout()
        strat_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ex: AAPL")
        strat_form.addRow("Ticker Symbole:", self.ticker_input)

        self.family_combo = QComboBox()
        self.family_combo.addItems(list(self.FAMILIES.keys()))
        self.family_combo.currentIndexChanged.connect(self._on_family_changed)
        strat_form.addRow("Famille:", self.family_combo)

        self.strategy_combo = QComboBox()
        strat_form.addRow("Stratégie:", self.strategy_combo)

        self.maturity_date_input = QDateEdit(QDate.currentDate().addMonths(3))
        self.maturity_date_input.setCalendarPopup(True)
        self.maturity_date_input.setDisplayFormat("dd/MM/yyyy")
        strat_form.addRow("Date d'échéance:", self.maturity_date_input)

        strat_group.setLayout(strat_form)
        left_layout.addWidget(strat_group)

        # Boutons
        self.fetch_data_button = QPushButton("Récupérer/Synchroniser les Données")
        self.fetch_data_button.clicked.connect(self._fetch_data)
        left_layout.addWidget(self.fetch_data_button)

        self.calculate_button = QPushButton("Calculer la Stratégie")
        self.calculate_button.clicked.connect(self._on_calculate)
        left_layout.addWidget(self.calculate_button)

        # Tableau des legs
        legs_group  = QGroupBox("Legs de la stratégie")
        legs_layout = QVBoxLayout(legs_group)
        self.legs_table = QTableWidget(0, 4)
        self.legs_table.setHorizontalHeaderLabels(["Type", "Position", "Strike", "Prime ($)"])
        self.legs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.legs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.legs_table.setMaximumHeight(160)
        legs_layout.addWidget(self.legs_table)
        left_layout.addWidget(legs_group)

        left_layout.addStretch(1)
        main_layout.addLayout(left_layout, 1)

        # Panneau droit 
        right_layout = QVBoxLayout()

        # Données actuelles
        data_group  = QGroupBox("Données Actuelles")
        data_layout = QFormLayout()
        self.live_price_label = QLabel("N/A")
        self.risk_free_label  = QLabel("N/A")
        self.dividend_label   = QLabel("N/A")
        self.vol_label        = QLabel("N/A")
        data_layout.addRow("Prix Actuel (S):",           self.live_price_label)
        data_layout.addRow("Taux Sans Risque SOFR (r):", self.risk_free_label)
        data_layout.addRow("Rendement Dividende (q):",   self.dividend_label)
        data_layout.addRow("Volatilité Utilisée (σ):",   self.vol_label)
        data_group.setLayout(data_layout)
        right_layout.addWidget(data_group)

        # Métriques
        metrics_group  = QGroupBox("Métriques de la stratégie")
        metrics_grid   = QGridLayout(metrics_group)

        self.cost_label    = QLabel("N/A")
        self.be_low_label  = QLabel("N/A")
        self.be_high_label = QLabel("N/A")
        self.gain_label    = QLabel("N/A")
        self.loss_label    = QLabel("N/A")

        for col, (lbl_text, val_lbl) in enumerate([
            ("Coût total ($)",   self.cost_label),
            ("Breakeven bas ($)", self.be_low_label),
            ("Breakeven haut ($)", self.be_high_label),
            ("Gain max ($)",     self.gain_label),
            ("Perte max ($)",    self.loss_label),
        ]):
            box = QWidget()
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(4, 4, 4, 4)
            box_layout.setSpacing(2)
            title = QLabel(lbl_text)
            title.setStyleSheet("font-size: 10px; color: gray;")
            box_layout.addWidget(title)
            val_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
            box_layout.addWidget(val_lbl)
            metrics_grid.addWidget(box, 0, col)

        right_layout.addWidget(metrics_group)

        # Grecs agrégés
        greeks_group = QGroupBox("Grecs de la stratégie (BSM agrégés)")
        greeks_grid  = QGridLayout(greeks_group)
        self.greeks_table = QTableWidget(1, 5)
        self.greeks_table.setHorizontalHeaderLabels(
            ["Delta (Δ)", "Gamma (Γ)", "Theta (Θ/j)", "Vega (ν)", "Rho (ρ)"])
        self.greeks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.greeks_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.greeks_table.setMaximumHeight(58)
        for col in range(5):
            self.greeks_table.setItem(0, col, QTableWidgetItem("N/A"))
        greeks_grid.addWidget(self.greeks_table, 0, 0)
        right_layout.addWidget(greeks_group)

        # Graphiques
        plot_group  = QGroupBox("Visualisation")
        plot_layout = QVBoxLayout(plot_group)
        self.canvas = StrategyCanvas(self)
        plot_layout.addWidget(self.canvas)
        right_layout.addWidget(plot_group)

        main_layout.addLayout(right_layout, 2)

        # Initialise le combo stratégies
        self._on_family_changed()

    # =========================================================================
    # Logique famille → stratégies
    # =========================================================================

    def _on_family_changed(self):
        family = self.family_combo.currentText()
        self.strategy_combo.clear()
        self.strategy_combo.addItems(self.FAMILIES.get(family, []))

    # =========================================================================
    # Fetch données marché
    # =========================================================================

    def _fetch_data(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer un symbole de ticker.")
            return
        self.app.fetch_data_for_tab(ticker, self)

    def update_financial_data(self, ticker: str, S: Optional[float],
                               r: Optional[float], q: Optional[float],
                               sigma: Optional[float],
                               pricing_method: str = "") -> None:
        """Appelé par update_all_tabs_financial_data dans gui_app.py."""
        self._ticker = ticker
        self._S      = S
        self._r      = r
        self._q      = q
        self._sigma  = sigma

        if ticker and self.ticker_input.text().strip() == "":
            self.ticker_input.setText(ticker)

        self.live_price_label.setText(f"{S:.2f}" if S is not None else "N/A")
        self.risk_free_label.setText(f"{r*100:.2f}%" if r is not None else "N/A")
        self.dividend_label.setText(f"{q*100:.2f}%" if q is not None else "N/A")
        if sigma is not None:
            suffix = f" ({pricing_method})" if pricing_method else ""
            self.vol_label.setText(f"{sigma*100:.2f}%{suffix}")
        else:
            self.vol_label.setText("N/A")

        self.fetch_data_button.setEnabled(True)
        self.fetch_data_button.setText("Récupérer/Synchroniser les Données")

    # =========================================================================
    # Lancement du calcul
    # =========================================================================

    def _on_calculate(self):
        if self._S is None or self._r is None or self._q is None or self._sigma is None:
            QMessageBox.warning(self, "Données manquantes",
                                "Veuillez d'abord récupérer les données financières.")
            return

        maturity_qdate    = self.maturity_date_input.date()
        maturity_date_obj = date(maturity_qdate.year(),
                                 maturity_qdate.month(),
                                 maturity_qdate.day())
        T = (maturity_date_obj - date.today()).days / 365.0
        if T <= 0:
            QMessageBox.warning(self, "Erreur", "La date d'échéance doit être dans le futur.")
            return

        strategy_name = self.strategy_combo.currentText()
        if not strategy_name:
            QMessageBox.warning(self, "Erreur", "Veuillez sélectionner une stratégie.")
            return

        self.calculate_button.setEnabled(False)
        self.calculate_button.setText("⏳ Calcul en cours...")

        params = {
            "strategy_name":    strategy_name,
            "ticker":           self._ticker or self.ticker_input.text().strip().upper(),
            "S":                self._S,
            "T":                T,
            "r":                self._r,
            "sigma":            self._sigma,
            "q":                self._q,
            "maturity_datetime": datetime(maturity_qdate.year(),
                                          maturity_qdate.month(),
                                          maturity_qdate.day()),
        }

        self._worker = StrategyWorker(params, self.manager,
                                      self.option_models, self.data_fetcher)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._reset_button)
        self._worker.start()

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_result(self, result: dict):
        legs    = result["legs"]
        metrics = result["metrics"]
        greeks  = result["greeks"]

        # Tableau des legs
        self.legs_table.setRowCount(len(legs))
        for row, leg in enumerate(legs):
            type_item = QTableWidgetItem(leg["option_type"].capitalize())
            pos_item  = QTableWidgetItem(leg["position"].capitalize())
            k_item    = QTableWidgetItem(f"{leg['strike']:.2f}")
            p_item    = QTableWidgetItem(f"{leg['premium']:.4f}")

            # Couleur selon position
            bg_pos  = QColor("#70E672") if leg["position"] == "long" else QColor("#FF6464")
            bg_type = QColor("#7CBFFE") if leg["option_type"] == "call" else QColor("#FBD28B")

            type_item.setBackground(bg_type)
            pos_item.setBackground(bg_pos)

            for item in (type_item, pos_item, k_item, p_item):
                item.setTextAlignment(Qt.AlignCenter)

            self.legs_table.setItem(row, 0, type_item)
            self.legs_table.setItem(row, 1, pos_item)
            self.legs_table.setItem(row, 2, k_item)
            self.legs_table.setItem(row, 3, p_item)

        # Métriques
        cost = metrics["cost"]
        self.cost_label.setText(f"${cost:.2f}")
        self.cost_label.setStyleSheet(
            "font-size:14px;font-weight:bold;color:" +
            ("#A32D2D" if cost > 0 else "#3B6D11"))

        breakevens = metrics.get("breakevens", [])
        self.be_low_label.setText(
            f"${breakevens[0]:.2f}" if len(breakevens) >= 1 else "N/A")
        self.be_high_label.setText(
            f"${breakevens[1]:.2f}" if len(breakevens) >= 2 else "N/A")

        gain = metrics.get("max_gain")
        loss = metrics.get("max_loss")

        gain_txt = f"${gain:.2f}" if gain is not None and np.isfinite(gain) else "Illimité"
        loss_txt = f"${loss:.2f}" if loss is not None and np.isfinite(loss) else "Illimité"
        self.gain_label.setText(gain_txt)
        self.gain_label.setStyleSheet("font-size:14px;font-weight:bold;color:#3B6D11")
        self.loss_label.setText(loss_txt)
        self.loss_label.setStyleSheet("font-size:14px;font-weight:bold;color:#A32D2D")

        # Grecs agrégés
        greek_keys = ["delta", "gamma", "theta", "vega", "rho"]
        for col, key in enumerate(greek_keys):
            val = greeks.get(key, 0)
            item = QTableWidgetItem(f"{val:.4f}")
            item.setTextAlignment(Qt.AlignCenter)
            if key in ("delta", "theta", "rho"):
                item.setForeground(QColor("#3B6D11") if val >= 0 else QColor("#A32D2D"))
            self.greeks_table.setItem(0, col, item)

        # Graphiques
        strategy_name = self.strategy_combo.currentText()
        self.canvas.plot(result, strategy_name)

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Erreur de calcul", msg)
        self._reset_button()

    def _reset_button(self):
        self.calculate_button.setEnabled(True)
        self.calculate_button.setText("Calculer la Stratégie")
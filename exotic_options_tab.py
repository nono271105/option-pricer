"""
exotic_options_tab.py
---------------------
Onglet PyQt5 "Options Exotiques" — style identique aux onglets BSM et CRR.

Structure :
  - Panneau gauche  : QFormLayout dans QGroupBox (ticker, paramètres, boutons)
  - Panneau droit   : "Données Actuelles" + tableau résultats + graphique Matplotlib
  - Connexion yfinance via DataFetcher (même appel que les autres onglets)
  - Calcul en QThread (pas de freeze UI)
  - Monte Carlo : trajectoires GBM affichées sur le graphique principal
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from datetime import date, datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox,
    QFormLayout, QGroupBox,
    QMessageBox, QDateEdit, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate
from PyQt5.QtGui import QDoubleValidator, QIntValidator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from data_fetcher import DataFetcher
from exotic_options_models import (
    ExoticResult,
    price_barrier_analytical, price_barrier_mc,
    price_asian_mc,
    price_lookback_mc,
    price_digital_analytical,  price_digital_mc,
)


# =============================================================================
# Worker QThread — calcul analytique + Monte Carlo en arrière-plan
# =============================================================================

class PricingWorker(QThread):
    result_ready   = pyqtSignal(object, object)   # (ana: ExoticResult, mc: ExoticResult)
    error_occurred = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            p     = self.params
            opt   = p["exotic_type"]
            otype = p["option_type"]

            if opt == "barrier":
                ana = price_barrier_analytical(
                    p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"],
                    p["barrier"], otype, p["barrier_type"])
                mc  = price_barrier_mc(
                    p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"],
                    p["barrier"], otype, p["barrier_type"],
                    n_sims=p["n_sims"], n_steps=p["n_steps"])

            elif opt == "asian":
                ana = None   # Pas de formule analytique pour l'asiatique
                mc  = price_asian_mc(
                    p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"],
                    otype, p["averaging"],
                    n_sims=p["n_sims"], n_steps=p["n_steps"])

            elif opt == "lookback":
                ana = None   # Pas de formule analytique pour le lookback
                mc  = price_lookback_mc(
                    p["S"], p["T"], p["r"], p["sigma"], p["q"], otype,
                    n_sims=p["n_sims"], n_steps=p["n_steps"])

            else:   # digital
                ana = price_digital_analytical(
                    p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"],
                    otype, p["payoff_amount"])
                mc  = price_digital_mc(
                    p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"],
                    otype, p["payoff_amount"],
                    n_sims=p["n_sims"], n_steps=p["n_steps"])

            self.result_ready.emit(ana, mc)

        except Exception as exc:
            self.error_occurred.emit(str(exc))


# =============================================================================
# Canvas Matplotlib — trajectoires MC, distribution payoffs, profil payoff
# =============================================================================

class ExoticCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 7))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._build_axes()

    def _build_axes(self):
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.42, wspace=0.32,
                               left=0.07, right=0.97,
                               top=0.93, bottom=0.08)
        self.ax_paths  = self.fig.add_subplot(gs[0, :])
        self.ax_dist   = self.fig.add_subplot(gs[1, 0])
        self.ax_payoff = self.fig.add_subplot(gs[1, 1])
        self._style_axes()
        self.fig.suptitle("Analyse Options Exotiques", fontsize=11, fontweight="bold")

    def _style_axes(self):
        for ax in (self.ax_paths, self.ax_dist, self.ax_payoff):
            ax.grid(True, linestyle=":", alpha=0.5)

    def clear_all(self):
        for ax in (self.ax_paths, self.ax_dist, self.ax_payoff):
            ax.cla()
        self._style_axes()

    def plot_results(self, mc: ExoticResult, params: dict):
        self.clear_all()
        S       = params["S"]
        T       = params["T"]
        n_steps = params["n_steps"]
        exotic  = params["exotic_type"]
        otype   = params["option_type"]

        time_axis = np.linspace(0, T * 252, n_steps + 1)

        # ── 1. Trajectoires Monte Carlo ───────────────────────────────────────
        ax = self.ax_paths
        if mc.price_paths is not None:
            paths  = mc.price_paths
            n_show = min(100, paths.shape[0])
            alpha  = max(0.03, 0.5 / n_show)
            for i in range(n_show):
                ax.plot(time_axis, paths[i], color="#1f77b4", alpha=alpha, lw=0.7)
            # Trajectoire moyenne
            ax.plot(time_axis, paths[:n_show].mean(axis=0),
                    color="darkorange", lw=2.0, label="Moyenne MC", zorder=5)
            ax.axhline(S, color="red", lw=1.3, ls="--", label=f"S₀ = {S:.2f}")
            if exotic == "barrier":
                ax.axhline(params["barrier"], color="purple", lw=1.5,
                           ls="--", label=f"H = {params['barrier']:.2f}")
            if exotic in ("barrier", "asian", "digital"):
                ax.axhline(params["K"], color="green", lw=1.2,
                           ls=":", label=f"K = {params['K']:.2f}")

        ax.set_title(f"Trajectoires GBM — {n_show} chemins affichés", fontsize=9)
        ax.set_xlabel("Jours de trading", fontsize=8)
        ax.set_ylabel("Prix sous-jacent ($)", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")

        # ── 2. Distribution des payoffs ───────────────────────────────────────
        ax = self.ax_dist
        if mc.payoffs is not None:
            all_pf  = mc.payoffs
            nonzero = all_pf[all_pf > 0]
            pct_itm = 100 * len(nonzero) / max(len(all_pf), 1)
            if len(nonzero) > 0:
                ax.hist(nonzero, bins=60, color="#2196F3", alpha=0.75,
                        edgecolor="none", density=True)
                ax.axvline(all_pf.mean(), color="darkorange", lw=1.5,
                           ls="--", label=f"Moyenne = {all_pf.mean():.3f}")
                ax.legend(fontsize=7)
            ax.set_title(f"Distribution payoffs — {pct_itm:.1f}% ITM", fontsize=9)
            ax.set_xlabel("Payoff ($)", fontsize=8)
            ax.set_ylabel("Densité", fontsize=8)

        # ── 3. Profil de payoff à maturité ────────────────────────────────────
        ax = self.ax_payoff
        S_range = np.linspace(S * 0.5, S * 1.5, 300)

        if exotic == "barrier":
            K        = params["K"]
            barrier  = params["barrier"]
            phi      = 1 if otype == "call" else -1
            is_up    = "up"  in params["barrier_type"]
            is_out   = "out" in params["barrier_type"]
            intrinsic = np.maximum(phi * (S_range - K), 0)
            breached  = (S_range >= barrier) if is_up else (S_range <= barrier)
            active    = ~breached if is_out else breached
            payoff_p  = np.where(active, intrinsic, 0)
            ax.fill_between(S_range, payoff_p, alpha=0.25, color="#9C27B0")
            ax.plot(S_range, payoff_p, color="#9C27B0", lw=1.5)
            ax.axvline(barrier, color="purple", lw=1.3, ls="--",
                       label=f"H = {barrier:.2f}")
            ax.axvline(K, color="green", lw=1.2, ls=":", label=f"K = {K:.2f}")

        elif exotic == "asian":
            K        = params["K"]
            phi      = 1 if otype == "call" else -1
            payoff_p = np.maximum(phi * (S_range - K), 0)
            ax.fill_between(S_range, payoff_p, alpha=0.25, color="green")
            ax.plot(S_range, payoff_p, color="green", lw=1.5,
                    label="Payoff approx")
            ax.axvline(K, color="green", lw=1.2, ls="--", label=f"K = {K:.2f}")

        elif exotic == "lookback":
            payoff_p = S_range - S_range[0]
            ax.fill_between(S_range, payoff_p, alpha=0.25, color="darkorange")
            ax.plot(S_range, payoff_p, color="darkorange", lw=1.5,
                    label="Payoff illustratif")

        else:  # digital
            K        = params["K"]
            amt      = params["payoff_amount"]
            phi      = 1 if otype == "call" else -1
            payoff_p = np.where(phi * (S_range - K) > 0, amt, 0.0)
            ax.step(S_range, payoff_p, color="#00BCD4", lw=2, where="post")
            ax.fill_between(S_range, payoff_p, alpha=0.20,
                            color="#00BCD4", step="post")
            ax.axvline(K, color="red", lw=1.2, ls="--", label=f"K = {K:.2f}")

        ax.axvline(S, color="red", lw=1.2, ls="--", label=f"S₀ = {S:.2f}")
        ax.set_title("Profil de payoff à maturité", fontsize=9)
        ax.set_xlabel("Prix sous-jacent ($)", fontsize=8)
        ax.set_ylabel("Payoff ($)", fontsize=8)
        ax.legend(fontsize=7)

        self.fig.canvas.draw_idle()


# =============================================================================
# Onglet principal — même structure que BSM / CRR
# =============================================================================

class ExoticOptionsTab(QWidget):
    """
    Onglet Options Exotiques.
    Structure identique à CRRModelTab :
      - Panneau gauche  : paramètres (QFormLayout / QGroupBox)
      - Panneau droit   : données actuelles + résultats + graphique Matplotlib
    """

    def __init__(self, app_instance, parent=None):
        super().__init__(parent)
        self.app          = app_instance
        self.data_fetcher = DataFetcher()
        self._worker: Optional[PricingWorker] = None

        # Données marché synchronisées depuis BSM / fetch
        self._S:     Optional[float] = None
        self._r:     Optional[float] = None
        self._q:     Optional[float] = None
        self._sigma: Optional[float] = None
        self._ticker: Optional[str]  = None

        self._build_ui()

    # =========================================================================
    # Construction de l'interface
    # =========================================================================

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # ── Panneau gauche ─────────────────────────────────────────────────────
        left_layout = QVBoxLayout()

        # --- Groupe paramètres principaux ---
        params_group = QGroupBox("Paramètres de l'option exotique")
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ex: AAPL")
        form.addRow("Ticker Symbole:", self.ticker_input)

        self.exotic_combo = QComboBox()
        self.exotic_combo.addItems([
            "Barrière",
            "Asiatique (moyenne)",
            "Lookback",
            "Digitale / Binaire",
        ])
        self.exotic_combo.currentIndexChanged.connect(self._on_exotic_changed)
        form.addRow("Type exotique:", self.exotic_combo)

        self.option_type_combo = QComboBox()
        self.option_type_combo.addItems(["call", "put"])
        form.addRow("Type d'option:", self.option_type_combo)

        self.strike_input = QLineEdit("150.00")
        self.strike_input.setValidator(QDoubleValidator(0.0, 1e6, 2))
        form.addRow("Prix d'exercice (K):", self.strike_input)

        self.maturity_date_input = QDateEdit(QDate.currentDate().addMonths(3))
        self.maturity_date_input.setCalendarPopup(True)
        self.maturity_date_input.setDisplayFormat("dd/MM/yyyy")
        form.addRow("Date d'échéance:", self.maturity_date_input)

        self.position_combo = QComboBox()
        self.position_combo.addItems(["long", "short"])
        form.addRow("Position:", self.position_combo)

        params_group.setLayout(form)
        left_layout.addWidget(params_group)

        # --- Groupe paramètres spécifiques (conditionnel) ---
        self.specific_group = QGroupBox("Paramètres spécifiques")
        self.specific_form  = QFormLayout()
        self.specific_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # Widgets barrière
        self.barrier_input = QLineEdit("120.00")
        self.barrier_input.setValidator(QDoubleValidator(0.0, 1e6, 2))
        self.barrier_type_combo = QComboBox()
        self.barrier_type_combo.addItems([
            "up-and-out", "up-and-in", "down-and-out", "down-and-in"
        ])
        # Widgets asiatique
        self.averaging_combo = QComboBox()
        self.averaging_combo.addItems(["arithmetic", "geometric"])
        # Widgets digitale
        self.payoff_amount_input = QLineEdit("1.00")
        self.payoff_amount_input.setValidator(QDoubleValidator(0.0, 1e6, 2))

        self.specific_group.setLayout(self.specific_form)
        left_layout.addWidget(self.specific_group)

        # --- Groupe Monte Carlo ---
        mc_group = QGroupBox("Monte Carlo")
        mc_form  = QFormLayout()
        mc_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.nsims_input = QLineEdit("50000")
        self.nsims_input.setValidator(QIntValidator(1000, 500000))
        mc_form.addRow("Simulations:", self.nsims_input)

        self.nsteps_input = QLineEdit("252")
        self.nsteps_input.setValidator(QIntValidator(10, 500))
        mc_form.addRow("Pas de temps:", self.nsteps_input)

        mc_group.setLayout(mc_form)
        left_layout.addWidget(mc_group)

        # --- Boutons (même style que BSM / CRR) ---
        self.fetch_data_button = QPushButton("Récupérer/Synchroniser les Données")
        self.fetch_data_button.clicked.connect(self._fetch_data)
        left_layout.addWidget(self.fetch_data_button)

        self.calculate_button = QPushButton("Calculer (Analytique + Monte Carlo)")
        self.calculate_button.clicked.connect(self._on_calculate)
        left_layout.addWidget(self.calculate_button)

        left_layout.addStretch(1)
        main_layout.addLayout(left_layout, 1)

        # ── Panneau droit ──────────────────────────────────────────────────────
        right_layout = QVBoxLayout()

        # --- Données actuelles (identique à CRR) ---
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

        # --- Résultats ---
        results_group  = QGroupBox("Résultats du Pricing")
        results_layout = QFormLayout()

        self.price_ana_label = QLabel("N/A")
        self.price_mc_label  = QLabel("N/A")
        self.stderr_mc_label = QLabel("N/A")
        self.diff_label      = QLabel("N/A")

        results_layout.addRow("Prix Analytique:",  self.price_ana_label)
        results_layout.addRow("Prix Monte Carlo:", self.price_mc_label)
        results_layout.addRow("Std Error MC:",     self.stderr_mc_label)
        results_layout.addRow("Écart Ana. / MC:",  self.diff_label)
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)

        # --- Graphique Matplotlib (même conteneur que BSM "Payoff de l'option") ---
        plot_group  = QGroupBox("Visualisation Monte Carlo")
        plot_layout = QVBoxLayout()
        self.canvas = ExoticCanvas(self)
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        right_layout.addWidget(plot_group)

        main_layout.addLayout(right_layout, 2)

        # Initialiser les widgets conditionnels
        self._on_exotic_changed()

    # =========================================================================
    # Affichage conditionnel des paramètres spécifiques
    # =========================================================================

    def _clear_specific_form(self):
        # takeRow détache la ligne sans détruire les widgets côté C++.
        # removeRow() les supprime définitivement → RuntimeError au prochain accès.
        while self.specific_form.rowCount() > 0:
            row = self.specific_form.takeRow(0)
            if row.labelItem and row.labelItem.widget():
                row.labelItem.widget().setParent(None)
            if row.fieldItem and row.fieldItem.widget():
                row.fieldItem.widget().setParent(None)

    def _on_exotic_changed(self):
        self._clear_specific_form()
        idx = self.exotic_combo.currentIndex()

        if idx == 0:    # Barrière
            self.specific_form.addRow("Niveau barrière H ($):", self.barrier_input)
            self.specific_form.addRow("Type de barrière:",       self.barrier_type_combo)
            self.strike_input.setEnabled(True)
            self.specific_group.setVisible(True)

        elif idx == 1:  # Asiatique
            self.specific_form.addRow("Moyenne:", self.averaging_combo)
            self.strike_input.setEnabled(True)
            self.specific_group.setVisible(True)

        elif idx == 2:  # Lookback — pas de strike, pas de params
            self.strike_input.setEnabled(False)
            self.specific_group.setVisible(False)

        else:           # Digitale
            self.specific_form.addRow("Montant payoff ($):", self.payoff_amount_input)
            self.strike_input.setEnabled(True)
            self.specific_group.setVisible(True)

    # =========================================================================
    # Récupération des données marché — via le mécanisme central de l'app
    # =========================================================================

    def _fetch_data(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer un symbole de ticker.")
            return
        # Utilise exactement le même chemin que le bouton "Récupérer" de BSM/CRR
        self.app.fetch_data_for_tab(ticker, self)

    # =========================================================================
    # Synchronisation depuis gui_app.py
    # update_financial_data : appelé par update_all_tabs_financial_data
    # =========================================================================

    def update_financial_data(self, ticker: str, S: Optional[float],
                               r: Optional[float], q: Optional[float],
                               sigma: Optional[float],
                               pricing_method: str = "") -> None:
        """
        Même signature que simulation_tab.update_financial_data.
        À appeler dans update_all_tabs_financial_data de gui_app.py :

            self.exotic_tab.update_financial_data(
                self.current_ticker, self.S, self.r, self.q,
                sigma_to_use, pricing_method_to_use
            )
        """
        self._ticker = ticker
        self._S      = S
        self._r      = r
        self._q      = q
        self._sigma  = sigma

        # Ticker : ne pas écraser ce que l'utilisateur a tapé lui-même
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

        # Réactiver le bouton si bloqué par le fetch
        self.fetch_data_button.setEnabled(True)
        self.fetch_data_button.setText("Récupérer/Synchroniser les Données")

        # Pré-remplir strike ATM si encore à la valeur par défaut
        if S is not None:
            try:
                if abs(float(self.strike_input.text()) - 150.0) < 0.01:
                    self.strike_input.setText(f"{S:.2f}")
            except ValueError:
                pass
            # Barrière par défaut à 120% de S
            try:
                if abs(float(self.barrier_input.text()) - 120.0) < 0.01:
                    self.barrier_input.setText(f"{S * 1.20:.2f}")
            except ValueError:
                pass

    # =========================================================================
    # Collecte des paramètres
    # =========================================================================

    def _collect_params(self) -> Optional[dict]:
        if self._S is None or self._r is None or self._q is None or self._ticker is None:
            QMessageBox.warning(
                self, "Données Manquantes",
                "Veuillez d'abord récupérer les données financières\n"
                "(bouton 'Récupérer/Synchroniser les Données')."
            )
            return None

        try:
            K = float(self.strike_input.text())
            if K <= 0 and self.strike_input.isEnabled():
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Erreur", "Prix d'exercice K invalide.")
            return None

        maturity_qdate = self.maturity_date_input.date()
        maturity_datetime = datetime(maturity_qdate.year(), maturity_qdate.month(), maturity_qdate.day())
        T = (date(maturity_qdate.year(), maturity_qdate.month(),
                  maturity_qdate.day()) - date.today()).days / 365.0
        if T <= 0:
            QMessageBox.warning(self, "Erreur",
                                "La date d'échéance doit être dans le futur.")
            return None
        
        # --- RÉCUPÉRATION DE L'IV ET DU PRIX MARCHÉ POUR LES PARAMÈTRES EXOTIQUES ---
        option_type = self.option_type_combo.currentText()
        fetched_iv, market_price, closest_date = self.data_fetcher.get_implied_volatility_and_price(
            self._ticker, K, maturity_datetime, option_type
        )
        
        # --- Logique de choix de la Volatilité (IV vs Historique) ---
        if fetched_iv is not None and fetched_iv > 0.001:
            sigma = fetched_iv
            pricing_method = "IV Marché"
        else:
            sigma = self._sigma if self._sigma is not None and self._sigma > 0 else 0.20
            pricing_method = "Vol Historique (Fallback)"
        
        # Mise à jour de l'affichage de la volatilité
        self.vol_label.setText(f"{sigma*100:.2f}% ({pricing_method})")

        try:
            n_sims  = int(self.nsims_input.text())
            n_steps = int(self.nsteps_input.text())
        except ValueError:
            QMessageBox.warning(self, "Erreur", "Paramètres Monte Carlo invalides.")
            return None

        idx_map = {0: "barrier", 1: "asian", 2: "lookback", 3: "digital"}
        exotic  = idx_map[self.exotic_combo.currentIndex()]

        try:
            barrier = float(self.barrier_input.text())
        except ValueError:
            barrier = self._S * 1.20 if self._S else 120.0

        try:
            payoff_amount = float(self.payoff_amount_input.text())
        except ValueError:
            payoff_amount = 1.0

        if exotic == "barrier" and barrier <= 0:
            QMessageBox.warning(self, "Erreur",
                                "Niveau de barrière H invalide (doit être > 0).")
            return None

        return {
            "exotic_type"  : exotic,
            "option_type"  : self.option_type_combo.currentText(),
            "S"            : self._S,
            "K"            : K,
            "T"            : T,
            "r"            : self._r,
            "sigma"        : sigma,  # Utiliser la sigma récupérée du marché
            "q"            : self._q,
            "barrier"      : barrier,
            "barrier_type" : self.barrier_type_combo.currentText(),
            "averaging"    : self.averaging_combo.currentText(),
            "payoff_amount": payoff_amount,
            "n_sims"       : n_sims,
            "n_steps"      : n_steps,
        }

    # =========================================================================
    # Lancement du calcul
    # =========================================================================

    def _on_calculate(self):
        params = self._collect_params()
        if params is None:
            return

        self.calculate_button.setEnabled(False)
        self.calculate_button.setText("⏳ Calcul en cours...")
        for lbl in (self.price_ana_label, self.price_mc_label,
                    self.stderr_mc_label, self.diff_label):
            lbl.setText("...")
            lbl.setStyleSheet("")   # Réinitialise le style (gris/italique éventuel)

        self._worker = PricingWorker(params)
        self._worker.result_ready.connect(
            lambda ana, mc: self._on_result(ana, mc, params))
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._reset_button)
        self._worker.start()

    # =========================================================================
    # Callbacks
    # =========================================================================

    def _on_result(self, ana: Optional[ExoticResult], mc: ExoticResult, params: dict):
        # Prix analytique — N/A pour asiatique et lookback
        if ana is not None:
            self.price_ana_label.setText(f"${ana.price:.4f}")
        else:
            self.price_ana_label.setText("N/A — Monte Carlo uniquement")
            self.price_ana_label.setStyleSheet("color: gray; font-style: italic;")

        # Prix Monte Carlo
        self.price_mc_label.setText(f"${mc.price:.4f}")
        self.stderr_mc_label.setText(
            f"±{mc.std_error:.4f}" if mc.std_error is not None else "N/A")

        # Écart — seulement si les deux prix sont disponibles
        if ana is not None:
            diff = abs(ana.price - mc.price)
            pct  = diff / ana.price * 100 if ana.price > 0 else 0.0
            self.diff_label.setText(f"${diff:.4f}  ({pct:.2f}%)")
        else:
            self.diff_label.setText("N/A — pas de référence analytique")
            self.diff_label.setStyleSheet("color: gray; font-style: italic;")

        self.canvas.plot_results(mc, params)

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Erreur de calcul", msg)
        self._reset_button()

    def _reset_button(self):
        self.calculate_button.setEnabled(True)
        self.calculate_button.setText("Calculer (Analytique + Monte Carlo)")
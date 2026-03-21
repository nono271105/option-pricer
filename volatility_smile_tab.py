# volatility_smile_tab.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLineEdit,
    QPushButton, QFormLayout, QMessageBox, QDateEdit
)
from PyQt5.QtCore import QDate
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime, date
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import pandas as pd

from data_fetcher import DataFetcher
from option_models import OptionModels


class VolatilitySmileTab(QWidget):
    """
    Onglet dédié à l'affichage du Sourire de Volatilité avec interpolation linéaire.
    Logique : Puts OTM à gauche (K < S), Calls OTM à droite (K >= S).
    Calcul de l'IV via inversion de BSM à partir du prix Mid.
    """
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.data_fetcher: DataFetcher = DataFetcher()
        self.option_models: OptionModels = OptionModels()
        self.current_S: Optional[float] = None
        self.current_r: float = 0.05
        self.current_q: float = 0.0

        self.init_ui()

    def init_ui(self) -> None:
        """Initialise l'interface utilisateur."""
        main_layout = QVBoxLayout(self)

        # ------------------- INPUT GROUP -------------------
        input_group = QGroupBox("Paramètres du Sourire de Volatilité")
        input_layout = QFormLayout()

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("Ex: AAPL")
        
        self.maturity_date_input = QDateEdit(QDate.currentDate().addMonths(1))
        self.maturity_date_input.setCalendarPopup(True)
        self.maturity_date_input.setDisplayFormat("yyyy-MM-dd")
        
        self.plot_button = QPushButton("Afficher le Sourire de Volatilité")
        self.plot_button.clicked.connect(self.plot_volatility_smile)

        input_layout.addRow("Ticker Symbole:", self.ticker_input)
        input_layout.addRow("Date d'Échéance:", self.maturity_date_input)
        input_layout.addRow(self.plot_button)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # ------------------- PLOT AREA -------------------
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Sourire de Volatilité")
        self.ax.set_xlabel("Strike (K)")
        self.ax.set_ylabel("Volatilité Implicite (%)")
        self.ax.grid(True)
        self.canvas.draw()

    def update_S(self, S):
        """Met à jour le prix actuel de l'actif sous-jacent."""
        self.current_S = S

    def update_financial_params(self, r, q):
        """Met à jour r et q pour le calcul BSM."""
        self.current_r = r if r is not None else 0.05
        self.current_q = q if q is not None else 0.0

    def calculate_iv_from_price(self, market_price, S, K, T, r, q, option_type):
        """
        Inverse le modèle Black-Scholes pour trouver la volatilité implicite.
        Utilise la méthode de Brent (brentq) pour résoudre : BS_price(sigma) - market_price = 0
        """
        if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return None
        
        # Valeur intrinsèque
        if option_type == 'call':
            intrinsic = max(0, S - K)
        else:
            intrinsic = max(0, K - S)
        
        # Le prix du marché doit être >= valeur intrinsèque
        if market_price < intrinsic:
            return None

        def objective(sigma):
            try:
                bs_price = self.option_models.black_scholes_price(
                    S, K, T, r, sigma, q, option_type
                )
                return bs_price - market_price
            except:
                return 1e10  # Valeur arbitraire en cas d'erreur

        try:
            # Recherche de sigma entre 0.01 (1%) et 3.0 (300%)
            iv = brentq(objective, 0.01, 3.0, xtol=1e-6, maxiter=100)
            return iv
        except:
            return None

    def plot_volatility_smile(self):
        ticker = self.ticker_input.text().upper().strip()
        maturity_date_str = self.maturity_date_input.date().toString("yyyy-MM-dd")

        if not ticker:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer un ticker.")
            return
        
        try:
            # 1. Récupération du prix actuel
            current_price = self.data_fetcher.get_live_price(ticker)
            if current_price is None or current_price <= 0:
                QMessageBox.warning(self, "Erreur", "Impossible de récupérer le prix actuel.")
                return
            
            self.current_S = current_price

            # 2. Récupération de r et q
            self.current_r = self.data_fetcher.get_sofr_rate() or 0.05
            self.current_q = self.data_fetcher.get_dividend_yield(ticker) or 0.0

            # 3. Récupération de la chaîne d'options
            maturity_date = datetime.strptime(maturity_date_str, "%Y-%m-%d").date()
            opt_chain, closest_date_str = self.data_fetcher.get_option_data_chain(ticker, datetime.combine(maturity_date, datetime.min.time()))

            if opt_chain is None or closest_date_str is None:
                QMessageBox.warning(self, "Données", f"Aucune chaîne d'options trouvée pour {ticker}.")
                return
            
            # 4. Calcul du temps jusqu'à maturité
            closest_date = datetime.strptime(closest_date_str, '%Y-%m-%d').date()
            today = date.today()
            T = (closest_date - today).days / 365.0
            
            if T <= 0:
                QMessageBox.warning(self, "Maturité", "La date d'échéance est dans le passé.")
                return

            # 5. Séparation Calls et Puts
            calls = opt_chain.calls.copy()
            puts = opt_chain.puts.copy()

            # Calcul du prix Mid (Bid + Ask) / 2
            calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
            puts['mid_price'] = (puts['bid'] + puts['ask']) / 2

            # Nettoyage : on garde seulement les options avec un prix > 0
            calls = calls[(calls['mid_price'] > 0) & (calls['bid'] > 0) & (calls['ask'] > 0)]
            puts = puts[(puts['mid_price'] > 0) & (puts['bid'] > 0) & (puts['ask'] > 0)]

            # 6. FILTRAGE OTM
            # Puts OTM : Strike < Prix Actuel (gauche du smile)
            puts_otm = puts[puts['strike'] < current_price].copy()
            
            # Calls OTM : Strike >= Prix Actuel (droite du smile)
            calls_otm = calls[calls['strike'] >= current_price].copy()

            if puts_otm.empty and calls_otm.empty:
                QMessageBox.warning(self, "Données", "Aucune option OTM trouvée.")
                return

            # 7. CALCUL DE L'IV PAR INVERSION DE BSM
            iv_data = []

            # Pour les Puts OTM
            for _, row in puts_otm.iterrows():
                K = row['strike']
                market_price = row['mid_price']
                
                iv = self.calculate_iv_from_price(
                    market_price, current_price, K, T, 
                    self.current_r, self.current_q, 'put'
                )
                
                if iv is not None and 0.01 < iv < 3.0:  # Filtrage IV raisonnable
                    iv_data.append({
                        'strike': K,
                        'iv': iv,
                        'type': 'put'
                    })

            # Pour les Calls OTM
            for _, row in calls_otm.iterrows():
                K = row['strike']
                market_price = row['mid_price']
                
                iv = self.calculate_iv_from_price(
                    market_price, current_price, K, T,
                    self.current_r, self.current_q, 'call'
                )
                
                if iv is not None and 0.01 < iv < 3.0:
                    iv_data.append({
                        'strike': K,
                        'iv': iv,
                        'type': 'call'
                    })

            if not iv_data:
                QMessageBox.warning(self, "Calcul IV", "Impossible de calculer l'IV pour les options disponibles.")
                return

            # 8. Création du DataFrame et tri
            smile_df = pd.DataFrame(iv_data)
            smile_df = smile_df.sort_values('strike')
            smile_df = smile_df.drop_duplicates(subset=['strike'])  # Suppression des duplicates

            if len(smile_df) < 2:
                QMessageBox.warning(self, "Données", "Pas assez de points pour tracer le smile.")
                return

            # 9. INTERPOLATION LINÉAIRE pour les strikes manquants
            strikes = smile_df['strike'].values
            ivs = smile_df['iv'].values * 100  # Conversion en %

            # Création d'un axe de strikes plus dense
            strike_min = strikes.min()
            strike_max = strikes.max()
            strikes_interp = np.linspace(strike_min, strike_max, 200)

            # Interpolation linéaire
            f_interp = interp1d(strikes, ivs, kind='linear', fill_value='extrapolate')
            ivs_interp = f_interp(strikes_interp)

            # 10. TRACÉ
            self.ax.clear()

            # Courbe interpolée
            self.ax.plot(strikes_interp, ivs_interp, color="#0062FF", linewidth=2, label='Smile')

            # Points OTM réels
            puts_df = smile_df[smile_df['type'] == 'put']
            calls_df = smile_df[smile_df['type'] == 'call']

            # Ligne verticale du prix actuel (ATM)
            self.ax.axvline(current_price, color='red', linestyle='--', 
                           linewidth=1.5, alpha=0.7, label=f'Spot: {current_price:.2f}')

            # Mise en forme
            self.ax.set_title(f"Sourire de Volatilité : {ticker} (Exp: {closest_date_str})", 
                             fontsize=12, fontweight='bold')
            self.ax.set_xlabel("Strike ($)", fontsize=10)
            self.ax.set_ylabel("Volatilité Implicite (%)", fontsize=10)
            self.ax.grid(True, linestyle=':', alpha=0.6)
            self.ax.legend(loc='best')
            
            # Zoom intelligent sur Y
            y_min, y_max = ivs.min(), ivs.max()
            margin = (y_max - y_min) * 0.1
            self.ax.set_ylim(max(0, y_min - margin), y_max + margin)

            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur de tracé : {str(e)}")
            import traceback
            traceback.print_exc()
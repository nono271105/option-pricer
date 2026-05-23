"""
Logique métier pour la simulation de prix d'options (matrice Volatilité / Prix).
"""

import numpy as np
from typing import Literal, Tuple, List


class SimulationLogic:
    """Génération de matrices de simulation BSM-Merton (Call ou Put)."""

    def __init__(self, option_models):
        self.option_models = option_models

    def run_simulation(
        self,
        K: float,
        T: float,
        r: float,
        q: float,
        vol_min: int,
        vol_max: int,
        vol_step: int,
        underlying_min: int,
        underlying_max: int,
        underlying_step: int,
        option_type: Literal['call', 'put'] = 'call',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """
        Génère une matrice de prix d'options BSM-Merton pour différentes
        volatilités (lignes) et prix sous-jacents (colonnes).

        Args:
            K              : Prix d'exercice
            T              : Maturité en années
            r              : Taux sans risque (annualisé)
            q              : Taux de dividende continu (annualisé)
            vol_min/max    : Bornes de volatilité en % (entiers)
            vol_step       : Pas de volatilité en %
            underlying_min/max : Bornes de prix sous-jacent (entiers)
            underlying_step: Pas de prix sous-jacent
            option_type    : 'call' ou 'put'

        Returns:
            (volatilities_percent, underlying_prices, results_matrix, all_prices)
        """
        volatilities_percent = np.arange(vol_min, vol_max + vol_step, vol_step)
        underlying_prices = np.arange(underlying_min, underlying_max + underlying_step, underlying_step)

        if len(volatilities_percent) == 0 or len(underlying_prices) == 0:
            return np.array([]), np.array([]), np.array([]), []

        sigma_array = (volatilities_percent / 100.0)[:, None]
        S_array = underlying_prices[None, :]

        # valorisation vectorisée via le modèle BSM
        results_matrix = self.option_models.black_scholes_price(
            S=S_array,
            K=K,
            T=T,
            r=r,
            sigma=sigma_array,
            q=q,
            option_type=option_type,
        )
        
        results_matrix = np.array(results_matrix)
        all_prices = results_matrix.flatten().tolist()

        return volatilities_percent, underlying_prices, results_matrix, all_prices

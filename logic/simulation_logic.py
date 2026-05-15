"""
Logique métier pour la simulation de prix d'options (matrice Volatilité / Prix).
"""

import numpy as np
from typing import Tuple, List

class SimulationLogic:
    """Logique de génération de matrices de simulation BSM."""
    
    def __init__(self, option_models):
        self.option_models = option_models

    def run_simulation(self, K: float, T: float, r: float, q: float, 
                       vol_min: int, vol_max: int, vol_step: int,
                       underlying_min: int, underlying_max: int, underlying_step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """
        Génère une matrice de prix d'options (Calls) pour différentes volatilités et prix sous-jacents.
        
        Returns:
            Tuple contenant:
            - volatilities_percent (1D array)
            - underlying_prices (1D array)
            - results_matrix (2D array: [volatilities, underlying_prices])
            - all_prices (liste plate de tous les prix calculés pour trouver min/max)
        """
        volatilities_percent = np.arange(vol_min, vol_max + vol_step, vol_step)
        underlying_prices = np.arange(underlying_min, underlying_max + underlying_step, underlying_step)

        if len(volatilities_percent) == 0 or len(underlying_prices) == 0:
            return np.array([]), np.array([]), np.array([]), []

        all_prices = []
        results_matrix = np.zeros((len(volatilities_percent), len(underlying_prices)))

        for i, vol_percent in enumerate(volatilities_percent):
            sigma = vol_percent / 100.0
            for j, S in enumerate(underlying_prices):
                price = self.option_models.black_scholes_price(
                    S=float(S), K=K, T=T, r=r, sigma=sigma, q=q, option_type='call'
                )
                results_matrix[i, j] = price
                all_prices.append(price)

        return volatilities_percent, underlying_prices, results_matrix, all_prices

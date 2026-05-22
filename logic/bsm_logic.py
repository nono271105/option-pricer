import numpy as np
from scipy.stats import norm
from typing import Dict, Literal

class OptionModels:
    def black_scholes_price(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float, 
        q: float, 
        option_type: Literal['call', 'put'] = 'call'
    ) -> float:
        """
        Calcule le prix d'une option européenne en utilisant le modèle Black-Scholes.
        
        Args:
            S: Prix actuel de l'actif sous-jacent
            K: Prix d'exercice (strike)
            T: Temps jusqu'à expiration (en années)
            r: Taux sans risque annualisé
            sigma: Volatilité annualisée
            q: Rendement de dividende annualisé
            option_type: 'call' ou 'put'
            
        Returns:
            float: Prix théorique de l'option
        """
        if T <= 0:
            if option_type == 'call':
                return np.maximum(0, S - K)
            elif option_type == 'put':
                return np.maximum(0, K - S)
            raise ValueError("option_type doit être 'call' ou 'put'")

        # sécurisation contre les volatilités physiquement impossibles
        if sigma <= 1e-6:
            if option_type == 'call':
                return np.maximum(0, S * np.exp(-q * T) - K * np.exp(-r * T))
            elif option_type == 'put':
                return np.maximum(0, K * np.exp(-r * T) - S * np.exp(-q * T))
            raise ValueError("option_type doit être 'call' ou 'put'")

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type doit être 'call' ou 'put'")

        return price

    def calculate_greeks(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float, 
        q: float, 
        option_type: Literal['call', 'put'] = 'call'
    ) -> Dict[str, float]:
        """
        Calcule les Grecs (Delta, Gamma, Theta, Vega, Rho) pour le modèle Black-Scholes.
        
        Args:
            S: Prix actuel de l'actif sous-jacent
            K: Prix d'exercice
            T: Temps jusqu'à expiration (en années)
            r: Taux sans risque annualisé
            sigma: Volatilité annualisée
            q: Rendement de dividende annualisé
            option_type: 'call' ou 'put'
            
        Returns:
            Dict[str, float]: Dictionnaire avec clés 'delta', 'gamma', 'theta', 'vega', 'rho'
        """
        if T <= 0 or sigma <= 1e-6:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        Nd1 = norm.cdf(d1)
        n_d1 = norm.pdf(d1)

        # sensibilité du prix à une variation du sous-jacent
        if option_type == 'call':
            delta = np.exp(-q * T) * Nd1
        else: # put
            delta = -np.exp(-q * T) * norm.cdf(-d1)

        # convexité du prix par rapport au sous-jacent
        gamma = np.exp(-q * T) * n_d1 / (S * sigma * np.sqrt(T))

        # sensibilité du prix à une variation de la volatilité
        vega = S * np.exp(-q * T) * n_d1 * np.sqrt(T)

        # dépréciation temporelle de l'option
        theta_part1 = -(S * np.exp(-q * T) * n_d1 * sigma) / (2 * np.sqrt(T))
        
        if option_type == 'call':
            theta_part2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta_part3 = q * S * np.exp(-q * T) * Nd1
        else: # put
            theta_part2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta_part3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            
        theta = theta_part1 + theta_part2 + theta_part3
        theta_daily = theta / 365.0

        # sensibilité aux taux d'intérêt, normalisée par point de base
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else: # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        rho = rho / 100.0  # sensibilté à une variation de 1% du taux d'intérêt
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta_daily,
            'vega': vega,
            'rho': rho
        }

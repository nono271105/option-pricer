import numpy as np
from typing import Dict, Literal

class CRRModels:
    """
    Modèle binomial Cox-Ross-Rubinstein pour options américaines.
    Classe autonome — pas d'héritage de OptionModels.
    """

    def cox_ross_rubinstein_price(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        q: float, 
        sigma: float, 
        N: int, 
        option_type: Literal['call', 'put']
    ) -> float:
        """
        Calcule le prix d'une option Américaine en utilisant le modèle binomial CRR.
        
        Args:
            S: Prix actuel de l'actif sous-jacent
            K: Prix d'exercice
            T: Temps jusqu'à expiration (en années)
            r: Taux sans risque annualisé
            q: Rendement de dividende annualisé
            sigma: Volatilité annualisée
            N: Nombre de pas dans l'arbre binomial
            option_type: 'call' ou 'put'
            
        Returns:
            float: Prix théorique de l'option américaine
        """
        if T <= 0 or N <= 0:
            if option_type == 'call':
                return max(0, S - K)
            elif option_type == 'put':
                return max(0, K - S)
            raise ValueError("option_type doit être 'call' ou 'put'")

        dt = T / N
        df = np.exp(-r * dt)

        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)

        # 1. Initialiser le prix de l'actif aux nœuds à l'échéance (t=N)
        # S * u^j * d^(N-j) où j va de N à 0 (pour correspondre à l'ordre up-down)
        j_values = np.arange(N, -1, -1)
        stock_prices = S * (u**j_values) * (d**(N - j_values))

        # 2. Calculer la valeur de l'option à l'échéance (t=N)
        if option_type == 'call':
            option_values = np.maximum(stock_prices - K, 0)
        elif option_type == 'put':
            option_values = np.maximum(K - stock_prices, 0)
        
        # 3. Rétropropagation (Backward Induction) vectorisée
        for i in range(N - 1, -1, -1):
            # Prix de l'actif aux noeuds à l'étape i
            j_i = np.arange(i, -1, -1)
            S_nodes = S * (u**j_i) * (d**(i - j_i))
            
            # Valeur de continuation
            continuation = df * (p * option_values[:-1] + (1 - p) * option_values[1:])
            
            # Valeur d'exercice immédiat
            if option_type == 'call':
                exercise = np.maximum(S_nodes - K, 0)
            elif option_type == 'put':
                exercise = np.maximum(K - S_nodes, 0)
            
            # Option Américaine: Max(Continuation, Exercice)
            option_values = np.maximum(continuation, exercise)

        return option_values[0]

    def calculate_greeks_crr(
        self, 
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        q: float, 
        sigma: float, 
        N: int, 
        option_type: Literal['call', 'put'], 
        epsilon: float = 1.0
    ) -> Dict[str, float]:
        """
        Calcule les Grecs du modèle CRR par différences finies.
        Méthode numérique pour options non-européennes (exercice anticipé possible).
        
        Args:
            S: Prix actuel de l'actif sous-jacent
            K: Prix d'exercice
            T: Temps jusqu'à expiration (en années)
            r: Taux sans risque annualisé
            q: Rendement de dividende annualisé
            sigma: Volatilité annualisée
            N: Nombre de pas dans l'arbre binomial
            option_type: 'call' ou 'put'
            epsilon: Pas pour les différences finies (par défaut 1.0 pour meilleure précision numérique)
            
        Returns:
            Dict[str, float]: Dictionnaire avec clés 'delta', 'gamma', 'theta', 'vega', 'rho'
        """
        
        # Pour Delta, Gamma, Theta, on reconstruit l'arbre jusqu'à t=2 pour
        # éviter les instabilités numériques sévères des différences finies
        # sur un modèle discret (le prix CRR n'est pas lisse par rapport à S et T).
        
        dt = T / N
        df = np.exp(-r * dt)
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        j_values = np.arange(N, -1, -1)
        stock_prices = S * (u**j_values) * (d**(N - j_values))
        
        if option_type == 'call':
            option_values = np.maximum(stock_prices - K, 0)
        else:
            option_values = np.maximum(K - stock_prices, 0)
            
        C2, C1, C0 = None, None, None
        
        for i in range(N - 1, -1, -1):
            j_i = np.arange(i, -1, -1)
            S_nodes = S * (u**j_i) * (d**(i - j_i))
            
            continuation = df * (p * option_values[:-1] + (1 - p) * option_values[1:])
            
            if option_type == 'call':
                exercise = np.maximum(S_nodes - K, 0)
            else:
                exercise = np.maximum(K - S_nodes, 0)
                
            option_values = np.maximum(continuation, exercise)
            
            if i == 2: C2 = option_values.copy()
            elif i == 1: C1 = option_values.copy()
            elif i == 0: C0 = option_values[0]
            
        # Delta (à partir des nœuds t=1)
        # C1[0] est le nœud up (S*u), C1[1] est le nœud down (S*d)
        delta = (C1[0] - C1[1]) / (S*u - S*d)
        
        # Gamma (à partir des nœuds t=2)
        # C2 = [up-up, up-down, down-down]
        delta_up = (C2[0] - C2[1]) / (S*u**2 - S)
        delta_dn = (C2[1] - C2[2]) / (S - S*d**2)
        gamma = (delta_up - delta_dn) / ((S*u**2 - S*d**2) / 2.0)
        
        # Theta (à partir des nœuds t=2 et t=0)
        # C2[1] est le nœud où l'actif vaut S après un up et un down (S*u*d = S)
        theta_annual = (C2[1] - C0) / (2 * dt)
        theta_daily = theta_annual / 365.0

        # Vega et Rho n'ont pas de formule simple dans l'arbre, on garde les différences finies
        def crr_price(sigma_local: float, r_local: float) -> float:
            return self.cox_ross_rubinstein_price(S, K, T, r_local, q, sigma_local, N, option_type)

        # Vega (dérivée par rapport à sigma)
        eps_sigma = 0.01
        sigma_plus  = sigma + eps_sigma
        sigma_minus = max(sigma - eps_sigma, 1e-4)
        vega = (crr_price(sigma_plus, r) - crr_price(sigma_minus, r)) / (sigma_plus - sigma_minus)
        
        # Rho (dérivée par rapport à r)
        eps_r = 0.0001
        rho_val = (crr_price(sigma, r + eps_r) - crr_price(sigma, r - eps_r)) / (2 * eps_r)
        rho_val = rho_val / 100.0

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta_daily,
            'vega': vega,
            'rho': rho_val
        }

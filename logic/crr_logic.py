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
        if option_type not in ('call', 'put'):
            raise ValueError("option_type doit être 'call' ou 'put'")

        if T <= 0 or N <= 0:
            if option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)

        if N > 5000:
            raise ValueError("N ne peut pas dépasser 5 000 (risque d'épuisement mémoire)")

        dt = T / N
        df = np.exp(-r * dt)

        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)

        if p < 0 or p > 1:
            raise ValueError(
                f"Probabilité risque-neutre p={p:.4f} hors [0,1]. "
                f"Vérifiez les paramètres (r={r}, q={q}, sigma={sigma}, N={N}).")

        # construction vectorisée des prix terminaux du sous-jacent à maturité
        j_values = np.arange(N, -1, -1)
        stock_prices = S * (u**j_values) * (d**(N - j_values))

        # valorisation intrinsèque aux nœuds finaux de l'arbre
        if option_type == 'call':
            option_values = np.maximum(stock_prices - K, 0)
        elif option_type == 'put':
            option_values = np.maximum(K - stock_prices, 0)
        
        # remontée de l'arbre par induction rétrograde pour valoriser l'option
        for i in range(N - 1, -1, -1):
            # états intermédiaires du sous-jacent
            j_i = np.arange(i, -1, -1)
            S_nodes = S * (u**j_i) * (d**(i - j_i))
            
            # valeur actualisée si l'option est conservée
            continuation = df * (p * option_values[:-1] + (1 - p) * option_values[1:])
            
            # payoff en cas d'exercice anticipé
            if option_type == 'call':
                exercise = np.maximum(S_nodes - K, 0)
            elif option_type == 'put':
                exercise = np.maximum(K - S_nodes, 0)
            
            # la prime américaine prime toujours sur l'exercice si elle est supérieure
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
            
        Returns:
            Dict[str, float]: Dictionnaire avec clés 'delta', 'gamma', 'theta', 'vega', 'rho'
        """
        if option_type not in ('call', 'put'):
            raise ValueError("option_type doit être 'call' ou 'put'")

        if N > 5000:
            raise ValueError("N ne peut pas dépasser 5 000 (risque d'épuisement mémoire)")
        
        # les différences finies directes étant instables sur CRR, l'extraction 
        # des Grecs s'effectue directement depuis la géométrie des premiers nœuds
        
        dt = T / N
        df = np.exp(-r * dt)
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)

        if p < 0 or p > 1:
            raise ValueError(
                f"Probabilité risque-neutre p={p:.4f} hors [0,1]. "
                f"Vérifiez les paramètres (r={r}, q={q}, sigma={sigma}, N={N}).")
        
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
            
        # extraction du delta depuis le premier pas binomial
        delta = (C1[0] - C1[1]) / (S*u - S*d)
        
        # le gamma nécessite la concavité mesurée au second pas
        delta_up = (C2[0] - C2[1]) / (S*u**2 - S)
        delta_dn = (C2[1] - C2[2]) / (S - S*d**2)
        gamma = (delta_up - delta_dn) / ((S*u**2 - S*d**2) / 2.0)
        
        # approximation de la perte de valeur temporelle
        theta_annual = (C2[1] - C0) / (2 * dt)
        theta_daily = theta_annual / 365.0

        # calcul numérique pour les sensibilités de second ordre
        def crr_price(sigma_local: float, r_local: float) -> float:
            return self.cox_ross_rubinstein_price(S, K, T, r_local, q, sigma_local, N, option_type)

        # choc infinitésimal sur la volatilité
        eps_sigma = 0.01
        sigma_plus  = sigma + eps_sigma
        sigma_minus = max(sigma - eps_sigma, 1e-4)
        vega = (crr_price(sigma_plus, r) - crr_price(sigma_minus, r)) / (sigma_plus - sigma_minus)
        
        # choc infinitésimal sur le taux sans risque
        eps_r = 0.0001
        rho_val = (crr_price(sigma, r + eps_r) - crr_price(sigma, r - eps_r)) / (2 * eps_r)
        rho_val = rho_val / 100.0 # sensibilté à une variation de 1% du taux d'intérêt

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta_daily,
            'vega': vega,
            'rho': rho_val
        }

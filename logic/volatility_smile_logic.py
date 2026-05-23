# Moteur de calcul et d'interpolation de la courbe de sourire de volatilité

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from typing import Optional, Tuple, Any

from logic.bsm_logic import OptionModels

class VolatilitySmileLogic:
    """Logique de calcul du sourire de volatilité."""
    
    def __init__(self):
        self.option_models = OptionModels()

    def calculate_iv_from_price(self, market_price: float, S: float, K: float, T: float, r: float, q: float, option_type: str) -> Optional[float]:
        """Inverse le modèle BSM pour trouver l'IV."""
        if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return None
            
        intrinsic = max(0, S * np.exp(-q*T) - K * np.exp(-r*T)) if option_type == 'call' else max(0, K * np.exp(-r*T) - S * np.exp(-q*T))
        
        if market_price < intrinsic:
            return None

        def objective(sigma):
            try:
                bs_price = self.option_models.black_scholes_price(S, K, T, r, sigma, q, option_type)
                return bs_price - market_price
            except Exception:
                return 1e10

        try:
            iv = brentq(objective, 0.01, 3.0, xtol=1e-6, maxiter=100)
            return iv
        except Exception:
            return None

    def process_smile_data(self, opt_chain: Any, current_price: float, T: float, r: float, q: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Traite les données d'options et retourne les points interpolés et les données réelles."""
        calls = opt_chain.calls.copy()
        puts = opt_chain.puts.copy()

        calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
        puts['mid_price'] = (puts['bid'] + puts['ask']) / 2

        calls = calls[(calls['mid_price'] > 0) & (calls['bid'] > 0) & (calls['ask'] > 0)]
        puts = puts[(puts['mid_price'] > 0) & (puts['bid'] > 0) & (puts['ask'] > 0)]

        puts_otm = puts[puts['strike'] < current_price].copy()
        calls_otm = calls[calls['strike'] >= current_price].copy()

        if puts_otm.empty and calls_otm.empty:
            return None, None, None

        iv_data = []

        for _, row in puts_otm.iterrows():
            K = row['strike']
            market_price = row['mid_price']
            iv = self.calculate_iv_from_price(market_price, current_price, K, T, r, q, 'put')
            if iv is not None and 0.01 < iv < 3.0:
                iv_data.append({'strike': K, 'iv': iv, 'type': 'put'})

        for _, row in calls_otm.iterrows():
            K = row['strike']
            market_price = row['mid_price']
            iv = self.calculate_iv_from_price(market_price, current_price, K, T, r, q, 'call')
            if iv is not None and 0.01 < iv < 3.0:
                iv_data.append({'strike': K, 'iv': iv, 'type': 'call'})

        if not iv_data:
            return None, None, None

        smile_df = pd.DataFrame(iv_data)
        smile_df = smile_df.sort_values('strike').drop_duplicates(subset=['strike'])

        if len(smile_df) < 2:
            return None, None, smile_df

        strikes = smile_df['strike'].values
        ivs = smile_df['iv'].values * 100

        strikes_interp = np.linspace(strikes.min(), strikes.max(), 200)
        if len(strikes) >= 4:
            f_interp = interp1d(strikes, ivs, kind='cubic', fill_value=(ivs[0], ivs[-1]), bounds_error=False)
        else:
            f_interp = interp1d(strikes, ivs, kind='linear', fill_value=(ivs[0], ivs[-1]), bounds_error=False)
        ivs_interp = f_interp(strikes_interp)
        ivs_interp = np.clip(ivs_interp, 1.0, 300.0)

        return strikes_interp, ivs_interp, smile_df

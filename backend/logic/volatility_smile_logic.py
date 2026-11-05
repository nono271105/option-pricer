# Moteur de calcul et d'interpolation de la courbe de sourire de volatilité

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any

from logic.bsm_logic import OptionModels
from logic.volatility_resolver import (
    implied_volatility_from_price,
    valid_mid_quote,
)

class VolatilitySmileLogic:
    """Logique de calcul du sourire de volatilité."""
    
    def __init__(self):
        self.option_models = OptionModels()

    def calculate_iv_from_price(self, market_price: float, S: float, K: float, T: float, r: float, q: float, option_type: str) -> Optional[float]:
        """Inverse le modèle BSM pour trouver l'IV."""
        return implied_volatility_from_price(
            market_price, S, K, T, r, q, option_type, self.option_models
        )

    def process_smile_data(self, opt_chain: Any, current_price: float, T: float, r: float, q: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Traite les données d'options et retourne les points interpolés et les données réelles."""
        calls = opt_chain.calls.copy()
        puts = opt_chain.puts.copy()

        calls['mid_price'] = calls.apply(
            lambda row: valid_mid_quote(row, require_liquidity=True), axis=1
        )
        puts['mid_price'] = puts.apply(
            lambda row: valid_mid_quote(row, require_liquidity=True), axis=1
        )
        calls = calls[calls['mid_price'].notna()]
        puts = puts[puts['mid_price'].notna()]
        forward = current_price * np.exp((r - q) * T)
        puts_otm = puts[puts['strike'] < forward].copy()
        calls_otm = calls[calls['strike'] >= forward].copy()

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
        log_moneyness = np.log(strikes / forward)
        ivs = smile_df['iv'].values * 100
        sort_idx = np.argsort(log_moneyness)
        log_moneyness = log_moneyness[sort_idx]
        ivs = ivs[sort_idx]

        strikes_interp = np.linspace(strikes.min(), strikes.max(), 200)
        target_moneyness = np.log(strikes_interp / forward)
        ivs_interp = np.interp(target_moneyness, log_moneyness, ivs)

        return strikes_interp, ivs_interp, smile_df

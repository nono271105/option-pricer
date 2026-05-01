"""
Logique métier pour le forecast TimesFM et le repricing BSM associé.
"""

import numpy as np
from typing import Tuple, Dict, Any

class ForecastLogic:
    """Logique pour générer des prévisions avec TimesFM et repricer l'option."""

    def __init__(self, option_models):
        self.option_models = option_models

    def run_forecast(self, ticker: str, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Exécute l'inférence TimesFM.
        ATTENTION : Cette méthode doit être appelée dans un thread séparé pour ne pas bloquer l'UI.
        """
        import yfinance as yf
        import timesfm

        tk = yf.Ticker(ticker)
        hist = tk.history(period="1y")
        if hist.empty or len(hist) < 30:
            raise ValueError(f"Historique insuffisant pour {ticker} ({len(hist)} points). Minimum requis : 30.")

        prices = hist["Close"].values.astype(np.float32)

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
            torch_compile=False,
        )

        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
            )
        )

        point_forecast, quantile_forecast = model.forecast(
            horizon=horizon,
            inputs=[prices],
        )

        return np.array(point_forecast), np.array(quantile_forecast), prices

    def process_forecast_results(
        self, point_forecast: np.ndarray, history_prices: np.ndarray,
        horizon: int, K: float, T_total: float, r: float, sigma: float, q: float, option_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcule les prix d'options et les grecs (delta) sur l'historique et le forecast.
        
        Returns:
            pf (point forecast array 1D)
            option_prices (array 1D)
            deltas (array 1D)
            hist_slice (array 1D)
            hist_option_prices (array 1D)
            hist_deltas (array 1D)
            x_hist (array 1D)
        """
        pf = point_forecast[0]

        option_prices = []
        deltas = []
        for i in range(horizon):
            S_i = float(pf[i])
            T_i = max(T_total - (i + 1) / 365.0, 1.0 / 365.0)

            price_i = self.option_models.black_scholes_price(S_i, K, T_i, r, sigma, q, option_type)
            greeks_i = self.option_models.calculate_greeks(S_i, K, T_i, r, sigma, q, option_type)
            
            option_prices.append(price_i)
            deltas.append(greeks_i.get("delta", 0.0))

        option_prices = np.array(option_prices)
        deltas = np.array(deltas)

        n_hist_display = min(30, len(history_prices))
        hist_slice = history_prices[-n_hist_display:]
        x_hist = np.arange(-n_hist_display, 0)

        hist_option_prices = []
        hist_deltas = []
        for i, days_offset in enumerate(x_hist):
            S_hist = float(hist_slice[i])
            T_hist = max(T_total - days_offset / 365.0, 1.0 / 365.0)
            
            price_h = self.option_models.black_scholes_price(S_hist, K, T_hist, r, sigma, q, option_type)
            greeks_h = self.option_models.calculate_greeks(S_hist, K, T_hist, r, sigma, q, option_type)
            
            hist_option_prices.append(price_h)
            hist_deltas.append(greeks_h.get("delta", 0.0))

        hist_option_prices = np.array(hist_option_prices)
        hist_deltas = np.array(hist_deltas)

        return pf, option_prices, deltas, hist_slice, hist_option_prices, hist_deltas, x_hist

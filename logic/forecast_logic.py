"""
Logique metier pour le forecast de volatilite implicite (IV) avec TimesFM
et le repricing BSM associe.

Pipeline :
  1. Recuperer l'historique des prix d'options via marketdata.app
  2. Inverser BSM (Brent) pour recalculer l'IV historique a partir des prix mid
  3. Alimenter TimesFM avec la serie d'IV
  4. Repricing BSM jour par jour avec l'IV predite (S, K constants)
"""

import numpy as np
import logging
from typing import Tuple, Optional, List
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


class ForecastLogic:
    """Prevision d'IV par TimesFM et repricing BSM a IV predite."""

    def __init__(self, option_models):
        self.option_models = option_models

    def compute_iv_from_prices(
        self,
        mid_prices: List[float],
        underlying_prices: List[float],
        dtes: List[int],
        strike: float,
        r: float,
        q: float,
        option_type: str,
    ) -> np.ndarray:
        """
        Recalcule la serie d'IV historique par inversion BSM (methode de Brent).

        Pour chaque point, resout : BSM(S, K, T, r, sigma, q, type) = mid_price
        Les points non inversibles (arbitrage, spread trop large) sont interpoles.

        Args:
            mid_prices: Prix mid du contrat pour chaque jour
            underlying_prices: Spot du sous-jacent pour chaque jour
            dtes: Jours restants jusqu'a expiration pour chaque jour
            strike: Prix d'exercice du contrat
            r: Taux sans risque
            q: Rendement de dividende
            option_type: 'call' ou 'put'

        Returns:
            np.ndarray: Serie d'IV (en decimal, ex: 0.25 = 25%)
        """
        n = len(mid_prices)
        iv_series = np.full(n, np.nan)

        for i in range(n):
            S_i = underlying_prices[i]
            T_i = max(dtes[i] / 365.0, 1.0 / 365.0)
            price_i = mid_prices[i]

            if price_i <= 0 or S_i <= 0:
                continue

            # la valeur intrinseque borne inferieurement le prix de l'option
            intrinsic = max(0, S_i * np.exp(-q*T_i) - strike * np.exp(-r*T_i)) if option_type == "call" else max(0, strike * np.exp(-r*T_i) - S_i * np.exp(-q*T_i))
            if price_i < intrinsic:
                continue

            def objective(sigma):
                return self.option_models.black_scholes_price(
                    S_i, strike, T_i, r, sigma, q, option_type
                ) - price_i

            try:
                iv = brentq(objective, 0.01, 5.0, xtol=1e-6, maxiter=100)
                iv_series[i] = iv
            except (ValueError, RuntimeError):
                continue

        # interpolation lineaire des points manquants
        valid_mask = ~np.isnan(iv_series)
        if valid_mask.sum() >= 2:
            indices = np.arange(n)
            iv_series = np.interp(indices, indices[valid_mask], iv_series[valid_mask])
        elif valid_mask.sum() == 1:
            iv_series[:] = iv_series[valid_mask][0]
        else:
            logger.warning("Aucun point d'IV inversible. Fallback a 20%.")
            iv_series[:] = 0.20

        return iv_series

    def run_iv_forecast(
        self, iv_history: np.ndarray, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute l'inference TimesFM sur la serie d'IV historique.

        Args:
            iv_history: Serie d'IV historique (en decimal)
            horizon: Nombre de jours a predire

        Returns:
            Tuple contenant (point_forecast, quantile_forecast, iv_history)
        """
        import torch
        import timesfm

        if torch.cuda.is_available():
            logger.info("[Forecast IV] CUDA detecte, TimesFM utilisera le GPU (%s).", torch.cuda.get_device_name(0))
        else:
            logger.info("[Forecast IV] CUDA non disponible, TimesFM utilisera le CPU.")

        iv_input = iv_history.astype(np.float32)

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
            inputs=[iv_input],
        )

        return np.array(point_forecast), np.array(quantile_forecast), iv_history

    def process_iv_forecast_results(
        self,
        iv_point_forecast: np.ndarray,
        iv_history: np.ndarray,
        horizon: int,
        K: float,
        T_total: float,
        S: float,
        r: float,
        q: float,
        option_type: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reprice BSM jour par jour en injectant l'IV predite.

        Le spot S reste constant (valeur actuelle). Seuls l'IV et le temps
        jusqu'a maturite varient d'un jour a l'autre.

        Args:
            iv_point_forecast: Previsions d'IV (shape: [1, horizon])
            iv_history: Serie d'IV historique
            horizon: Nombre de jours predits
            K: Prix d'exercice
            T_total: Temps restant jusqu'a maturite (annees)
            S: Prix spot actuel (constant sur tout l'horizon)
            r: Taux sans risque
            q: Rendement de dividende
            option_type: 'call' ou 'put'

        Returns:
            Tuple : (iv_forecast, option_prices, deltas,
                     iv_hist_slice, hist_option_prices, hist_deltas, x_hist)
        """
        iv_fc = iv_point_forecast[0]

        # repricing sur l'horizon de prevision
        option_prices = []
        deltas = []
        for i in range(horizon):
            sigma_i = float(iv_fc[i])
            # borne inferieure pour eviter les IV negatives du modele
            sigma_i = max(sigma_i, 0.01)
            T_i = max(T_total - (i + 1) / 365.0, 1.0 / 365.0)

            price_i = self.option_models.black_scholes_price(S, K, T_i, r, sigma_i, q, option_type)
            greeks_i = self.option_models.calculate_greeks(S, K, T_i, r, sigma_i, q, option_type)

            option_prices.append(price_i)
            deltas.append(greeks_i.get("delta", 0.0))

        option_prices = np.array(option_prices)
        deltas = np.array(deltas)

        # tranche historique pour la continuite du trace
        n_hist_display = min(30, len(iv_history))
        iv_hist_slice = iv_history[-n_hist_display:]
        x_hist = np.arange(-n_hist_display, 0)

        # repricing historique avec les IV reelles
        hist_option_prices = []
        hist_deltas = []
        for i, days_offset in enumerate(x_hist):
            sigma_hist = float(iv_hist_slice[i])
            T_hist = max(T_total - days_offset / 365.0, 1.0 / 365.0)

            price_h = self.option_models.black_scholes_price(S, K, T_hist, r, sigma_hist, q, option_type)
            greeks_h = self.option_models.calculate_greeks(S, K, T_hist, r, sigma_hist, q, option_type)

            hist_option_prices.append(price_h)
            hist_deltas.append(greeks_h.get("delta", 0.0))

        hist_option_prices = np.array(hist_option_prices)
        hist_deltas = np.array(hist_deltas)

        return iv_fc, option_prices, deltas, iv_hist_slice, hist_option_prices, hist_deltas, x_hist

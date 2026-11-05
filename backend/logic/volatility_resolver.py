"""Quality-controlled implied-volatility resolution from cached option chains."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from math import exp, isfinite, log, sqrt
from typing import Any, Optional

import numpy as np
from scipy.optimize import brentq

from cache import global_cache_iv
from data_fetcher import DataFetcher
from logic.bsm_logic import OptionModels

logger = logging.getLogger(__name__)

MIN_IV = 0.01
MAX_IV = 3.0
MAX_SPREAD_RATIO = 0.50
MIN_NEIGHBOR_OPEN_INTEREST = 10
MAX_NEIGHBOR_EXPIRIES = 2


@dataclass(frozen=True)
class VolatilityEstimate:
    sigma: float
    source: str
    market_price: Optional[float] = None
    expiry_used: Optional[str] = None
    note: Optional[str] = None


def valid_iv(value: Any) -> Optional[float]:
    try:
        iv = float(value)
    except (TypeError, ValueError):
        return None
    return iv if isfinite(iv) and MIN_IV <= iv <= MAX_IV else None


def valid_mid_quote(row: Any, *, require_liquidity: bool = False) -> Optional[float]:
    """Return a positive, reasonably narrow bid/ask midpoint."""
    try:
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        if not isfinite(bid) or not isfinite(ask) or bid <= 0 or ask < bid:
            return None
        mid = (bid + ask) / 2.0
        if mid <= 0 or (ask - bid) / mid > MAX_SPREAD_RATIO:
            return None
        if require_liquidity:
            volume = float(row.get("volume", 0) or 0)
            open_interest = float(row.get("openInterest", 0) or 0)
            if not isfinite(volume):
                volume = 0.0
            if not isfinite(open_interest):
                open_interest = 0.0
            if volume <= 0 and open_interest < MIN_NEIGHBOR_OPEN_INTEREST:
                return None
        return mid
    except (TypeError, ValueError):
        return None


def implied_volatility_from_price(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    option_models: Optional[OptionModels] = None,
) -> Optional[float]:
    """Invert Black-Scholes for a valid European option midpoint."""
    try:
        market_price, S, K, T, r, q = map(
            float, (market_price, S, K, T, r, q)
        )
    except (TypeError, ValueError):
        return None
    if not all(isfinite(value) for value in (market_price, S, K, T, r, q)):
        return None
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    if option_type not in ("call", "put"):
        return None

    discounted_spot = S * exp(-q * T)
    discounted_strike = K * exp(-r * T)
    intrinsic = max(
        0.0,
        discounted_spot - discounted_strike
        if option_type == "call"
        else discounted_strike - discounted_spot,
    )
    upper_bound = discounted_spot if option_type == "call" else discounted_strike
    tolerance = max(0.01, upper_bound * 1e-4)
    if market_price < intrinsic - tolerance or market_price >= upper_bound:
        return None

    models = option_models or OptionModels()

    def objective(sigma: float) -> float:
        return models.black_scholes_price(S, K, T, r, sigma, q, option_type) - market_price

    try:
        return valid_iv(brentq(objective, MIN_IV, MAX_IV, xtol=1e-7, maxiter=100))
    except (ValueError, RuntimeError, OverflowError):
        return None


class ImpliedVolatilityResolver:
    """Resolve volatility as exact market IV, local interpolation, history, then 20%."""

    def __init__(
        self,
        data_fetcher: Optional[DataFetcher] = None,
        option_models: Optional[OptionModels] = None,
    ) -> None:
        self.data_fetcher = data_fetcher or DataFetcher()
        self.option_models = option_models or OptionModels()

    def resolve(
        self,
        ticker: str,
        S: float,
        K: float,
        expiry: str,
        r: float,
        q: float,
        option_type: str,
        hist_vol: Optional[float] = None,
    ) -> VolatilityEstimate:
        ticker = ticker.strip().upper()
        try:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        except ValueError:
            expiry_date = datetime.fromisoformat(expiry).date()
            expiry = expiry_date.isoformat()

        T = max((expiry_date - date.today()).days, 1) / 365.0
        cache_key = (
            f"iv_{ticker}_{expiry}_{option_type}_{K:.8f}_{S:.6f}_"
            f"{r:.8f}_{q:.8f}_{hist_vol}"
        )
        cached = global_cache_iv.get(cache_key)
        if isinstance(cached, VolatilityEstimate):
            return cached

        try:
            expiries = sorted(self.data_fetcher.get_available_expiries(ticker) or [])
        except Exception as exc:
            logger.warning("Échéances indisponibles pour la résolution IV de %s : %s", ticker, exc)
            expiries = []

        chains: dict[str, Any] = {}
        if expiry in expiries:
            chain = self._get_chain(ticker, expiry)
            if chain is not None:
                chains[expiry] = chain
                direct = self._exact_contract(
                    chain, expiry, S, K, T, r, q, option_type
                )
                if direct is not None:
                    global_cache_iv.set(cache_key, direct)
                    return direct
                same_expiry = self._interpolate_expiry(
                    chain, expiry, S, K, r, q
                )
                if same_expiry is not None:
                    result = VolatilityEstimate(
                        same_expiry, "IV interpolée (même échéance)",
                        note="Estimation entre cotations liquides de la même échéance",
                    )
                    global_cache_iv.set(cache_key, result)
                    return result

        surrounding = self._surrounding_expiries(expiries, expiry)
        if len(surrounding) == 2:
            for chain_expiry in surrounding:
                if chain_expiry not in chains:
                    chain = self._get_chain(ticker, chain_expiry)
                    if chain is not None:
                        chains[chain_expiry] = chain

            term_points: list[tuple[float, float]] = []
            for chain_expiry, chain in chains.items():
                chain_date = datetime.strptime(chain_expiry, "%Y-%m-%d").date()
                chain_T = max((chain_date - date.today()).days, 1) / 365.0
                sigma_at_moneyness = self._interpolate_expiry(
                    chain, chain_expiry, S, K, r, q
                )
                if sigma_at_moneyness is not None:
                    term_points.append((chain_T, sigma_at_moneyness))
            term_points.sort()
            before = [point for point in term_points if point[0] < T]
            after = [point for point in term_points if point[0] > T]
            if before and after:
                t0, sigma0 = before[-1]
                t1, sigma1 = after[0]
                w0, w1 = sigma0 * sigma0 * t0, sigma1 * sigma1 * t1
                variance = w0 + (w1 - w0) * (T - t0) / (t1 - t0)
                if variance > 0 and isfinite(variance):
                    result = VolatilityEstimate(
                        sqrt(variance / T),
                        "IV interpolée (maturité et moneyness)",
                        note="Variance totale interpolée entre échéances cotées",
                    )
                    global_cache_iv.set(cache_key, result)
                    return result

        historical = valid_iv(hist_vol)
        if historical is None:
            try:
                historical = valid_iv(
                    self.data_fetcher.get_historical_volatility(ticker, "1y")
                )
            except Exception as exc:
                logger.warning("Volatilité historique indisponible pour %s : %s", ticker, exc)
        if historical is not None:
            result = VolatilityEstimate(historical, "Volatilité historique")
        else:
            result = VolatilityEstimate(0.20, "Défaut (20 %)")

        global_cache_iv.set(cache_key, result)
        return result

    def resolve_in_chain(
        self,
        chain: Any,
        expiry: str,
        S: float,
        K: float,
        r: float,
        q: float,
        option_type: str,
        hist_vol: Optional[float] = None,
        curve: Optional[list[tuple[float, float]]] = None,
    ) -> VolatilityEstimate:
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        T = max((expiry_date - date.today()).days, 1) / 365.0
        direct = self._exact_contract(chain, expiry, S, K, T, r, q, option_type)
        if direct is not None:
            return direct
        sigma = (
            self._interpolate_curve(curve, expiry, S, K, r, q)
            if curve is not None
            else self._interpolate_expiry(chain, expiry, S, K, r, q)
        )
        if sigma is not None:
            return VolatilityEstimate(
                sigma, "IV interpolée (même échéance)",
                note="Estimation entre cotations liquides de la même échéance",
            )
        historical = valid_iv(hist_vol)
        if historical is not None:
            return VolatilityEstimate(historical, "Volatilité historique")
        return VolatilityEstimate(0.20, "Défaut (20 %)")

    def _get_chain(self, ticker: str, expiry: str) -> Any:
        try:
            return self.data_fetcher.get_option_chain_for_expiry(ticker, expiry)
        except Exception as exc:
            logger.warning("Chaîne indisponible pour %s %s : %s", ticker, expiry, exc)
            return None

    @staticmethod
    def _surrounding_expiries(expiries: list[str], target: str) -> list[str]:
        before = [expiry for expiry in expiries if expiry < target]
        after = [expiry for expiry in expiries if expiry > target]
        if before and after:
            return [before[-1], after[0]][:MAX_NEIGHBOR_EXPIRIES]
        return []

    def _exact_contract(
        self,
        chain: Any,
        expiry: str,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        option_type: str,
    ) -> Optional[VolatilityEstimate]:
        option_data = getattr(chain, "calls" if option_type == "call" else "puts", None)
        if option_data is None or option_data.empty or "strike" not in option_data:
            return None
        matches = option_data[np.isclose(option_data["strike"].astype(float), K, rtol=0, atol=1e-8)]
        if matches.empty:
            return None
        row = matches.iloc[0]
        mid = valid_mid_quote(row)
        if mid is not None:
            sigma = implied_volatility_from_price(
                mid, S, K, T, r, q, option_type, self.option_models
            )
            if sigma is not None:
                return VolatilityEstimate(
                    sigma, "IV recalculée (bid/ask)", mid, expiry
                )

        vendor_iv = valid_iv(row.get("impliedVolatility"))
        try:
            has_partial_quote = (
                float(row.get("bid", 0) or 0) > 0
                or float(row.get("ask", 0) or 0) > 0
            )
        except (TypeError, ValueError):
            has_partial_quote = True
        if vendor_iv is not None and mid is None and not has_partial_quote:
            return VolatilityEstimate(
                vendor_iv, "IV Yahoo (contrat exact)",
                expiry_used=expiry,
                note="IV fournie par Yahoo, cotation bid/ask inexploitable",
            )
        return None

    def _interpolate_expiry(
        self,
        chain: Any,
        expiry: str,
        S: float,
        K: float,
        r: float,
        q: float,
    ) -> Optional[float]:
        return self._interpolate_curve(
            self.build_expiry_curve(chain, expiry, S, r, q), expiry, S, K, r, q
        )

    def build_expiry_curve(
        self,
        chain: Any,
        expiry: str,
        S: float,
        r: float,
        q: float,
    ) -> list[tuple[float, float]]:
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        T = max((expiry_date - date.today()).days, 1) / 365.0
        forward = S * exp((r - q) * T)
        points: list[tuple[float, float]] = []

        for option_type in ("put", "call"):
            option_data = getattr(chain, "calls" if option_type == "call" else "puts", None)
            if option_data is None or option_data.empty:
                continue
            for _, row in option_data.iterrows():
                try:
                    strike = float(row["strike"])
                    if not isfinite(strike) or strike <= 0:
                        continue
                    if option_type == "put" and strike >= forward:
                        continue
                    if option_type == "call" and strike < forward:
                        continue
                    mid = valid_mid_quote(row, require_liquidity=True)
                    if mid is None:
                        continue
                    sigma = implied_volatility_from_price(
                        mid, S, strike, T, r, q, option_type, self.option_models
                    )
                    if sigma is not None:
                        points.append((log(strike / forward), sigma))
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue

        return sorted(points)

    @staticmethod
    def _interpolate_curve(
        points: Optional[list[tuple[float, float]]],
        expiry: str,
        S: float,
        K: float,
        r: float,
        q: float,
    ) -> Optional[float]:
        if points is None or len(points) < 2:
            return None
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        T = max((expiry_date - date.today()).days, 1) / 365.0
        forward = S * exp((r - q) * T)
        target_k = log(K / forward)
        x_values = np.array([point[0] for point in points])
        y_values = np.array([point[1] for point in points])
        if target_k < x_values[0] or target_k > x_values[-1]:
            return None
        return float(np.interp(target_k, x_values, y_values))

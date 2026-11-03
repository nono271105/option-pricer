"""
backend/services.py : Logique métier de l'API (anciennement api.py, sans Eel ni état global).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from data_fetcher import DataFetcher
from logic.bsm_logic import OptionModels
from logic.crr_logic import CRRModels
from logic.simulation_logic import SimulationLogic
from logic.exotic_options_logic import (
    price_barrier_analytical,
    price_barrier_mc,
    price_asian_mc,
    price_lookback_mc,
    price_digital_analytical,
    price_digital_mc,
)
from logic.volatility_smile_logic import VolatilitySmileLogic
from logic.strategy_logic import StrategyManager
from logic.volatility_surface_logic import ImpliedVolatilitySurface
from validation import (
    validate_pricing_params,
    validate_N_crr,
    validate_option_type,
    validate_position,
    validate_n_sims,
    validate_n_steps,
    validate_barrier_type,
    validate_averaging,
    validate_exotic_type,
    validate_r,
    validate_q,
    validate_rebate,
)

logger = logging.getLogger(__name__)


class DataSourceError(Exception):
    """Exception levée en cas de défaillance d'une source de données externe (HTTP 502/503)."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ── Singletons partagés ───────────────────────────────────────────────────────
_data_fetcher = DataFetcher()
_option_models = OptionModels()
_crr_models = CRRModels()
_simulation = SimulationLogic(_option_models)
_smile_logic = VolatilitySmileLogic()
_strategy_mgr = StrategyManager()
_surface_logic = ImpliedVolatilitySurface()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────

def _safe(val: Any) -> Any:
    """Convertit les valeurs numpy non-sérialisables en types Python natifs."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, np.ndarray):
        return [_safe(v) for v in val.tolist()]
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


def _greek_curve(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    option_type: str,
    greek: str,
    position: str = "long",
) -> List[Dict[str, Any]]:
    """Génère 100 points de la courbe d'un grec en fonction du spot."""
    sign = 1 if position == "long" else -1
    s_min = S * 0.6
    s_max = S * 1.4
    spots = np.linspace(s_min, s_max, 100)
    result = []
    for s in spots:
        try:
            g = _option_models.calculate_greeks(s, K, T, r, sigma, q, option_type)
            result.append({"spot": round(float(s), 2), "value": round(sign * g[greek], 6)})
        except Exception:
            result.append({"spot": round(float(s), 2), "value": None})
    return result


def _payoff_curve(
    K: float,
    premium: float,
    option_type: str,
    position: str,
    S: float,
) -> List[Dict[str, Any]]:
    """Génère le profil de payoff à maturité autour du spot courant."""
    s_arr = np.linspace(S * 0.6, S * 1.4, 200)
    if option_type == "call":
        gross = np.maximum(s_arr - K, 0)
    else:
        gross = np.maximum(K - s_arr, 0)
    net = gross - premium if position == "long" else premium - gross
    return [
        {"spot": round(float(s), 2), "payoff": round(float(p), 4)}
        for s, p in zip(s_arr, net)
    ]


def _next_weekday_after(days_ahead: int) -> str:
    """Retourne la date d'aujourd'hui + days_ahead jours, ajustée au prochain jour ouvré."""
    d = date.today() + timedelta(days=days_ahead)
    if d.weekday() == 5:
        d += timedelta(days=2)
    elif d.weekday() == 6:
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _parse_date(date_str: str, default_days: int = 90) -> datetime:
    """Parse la date (YYYY-MM-DD ou DD/MM/YYYY). En cas d'échec, retourne j+default_days."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            return datetime.strptime(_next_weekday_after(default_days), "%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DONNÉES DE MARCHÉ
# ─────────────────────────────────────────────────────────────────────────────

def fetch_market_data(ticker: str) -> Dict[str, Any]:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("VALIDATION_INVALID_VALUE: Ticker vide", "VALIDATION_INVALID_VALUE")

    try:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                "price": ex.submit(_data_fetcher.get_live_price, ticker),
                "sofr": ex.submit(_data_fetcher.get_sofr_rate),
                "div": ex.submit(_data_fetcher.get_dividend_yield, ticker),
                "vol": ex.submit(_data_fetcher.get_historical_volatility, ticker, "1y"),
                "company": ex.submit(_data_fetcher.get_company_name, ticker),
            }
            results: Dict[str, Any] = {}
            for key, fut in futures.items():
                try:
                    results[key] = fut.result(timeout=15)
                except Exception:
                    results[key] = None

        S = _safe(results["price"])
        r = _safe(results["sofr"]) if results["sofr"] is not None else 0.05
        q = _safe(results["div"]) if results["div"] is not None else 0.0
        hist_vol = _safe(results["vol"]) if results["vol"] is not None else 0.20
        company = results["company"] or ticker

        return {
            "ticker": ticker,
            "company_name": company,
            "S": S,
            "r": r,
            "q": q,
            "hist_vol": hist_vol,
            "error": None,
        }
    except Exception as e:
        logger.exception("fetch_market_data error for %s", ticker)
        raise DataSourceError(f"Erreur lors de la récupération des données de marché pour {ticker}: {e}")


def get_available_expiries(ticker: str) -> Dict[str, Any]:
    ticker = ticker.strip().upper()
    try:
        expiries = _data_fetcher.get_available_expiries(ticker)
        return {"expiries": expiries, "error": None}
    except Exception as e:
        logger.exception("get_available_expiries error for %s", ticker)
        raise DataSourceError(f"Erreur lors de la récupération des échéances pour {ticker}: {e}")


def get_option_chain(
    ticker: str,
    expiry_str: str,
    S: Optional[float] = None,
    r: Optional[float] = None,
    q: Optional[float] = None,
) -> Dict[str, Any]:
    ticker = ticker.strip().upper()
    maturity_dt = _parse_date(expiry_str, 60)

    try:
        opt_chain, closest_date = _data_fetcher.get_option_data_chain(ticker, maturity_dt)
        if opt_chain is None or closest_date is None:
            raise DataSourceError(f"Aucune chaîne d'options pour {ticker}")

        S_val = S if S is not None else (_data_fetcher.get_live_price(ticker) or 100.0)
        r_val = r if r is not None else 0.05
        q_val = q if q is not None else 0.0
        T = max((datetime.strptime(closest_date, "%Y-%m-%d").date() - date.today()).days / 365.0, 1 / 365)

        def _process_side(df, otype: str) -> List[Dict[str, Any]]:
            rows = []
            for _, row in df.iterrows():
                K = float(row.get("strike", 0))
                bid = float(row.get("bid", 0) or 0)
                ask = float(row.get("ask", 0) or 0)
                iv_raw = float(row.get("impliedVolatility", 0) or 0)
                v = row.get("volume", 0)
                vol_raw = 0 if v is None or np.isnan(v) else int(v)
                oi = row.get("openInterest", 0)
                oi_raw = 0 if oi is None or np.isnan(oi) else int(oi)

                sigma = iv_raw if iv_raw > 0.01 else 0.25
                try:
                    g = _option_models.calculate_greeks(S_val, K, T, r_val, sigma, q_val, otype)
                    delta_val = round(g["delta"], 4)
                except Exception:
                    delta_val = None

                rows.append({
                    "strike": K,
                    "bid": bid,
                    "ask": ask,
                    "iv": round(iv_raw * 100, 2),
                    "volume": vol_raw,
                    "oi": oi_raw,
                    "delta": delta_val,
                })
            return rows

        return {
            "expiry_used": closest_date,
            "calls": _process_side(opt_chain.calls, "call"),
            "puts": _process_side(opt_chain.puts, "put"),
            "error": None,
        }
    except DataSourceError:
        raise
    except Exception as e:
        logger.exception("get_option_chain error")
        raise DataSourceError(f"Erreur chaîne d'options pour {ticker}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. BSM
# ─────────────────────────────────────────────────────────────────────────────

def calculate_bsm(
    ticker: str,
    S: float,
    K: float,
    maturity_date: str,
    r: float,
    q: float,
    option_type: str,
    position: str = "long",
    hist_vol: Optional[float] = None,
) -> Dict[str, Any]:
    S_f, K_f, r_f, q_f = float(S), float(K), float(r), float(q)
    mat_dt = _parse_date(maturity_date, 90)
    T_days = max((mat_dt.date() - date.today()).days, 1)
    T = T_days / 365.0
    validate_pricing_params(S_f, K_f, T, r_f, q_f, option_type)
    validate_position(position)

    iv, _, _ = _data_fetcher.get_implied_volatility_and_price(ticker, K_f, mat_dt, option_type)
    sigma = iv if iv is not None else (hist_vol if hist_vol is not None else 0.20)
    sigma_source = "IV" if iv is not None else "Historique"

    price = _option_models.black_scholes_price(S_f, K_f, T, r_f, sigma, q_f, option_type)
    price = round(float(price), 4)

    raw_greeks = _option_models.calculate_greeks(S_f, K_f, T, r_f, sigma, q_f, option_type)
    sign = 1 if position == "long" else -1
    greeks = {k: round(sign * float(v), 6) for k, v in raw_greeks.items()}

    payoff_data = _payoff_curve(K_f, price, option_type, position, S_f)

    greek_curves = {}
    for greek_name in ("delta", "gamma", "theta", "vega", "rho"):
        greek_curves[f"{greek_name}_data"] = _greek_curve(
            S_f, K_f, T, r_f, sigma, q_f, option_type, greek_name, position
        )

    if option_type == "call":
        breakeven = round(K_f + price, 2) if position == "long" else round(K_f - price, 2)
    else:
        breakeven = round(K_f - price, 2) if position == "long" else round(K_f + price, 2)

    return {
        "price": price,
        "sigma": sigma,
        "sigma_source": sigma_source,
        "payoff_data": payoff_data,
        "greeks": greeks,
        **greek_curves,
        "breakeven": breakeven,
        "S": S_f,
        "K": K_f,
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CRR
# ─────────────────────────────────────────────────────────────────────────────

def calculate_crr(
    ticker: str,
    S: float,
    K: float,
    maturity_date: str,
    r: float,
    q: float,
    N: int,
    option_type: str,
    position: str = "long",
    hist_vol: Optional[float] = None,
) -> Dict[str, Any]:
    S_f, K_f, r_f, q_f = float(S), float(K), float(r), float(q)
    N_i = int(N)
    mat_dt = _parse_date(maturity_date, 90)
    T_days = max((mat_dt.date() - date.today()).days, 1)
    T = T_days / 365.0
    validate_pricing_params(S_f, K_f, T, r_f, q_f, option_type)
    validate_position(position)
    validate_N_crr(N_i)

    iv, _, _ = _data_fetcher.get_implied_volatility_and_price(ticker, K_f, mat_dt, option_type)
    sigma_used = iv if iv is not None else (hist_vol if hist_vol is not None else 0.20)
    sigma_source = "IV" if iv is not None else "Historique"

    price = _crr_models.cox_ross_rubinstein_price(
        S=S_f, K=K_f, T=T, r=r_f, sigma=sigma_used, q=q_f, N=N_i, option_type=option_type
    )
    price = round(float(price), 4)

    raw_greeks = _crr_models.calculate_greeks_crr(
        S=S_f, K=K_f, T=T, r=r_f, sigma=sigma_used, q=q_f, N=N_i, option_type=option_type
    )
    sign = 1 if position == "long" else -1
    greeks = {k: round(sign * float(v), 6) for k, v in raw_greeks.items()}

    payoff_data = _payoff_curve(K_f, price, option_type, position, S_f)

    n_curve = min(N_i, 100)
    greek_names = ("delta", "gamma", "theta", "vega", "rho")
    greek_curves = {f"{gn}_data": [] for gn in greek_names}
    s_min, s_max = S_f * 0.6, S_f * 1.4
    spots = np.linspace(s_min, s_max, 60)
    for s in spots:
        spot_val = round(float(s), 2)
        try:
            g = _crr_models.calculate_greeks_crr(
                S=float(s), K=K_f, T=T, r=r_f, sigma=sigma_used, q=q_f, N=n_curve, option_type=option_type
            )
            for gn in greek_names:
                val = round(sign * g[gn], 6) if g and gn in g and g[gn] is not None else None
                greek_curves[f"{gn}_data"].append({"spot": spot_val, "value": val})
        except Exception:
            for gn in greek_names:
                greek_curves[f"{gn}_data"].append({"spot": spot_val, "value": None})

    if option_type == "call":
        breakeven = round(K_f + price, 2) if position == "long" else round(K_f - price, 2)
    else:
        breakeven = round(K_f - price, 2) if position == "long" else round(K_f + price, 2)

    return {
        "price": price,
        "sigma": sigma_used,
        "sigma_source": sigma_source,
        "greeks": greeks,
        "payoff_data": payoff_data,
        **greek_curves,
        "breakeven": breakeven,
        "S": S_f,
        "K": K_f,
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    K: float,
    T_days: float,
    r: float,
    q: float,
    vol_min: int,
    vol_max: int,
    vol_step: int,
    underlying_min: int,
    underlying_max: int,
    underlying_step: int,
    option_type: str = "call",
) -> Dict[str, Any]:
    K_f, r_f, q_f = float(K), float(r), float(q)
    T_f = max(float(T_days) / 365.0, 1e-6)
    validate_r(r_f)
    validate_q(q_f)
    validate_option_type(option_type)

    vols, prices, matrix, _ = _simulation.run_simulation(
        K=K_f,
        T=T_f,
        r=r_f,
        q=q_f,
        vol_min=int(vol_min),
        vol_max=int(vol_max),
        vol_step=int(vol_step),
        underlying_min=int(underlying_min),
        underlying_max=int(underlying_max),
        underlying_step=int(underlying_step),
        option_type=option_type,
    )

    return {
        "vols": [int(v) for v in vols.tolist()],
        "prices": [int(p) for p in prices.tolist()],
        "matrix": [[round(float(v), 4) for v in row] for row in matrix.tolist()],
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. SMILE DE VOLATILITÉ
# ─────────────────────────────────────────────────────────────────────────────

def calculate_smile(
    ticker: str,
    expiry_str: str,
    S: Optional[float] = None,
    r: Optional[float] = None,
    q: Optional[float] = None,
) -> Dict[str, Any]:
    ticker = ticker.strip().upper()
    maturity_dt = _parse_date(expiry_str, 60)

    try:
        opt_chain, closest_date = _data_fetcher.get_option_data_chain(ticker, maturity_dt)
        if opt_chain is None:
            raise DataSourceError(f"Aucune chaîne d'options disponible pour {ticker}")

        T = max((datetime.strptime(closest_date, "%Y-%m-%d").date() - date.today()).days / 365.0, 1 / 365)
        S_val = S if S is not None else (_data_fetcher.get_live_price(ticker) or 100.0)
        r_val = r if r is not None else 0.05
        q_val = q if q is not None else 0.0

        strikes_interp, ivs_interp, smile_df = _smile_logic.process_smile_data(
            opt_chain, float(S_val), T, r_val, q_val
        )

        if strikes_interp is None or smile_df is None:
            raise DataSourceError("Données insuffisantes pour calculer le smile de volatilité")

        raw = smile_df.to_dict("records") if smile_df is not None else []
        for row in raw:
            row["iv"] = round(float(row["iv"]) * 100, 2)

        return {
            "expiry_used": closest_date,
            "strikes_interp": [round(float(s), 2) for s in strikes_interp.tolist()],
            "ivs_interp": [round(float(v), 2) for v in ivs_interp.tolist()],
            "raw_data": raw,
            "current_price": float(S_val),
            "error": None,
        }
    except DataSourceError:
        raise
    except Exception as e:
        logger.exception("calculate_smile error")
        raise DataSourceError(f"Erreur lors du calcul du smile pour {ticker}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. SURFACE DE VOLATILITÉ
# ─────────────────────────────────────────────────────────────────────────────

def calculate_surface(
    ticker: str,
    S: Optional[float] = None,
    r: Optional[float] = None,
    q: Optional[float] = None,
) -> Dict[str, Any]:
    ticker = ticker.strip().upper()
    try:
        S_val = S if S is not None else (_data_fetcher.get_live_price(ticker) or 100.0)
        r_val = r if r is not None else 0.05
        q_val = q if q is not None else 0.0

        raw_df, grids = _surface_logic.get_surface_for_ticker(
            ticker, float(S_val), r_val, q_val
        )
        if grids is None or raw_df is None:
            raise DataSourceError("Données insuffisantes pour calculer la surface IV")

        X_grid, Y_grid, Z_grid = grids

        return {
            "strikes": [round(float(v), 2) for v in X_grid[0].tolist()],
            "maturities": [round(float(v), 1) for v in Y_grid[:, 0].tolist()],
            "iv_surface": [
                [round(float(v) * 100, 2) if not np.isnan(v) else None for v in row]
                for row in Z_grid.tolist()
            ],
            "error": None,
        }
    except DataSourceError:
        raise
    except Exception as e:
        logger.exception("calculate_surface error")
        raise DataSourceError(f"Erreur surface IV pour {ticker}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. OPTIONS EXOTIQUES
# ─────────────────────────────────────────────────────────────────────────────

def price_exotic(
    exotic_type: str,
    ticker: str,
    S: float,
    K: float,
    maturity_date: str,
    r: float,
    q: float,
    option_type: str,
    barrier: Optional[float] = None,
    barrier_type: str = "down-and-out",
    averaging: str = "arithmetic",
    payoff_amount: float = 1.0,
    n_sims: int = 50_000,
    n_steps: int = 252,
    seed: int = 42,
    rebate: float = 0.0,
    hist_vol: Optional[float] = None,
) -> Dict[str, Any]:
    res_mc = None
    res_ana = None

    S_f, K_f, r_f, q_f = float(S), float(K), float(r), float(q)
    rebate_f = float(rebate)
    validate_rebate(rebate_f)
    mat_dt = _parse_date(maturity_date, 90)
    T_days_i = max((mat_dt.date() - date.today()).days, 1)
    T = T_days_i / 365.0
    validate_pricing_params(S_f, K_f, T, r_f, q_f, option_type)
    validate_exotic_type(exotic_type)
    validate_n_sims(n_sims)
    validate_n_steps(n_steps)
    if exotic_type == "barrier":
        if barrier is None:
            raise ValueError("VALIDATION_INVALID_VALUE: barrier requis pour barrier option", "VALIDATION_INVALID_VALUE")
        validate_barrier_type(barrier_type)
    if exotic_type == "asian":
        validate_averaging(averaging)

    iv, _, _ = _data_fetcher.get_implied_volatility_and_price(ticker, K_f, mat_dt, option_type)
    sigma = iv if iv is not None else (hist_vol if hist_vol is not None else 0.20)
    sigma_source = "IV" if iv is not None else "Historique"

    results: Dict[str, Any] = {}

    if exotic_type == "barrier":
        try:
            res_ana = price_barrier_analytical(
                S_f, K_f, T, r_f, sigma, q_f, float(barrier),
                option_type, barrier_type, rebate=rebate_f,
            )
            results["analytical"] = {
                "price": float(res_ana.price),
                "method": res_ana.method,
            }
        except Exception as e:
            logger.warning("Erreur pricing barrière analytique : %s", e, exc_info=True)
        try:
            res_mc = price_barrier_mc(
                S_f, K_f, T, r_f, sigma, q_f, float(barrier),
                option_type, barrier_type, n_sims, n_steps, seed, rebate=rebate_f,
            )
        except Exception as e:
            logger.warning("Erreur pricing barrière MC : %s", e, exc_info=True)
            res_mc = None

    elif exotic_type == "asian":
        res_mc = price_asian_mc(
            S_f, K_f, T, r_f, sigma, q_f, option_type, averaging,
            n_sims, n_steps, seed
        )

    elif exotic_type == "lookback":
        res_mc = price_lookback_mc(S_f, T, r_f, sigma, q_f, option_type, n_sims, n_steps, seed)

    elif exotic_type == "digital":
        try:
            res_ana = price_digital_analytical(S_f, K_f, T, r_f, sigma, q_f, option_type, payoff_amount)
            results["analytical"] = {
                "price": float(res_ana.price),
                "method": res_ana.method,
            }
        except Exception as e:
            logger.warning("Erreur pricing digital analytique : %s", e, exc_info=True)
        try:
            res_mc = price_digital_mc(
                S_f, K_f, T, r_f, sigma, q_f, option_type, payoff_amount,
                n_sims, n_steps, seed
            )
        except Exception as e:
            logger.warning("Erreur pricing digital MC : %s", e, exc_info=True)
            res_mc = None
    else:
        raise ValueError(f"VALIDATION_INVALID_EXOTIC_TYPE: Type exotique inconnu: {exotic_type}", "VALIDATION_INVALID_EXOTIC_TYPE")

    paths_data = None
    dist_data = None
    if res_mc is not None:
        results["mc"] = {
            "price": float(res_mc.price),
            "method": res_mc.method,
            "std_error": float(res_mc.std_error) if res_mc.std_error is not None else None,
            "ci_95": list(res_mc.ci_95) if res_mc.ci_95 is not None else None,
        }
        if res_mc.price_paths is not None:
            paths = res_mc.price_paths
            n_sample = min(50, paths.shape[0])
            rng = np.random.default_rng(seed)
            idx = rng.choice(paths.shape[0], size=n_sample, replace=False)
            paths_data = []
            for path in paths[idx]:
                paths_data.append([round(float(v), 4) for v in path])
        if res_mc.payoffs is not None:
            payoffs = res_mc.payoffs
            counts, bin_edges = np.histogram(payoffs, bins=30)
            dist_data = [
                {"bucket": round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 2), "count": int(counts[i])}
                for i in range(len(counts))
            ]

    if not results:
        raise ValueError("VALIDATION_INVALID_VALUE: Le calcul a échoué pour l'option demandée.", "VALIDATION_INVALID_VALUE")

    main_price = results.get("analytical", results.get("mc"))["price"]

    return {
        "price": main_price,
        "sigma": sigma,
        "sigma_source": sigma_source,
        "results": results,
        "price_paths": paths_data,
        "payoff_distribution": dist_data,
        "S": S_f,
        "K": K_f,
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. STRATÉGIES
# ─────────────────────────────────────────────────────────────────────────────

def get_strategy_names() -> List[str]:
    return list(StrategyManager.STRATEGY_DEFINITIONS.keys())


def calculate_strategy(
    strategy_name: str,
    ticker: str,
    S: float,
    T_days: float,
    r: float,
    q: float,
    expiry_str: str,
    hist_vol: Optional[float] = None,
) -> Dict[str, Any]:
    S_f, r_f, q_f = float(S), float(r), float(q)
    T_f = max(float(T_days) / 365.0, 1e-6)
    validate_r(r_f)
    validate_q(q_f)

    maturity_dt = _parse_date(expiry_str, int(T_days))

    iv, _, _ = _data_fetcher.get_implied_volatility_and_price(ticker, S_f, maturity_dt, "call")
    sigma_used = iv if iv is not None else (hist_vol if hist_vol is not None else 0.20)
    sigma_source = "IV" if iv is not None else "Historique"

    legs, T_eff = _strategy_mgr.build_legs(
        strategy_name, S_f, T_f, r_f, sigma_used, q_f, maturity_dt, ticker,
        _data_fetcher, _option_models
    )

    S_range = np.linspace(S_f * 0.6, S_f * 1.4, 300)
    payoff = _strategy_mgr.compute_payoff(legs, S_range)
    value_today = _strategy_mgr.compute_value_today(
        legs, S_range, S_f, T_eff, r_f, sigma_used, q_f, _option_models
    )
    metrics = _strategy_mgr.compute_metrics(legs, S_range, payoff)
    greeks = _strategy_mgr.compute_greeks(legs, S_f, T_eff, r_f, sigma_used, q_f, _option_models)

    def _fmt(val):
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return None
        return val

    ser_metrics = {
        "cost": _fmt(metrics["cost"]),
        "breakevens": metrics["breakevens"],
        "max_gain": _fmt(metrics["max_gain"]),
        "max_loss": _fmt(metrics["max_loss"]),
    }

    return {
        "strategy_name": strategy_name,
        "legs": legs,
        "payoff_data": [
            {"spot": round(float(s), 2), "payoff": round(float(p), 4)}
            for s, p in zip(S_range, payoff)
        ],
        "value_today_data": [
            {"spot": round(float(s), 2), "value": round(float(v), 4)}
            for s, v in zip(S_range, value_today)
        ],
        "metrics": ser_metrics,
        "sigma": sigma_used,
        "sigma_source": sigma_source,
        "greeks": greeks,
        "error": None,
    }


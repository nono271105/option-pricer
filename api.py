"""
api.py — Pont Eel : expose toutes les fonctions Python au frontend React.

Chaque fonction décorée @eel.expose est appelable depuis TypeScript via :
    window.eel.nom_fonction(args)()
"""

from __future__ import annotations

import logging
import eel
import numpy as np
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

from data_fetcher import DataFetcher
from market_data_store import MarketDataStore
from logic.bsm_logic import OptionModels
from logic.crr_logic import CRRModels
from logic.simulation_logic import SimulationLogic
from logic.exotic_options_logic import (
    price_barrier_analytical, price_barrier_mc,
    price_asian_mc, price_lookback_mc,
    price_digital_analytical, price_digital_mc,
)
from logic.volatility_smile_logic import VolatilitySmileLogic
from logic.strategy_logic import StrategyManager
from logic.volatility_surface_logic import ImpliedVolatilitySurface

logger = logging.getLogger(__name__)

# ── Singletons partagés (initialisés une seule fois) ──────────────────────────
_data_fetcher = DataFetcher()
_store = MarketDataStore()
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


def _greek_curve(S: float, K: float, T: float, r: float, sigma: float,
                 q: float, option_type: str, greek: str) -> List[Dict]:
    """Génère 100 points de la courbe d'un grec en fonction du spot."""
    s_min = S * 0.6
    s_max = S * 1.4
    spots = np.linspace(s_min, s_max, 100)
    result = []
    for s in spots:
        try:
            g = _option_models.calculate_greeks(s, K, T, r, sigma, q, option_type)
            result.append({"spot": round(float(s), 2), "value": round(g[greek], 6)})
        except Exception:
            result.append({"spot": round(float(s), 2), "value": None})
    return result


def _payoff_curve(K: float, premium: float, option_type: str,
                  position: str, S: float) -> List[Dict]:
    """Génère le profil de payoff à maturité autour du spot courant."""
    s_arr = np.linspace(S * 0.6, S * 1.4, 200)
    if option_type == "call":
        gross = np.maximum(s_arr - K, 0)
    else:
        gross = np.maximum(K - s_arr, 0)
    net = gross - premium if position == "long" else premium - gross
    return [{"spot": round(float(s), 2), "payoff": round(float(p), 4)}
            for s, p in zip(s_arr, net)]


def _next_weekday_after(days_ahead: int) -> str:
    """Retourne la date d'aujourd'hui + days_ahead jours, ajustée au prochain jour ouvré."""
    d = date.today() + timedelta(days=days_ahead)
    # Si samedi → lundi, si dimanche → lundi
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

@eel.expose
def fetch_market_data(ticker: str) -> Dict:
    """
    Récupère les données de marché complètes pour un ticker donné.

    Retourne:
        {
          "ticker": str, "company_name": str,
          "S": float|None, "r": float|None,
          "q": float|None, "hist_vol": float|None,
          "error": str|None
        }
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return {"error": "Ticker vide"}

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                "price":   ex.submit(_data_fetcher.get_live_price, ticker),
                "sofr":    ex.submit(_data_fetcher.get_sofr_rate),
                "div":     ex.submit(_data_fetcher.get_dividend_yield, ticker),
                "vol":     ex.submit(_data_fetcher.get_historical_volatility, ticker, "1y"),
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
        q = _safe(results["div"])  if results["div"]  is not None else 0.0
        hist_vol = _safe(results["vol"]) if results["vol"] is not None else 0.20
        company = results["company"] or ticker

        # Mise à jour du store partagé
        _store.update(ticker=ticker, S=S, r=r, q=q,
                      historical_vol=hist_vol, sigma=None,
                      company_name=company, pricing_method="NC")

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
        return {"error": str(e)}


@eel.expose
def get_option_chain(ticker: str, expiry_str: str) -> Dict:
    """
    Récupère la chaîne d'options pour la date d'expiration la plus proche.

    Args:
        ticker: Symbole du titre (ex: 'AAPL')
        expiry_str: Date d'expiration souhaitée 'YYYY-MM-DD'

    Retourne:
        {
          "expiry_used": str,
          "calls": [{"strike": float, "bid": float, "ask": float,
                     "iv": float, "volume": int, "oi": int, "delta": float}],
          "puts":  [...same structure...],
          "error": str|None
        }
    """
    ticker = ticker.strip().upper()
    maturity_dt = _parse_date(expiry_str, 60)

    try:
        opt_chain, closest_date = _data_fetcher.get_option_data_chain(ticker, maturity_dt)
        if opt_chain is None or closest_date is None:
            return {"error": f"Aucune chaîne d'options pour {ticker}"}

        S = _store.S or _data_fetcher.get_live_price(ticker) or 100.0
        r = _store.r or 0.05
        q = _store.q or 0.0
        T = max((datetime.strptime(closest_date, "%Y-%m-%d").date() - date.today()).days / 365.0, 1/365)

        def _process_side(df, otype: str) -> List[Dict]:
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

                # Calcul du delta BSM
                sigma = iv_raw if iv_raw > 0.01 else 0.25
                try:
                    g = _option_models.calculate_greeks(S, K, T, r, sigma, q, otype)
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
            "puts":  _process_side(opt_chain.puts, "put"),
            "error": None,
        }
    except Exception as e:
        logger.exception("get_option_chain error")
        return {"error": str(e)}


@eel.expose
def get_available_expiries(ticker: str) -> Dict:
    """Retourne la liste des dates d'expiration disponibles pour un ticker."""
    ticker = ticker.strip().upper()
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        expiries = list(t.options)
        return {"expiries": expiries, "error": None}
    except Exception as e:
        return {"expiries": [], "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. BSM
# ─────────────────────────────────────────────────────────────────────────────

@eel.expose
def calculate_bsm(
    ticker: str, S: float, K: float, maturity_date: str, r: float, q: float,
    option_type: str, position: str
) -> Dict:
    """
    Calcule le prix BSM, les Grecs et les courbes graphiques.

    Retourne:
        {
          "price": float,
          "greeks": {"delta", "gamma", "theta", "vega", "rho"},
          "payoff_data": [{"spot", "payoff"}, ...],
          "delta_data":  [{"spot", "value"}, ...],
          "gamma_data":  [...], "theta_data": [...],
          "vega_data":   [...], "rho_data":   [...],
          "breakeven": float,
          "error": str|None
        }
    """
    try:
        mat_dt = _parse_date(maturity_date, 90)
        
        T_days = max((mat_dt.date() - date.today()).days, 1)
        T = T_days / 365.0
        S, K, r, q = float(S), float(K), float(r), float(q)
        
        iv, _, _ = _data_fetcher.get_implied_volatility_and_price(ticker, K, mat_dt, option_type)
        sigma = iv if iv is not None else (_store.historical_vol or 0.20)
        sigma_source = "IV" if iv is not None else "Historique"

        price = _option_models.black_scholes_price(S, K, T, r, sigma, q, option_type)
        price = round(float(price), 4)

        greeks = _option_models.calculate_greeks(S, K, T, r, sigma, q, option_type)
        greeks = {k: round(float(v), 6) for k, v in greeks.items()}

        payoff_data = _payoff_curve(K, price, option_type, position, S)

        # Courbes des grecs vs spot
        greek_curves = {}
        for greek_name in ("delta", "gamma", "theta", "vega", "rho"):
            greek_curves[f"{greek_name}_data"] = _greek_curve(
                S, K, T, r, sigma, q, option_type, greek_name
            )

        # Point mort
        if option_type == "call":
            breakeven = round(K + price, 2) if position == "long" else round(K - price, 2)
        else:
            breakeven = round(K - price, 2) if position == "long" else round(K + price, 2)

        # Mise à jour du store
        _store.update(sigma=sigma, pricing_method="BSM")

        return {
            "price": price,
            "sigma": sigma,
            "sigma_source": sigma_source,
            "payoff_data": payoff_data,
            "greeks": greeks,
            **greek_curves,
            "breakeven": breakeven,
            "S": S,
            "K": K,
            "error": None,
        }
    except Exception as e:
        logger.exception("calculate_bsm error")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. CRR
# ─────────────────────────────────────────────────────────────────────────────

@eel.expose
def calculate_crr(
    ticker: str, S: float, K: float, maturity_date: str, r: float, q: float,
    N: int, option_type: str, position: str
) -> Dict:
    """
    Calcule le prix CRR (binomial américain) et les Grecs par différences finies.

    Retourne:
        {
          "price": float,
          "greeks": {"delta", "gamma", "theta", "vega", "rho"},
          "payoff_data": [{"spot", "payoff"}, ...],
          "delta_data": [...], ... (idem BSM)
          "breakeven": float,
          "error": str|None
        }
    """
    try:
        mat_dt = _parse_date(maturity_date, 90)
            
        T_days = max((mat_dt.date() - date.today()).days, 1)
        T = T_days / 365.0
        S, K, r, q = float(S), float(K), float(r), float(q)
        N = int(N)

        iv, _, _ = _data_fetcher.get_implied_volatility_and_price(ticker, K, mat_dt, option_type)
        sigma_used = iv if iv is not None else (_store.historical_vol or 0.20)
        sigma_source = "IV" if iv is not None else "Historique"

        price = _crr_models.cox_ross_rubinstein_price(S, K, T, r, q, sigma_used, N, option_type)
        price = round(float(price), 4)

        greeks = _crr_models.calculate_greeks_crr(S, K, T, r, q, sigma_used, N, option_type)
        greeks = {k: round(float(v), 6) for k, v in greeks.items()}

        payoff_data = _payoff_curve(K, price, option_type, position, S)

        # Courbes grecs (différences finies sur CRR — plus lent, N réduit)
        n_curve = min(N, 100)
        greek_curves = {}
        s_min, s_max = S * 0.6, S * 1.4
        spots = np.linspace(s_min, s_max, 60)
        for greek_name in ("delta", "gamma", "theta", "vega", "rho"):
            curve = []
            for s in spots:
                try:
                    g = _crr_models.calculate_greeks_crr(
                        float(s), K, T, r, q, sigma_used, n_curve, option_type
                    )
                    curve.append({"spot": round(float(s), 2), "value": round(g[greek_name], 6)})
                except Exception:
                    curve.append({"spot": round(float(s), 2), "value": None})
            greek_curves[f"{greek_name}_data"] = curve

        if option_type == "call":
            breakeven = round(K + price, 2) if position == "long" else round(K - price, 2)
        else:
            breakeven = round(K - price, 2) if position == "long" else round(K + price, 2)

        _store.update(sigma=sigma_used, pricing_method="CRR")

        return {
            "price": price,
            "sigma": sigma_used,
            "sigma_source": sigma_source,
            "greeks": greeks,
            "payoff_data": payoff_data,
            **greek_curves,
            "breakeven": breakeven,
            "S": S,
            "K": K,
            "error": None,
        }
    except Exception as e:
        logger.exception("calculate_crr error")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

@eel.expose
def run_simulation(
    K: float, T_days: float, r: float, q: float,
    vol_min: int, vol_max: int, vol_step: int,
    underlying_min: int, underlying_max: int, underlying_step: int,
    option_type: str = "call"
) -> Dict:
    """
    Génère la matrice de prix BSM-Merton (vol × prix sous-jacent).

    Retourne:
        {
          "vols": [int, ...],
          "prices": [int, ...],
          "matrix": [[float, ...], ...],
          "error": str|None
        }
    """
    try:
        T = max(float(T_days) / 365.0, 1e-6)
        vols, prices, matrix, _ = _simulation.run_simulation(
            K=float(K), T=T, r=float(r), q=float(q),
            vol_min=int(vol_min), vol_max=int(vol_max), vol_step=int(vol_step),
            underlying_min=int(underlying_min), underlying_max=int(underlying_max),
            underlying_step=int(underlying_step),
            option_type=option_type,
        )

        return {
            "vols": [int(v) for v in vols.tolist()],
            "prices": [int(p) for p in prices.tolist()],
            "matrix": [[round(float(v), 4) for v in row] for row in matrix.tolist()],
            "error": None,
        }
    except Exception as e:
        logger.exception("run_simulation error")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 5. SMILE DE VOLATILITÉ
# ─────────────────────────────────────────────────────────────────────────────

@eel.expose
def calculate_smile(ticker: str, expiry_str: str) -> Dict:
    """
    Calcule le sourire de volatilité pour un ticker et une maturité donnés.

    Retourne:
        {
          "expiry_used": str,
          "strikes_interp": [float, ...],
          "ivs_interp": [float, ...],
          "raw_data": [{"strike": float, "iv": float, "type": str}, ...],
          "current_price": float,
          "error": str|None
        }
    """
    ticker = ticker.strip().upper()
    try:
        maturity_dt = _parse_date(expiry_str, 60)

        opt_chain, closest_date = _data_fetcher.get_option_data_chain(ticker, maturity_dt)
        if opt_chain is None:
            return {"error": f"Aucune chaîne d'options disponible pour {ticker}"}

        T = max((datetime.strptime(closest_date, "%Y-%m-%d").date() - date.today()).days / 365.0, 1/365)
        S = _store.S or _data_fetcher.get_live_price(ticker) or 100.0
        r = _store.r or 0.05
        q = _store.q or 0.0

        strikes_interp, ivs_interp, smile_df = _smile_logic.process_smile_data(
            opt_chain, float(S), T, r, q
        )

        if strikes_interp is None or smile_df is None:
            return {"error": "Données insuffisantes pour calculer le smile de volatilité"}

        raw = smile_df.to_dict("records") if smile_df is not None else []
        for row in raw:
            row["iv"] = round(float(row["iv"]) * 100, 2)

        return {
            "expiry_used": closest_date,
            "strikes_interp": [round(float(s), 2) for s in strikes_interp.tolist()],
            "ivs_interp": [round(float(v), 2) for v in ivs_interp.tolist()],
            "raw_data": raw,
            "current_price": float(S),
            "error": None,
        }
    except Exception as e:
        logger.exception("calculate_smile error")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 6. SURFACE DE VOLATILITÉ
# ─────────────────────────────────────────────────────────────────────────────

@eel.expose
def calculate_surface(ticker: str) -> Dict:
    """
    Calcule la surface de volatilité implicite 3D.

    Retourne:
        {
          "strikes": [float, ...],
          "maturities": [float, ...],
          "iv_surface": [[float, ...], ...],
          "error": str|None
        }
    """
    ticker = ticker.strip().upper()
    try:
        S = _store.S or _data_fetcher.get_live_price(ticker) or 100.0
        r = _store.r or 0.05
        q = _store.q or 0.0

        raw_df, grids = _surface_logic.get_surface_for_ticker(
            ticker, float(S), r, q
        )
        if grids is None or raw_df is None:
            return {"error": "Données insuffisantes pour calculer la surface IV"}

        X_grid, Y_grid, Z_grid = grids

        # Sérialiser la grille pour Plotly 3D (listes de listes)
        return {
            "strikes": [round(float(v), 2) for v in X_grid[0].tolist()],
            "maturities": [round(float(v), 1) for v in Y_grid[:, 0].tolist()],
            "iv_surface": [
                [round(float(v) * 100, 2) if not np.isnan(v) else None for v in row]
                for row in Z_grid.tolist()
            ],
            "error": None,
        }
    except Exception as e:
        logger.exception("calculate_surface error")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 7. OPTIONS EXOTIQUES
# ─────────────────────────────────────────────────────────────────────────────

@eel.expose
def price_exotic(
    exotic_type: str,
    ticker: str,
    S: float, K: float, maturity_date: str, r: float, q: float,
    option_type: str,
    # Paramètres spécifiques (optionnels selon le type)
    barrier: Optional[float] = None,
    barrier_type: str = "down-and-out",
    averaging: str = "arithmetic",
    payoff_amount: float = 1.0,
    n_sims: int = 50_000,
    n_steps: int = 252,
    seed: int = 42,
) -> Dict:
    """
    Valorise une option exotique.

    exotic_type: 'barrier_analytical' | 'barrier_mc' |
                 'asian_mc' | 'lookback_mc' |
                 'digital_analytical' | 'digital_mc'

    Retourne:
        {
          "price": float,
          "method": str,
          "std_error": float|None,
          "ci_95": [float, float]|None,
          "price_paths": [[float,...], ...]|None (échantillon de trajectoires),
          "payoff_distribution": [{"bucket": str, "count": int}, ...]|None,
          "error": str|None
        }
    """
    try:
        mat_dt = _parse_date(maturity_date, 90)

        T_days = max((mat_dt.date() - date.today()).days, 1)
        T = T_days / 365.0
        S, K, r, q = float(S), float(K), float(r), float(q)

        iv, _, _ = _data_fetcher.get_implied_volatility_and_price(ticker, K, mat_dt, option_type)
        sigma = iv if iv is not None else (_store.historical_vol or 0.20)
        sigma_source = "IV" if iv is not None else "Historique"

        results = {}

        if exotic_type == "barrier":
            if barrier is None:
                return {"error": "barrier requis pour barrier option"}
            # Analytical
            try:
                res_ana = price_barrier_analytical(S, K, T, r, sigma, q, float(barrier),
                                                   option_type, barrier_type)
                results["analytical"] = {
                    "price": float(res_ana.price),
                    "method": res_ana.method
                }
            except Exception as e:
                pass
            # Monte Carlo
            try:
                res_mc = price_barrier_mc(S, K, T, r, sigma, q, float(barrier),
                                          option_type, barrier_type, n_sims, n_steps, seed)
            except Exception as e:
                res_mc = None

        elif exotic_type == "asian":
            res_mc = price_asian_mc(S, K, T, r, sigma, q, option_type, averaging,
                                    n_sims, n_steps, seed)

        elif exotic_type == "lookback":
            res_mc = price_lookback_mc(S, T, r, sigma, q, option_type, n_sims, n_steps, seed)

        elif exotic_type == "digital":
            try:
                res_ana = price_digital_analytical(S, K, T, r, sigma, q, option_type, payoff_amount)
                results["analytical"] = {
                    "price": float(res_ana.price),
                    "method": res_ana.method
                }
            except Exception as e:
                pass
            try:
                res_mc = price_digital_mc(S, K, T, r, sigma, q, option_type, payoff_amount,
                                          n_sims, n_steps, seed)
            except Exception as e:
                res_mc = None
        else:
            return {"error": f"Type exotique inconnu: {exotic_type}"}

        paths_data = None
        dist_data = None
        if "res_mc" in locals() and res_mc is not None:
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
                dist_data = [{"bucket": round(float((bin_edges[i] + bin_edges[i+1]) / 2), 2),
                              "count": int(counts[i])} for i in range(len(counts))]

        if not results:
            return {"error": "Le calcul a echoue pour l'option demandee."}

        # fallback pour price general a fournir
        main_price = results.get("analytical", results.get("mc"))["price"]

        return {
            "price": main_price,
            "sigma": sigma,
            "sigma_source": sigma_source,
            "results": results,
            "price_paths": paths_data,
            "payoff_distribution": dist_data,
            "S": S,
            "K": K,
            "error": None,
        }
    except Exception as e:
        logger.exception("price_exotic error")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 8. STRATÉGIES
# ─────────────────────────────────────────────────────────────────────────────

@eel.expose
def get_strategy_names() -> List[str]:
    """Retourne la liste de toutes les stratégies disponibles."""
    return list(StrategyManager.STRATEGY_DEFINITIONS.keys())


@eel.expose
def calculate_strategy(
    strategy_name: str,
    ticker: str,
    S: float,
    T_days: float,
    r: float,
    q: float,
    expiry_str: str,
) -> Dict:
    """
    Construit les legs d'une stratégie, calcule le payoff, les métriques et les Grecs agrégés.

    Retourne:
        {
          "strategy_name": str,
          "legs": [{"option_type", "position", "strike", "premium"}, ...],
          "payoff_data": [{"spot", "payoff"}, ...],
          "value_today_data": [{"spot", "value"}, ...],
          "metrics": {"cost", "breakevens", "max_gain", "max_loss"},
          "greeks": {"delta", "gamma", "theta", "vega", "rho"},
          "error": str|None
        }
    """
    try:
        T = max(float(T_days) / 365.0, 1e-6)
        S, r, q = float(S), float(r), float(q)

        maturity_dt = _parse_date(expiry_str, int(T_days))

        iv, _, _ = _data_fetcher.get_implied_volatility_and_price(ticker, S, maturity_dt, "call")
        sigma_used = iv if iv is not None else (_store.historical_vol or 0.20)
        sigma_source = "IV" if iv is not None else "Historique"

        legs, T_eff = _strategy_mgr.build_legs(
            strategy_name, S, T, r, sigma_used, q, maturity_dt, ticker,
            _data_fetcher, _option_models
        )

        S_range = np.linspace(S * 0.6, S * 1.4, 300)
        payoff = _strategy_mgr.compute_payoff(legs, S_range)
        value_today = _strategy_mgr.compute_value_today(
            legs, S_range, S, T_eff, r, sigma_used, q, _option_models
        )
        metrics = _strategy_mgr.compute_metrics(legs, S_range, payoff)
        greeks = _strategy_mgr.compute_greeks(legs, S, T_eff, r, sigma_used, q, _option_models)
        
        _store.update(sigma=sigma_used, pricing_method="Strategy")

        # Sérialisation
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
    except Exception as e:
        logger.exception("calculate_strategy error for %s", strategy_name)
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# 9. FORECAST (TimesFM)
# ─────────────────────────────────────────────────────────────────────────────

@eel.expose
def run_forecast(
    ticker: str,
    strike: float,
    T_days: float,
    option_type: str,
    expiry_str: str,
    history_days: int = 60,
    forecast_days: int = 10,
) -> Dict:
    """
    Lance la prédiction IV avec TimesFM et le repricing de l'option.

    Retourne:
        {
          "iv_forecast": [float, ...],
          "option_prices_forecast": [float, ...],
          "deltas_forecast": [float, ...],
          "iv_history": [float, ...],
          "option_prices_history": [float, ...],
          "deltas_history": [float, ...],
          "x_history": [int, ...],
          "occ_symbol": str,
          "error": str|None
        }
    """
    try:
        from logic.forecast_logic import ForecastLogic
        fl = ForecastLogic(_option_models)

        T_total = max(float(T_days) / 365.0, 1e-6)
        K = float(strike)
        S = _store.S or _data_fetcher.get_live_price(ticker) or 100.0
        r = _store.r or 0.05
        q = _store.q or 0.0
        ticker = ticker.strip().upper()

        occ_symbol = _data_fetcher.build_occ_symbol(ticker, expiry_str, K, option_type)
        hist = _data_fetcher.get_option_history_marketdata(
            ticker, expiry_str, K, option_type, history_days
        )
        if hist is None:
            return {"error": "Impossible de récupérer l'historique de l'option (vérifiez MARKET_DATA_TOKEN)"}

        iv_series = fl.compute_iv_from_prices(
            mid_prices=hist["mid"],
            underlying_prices=hist["underlyingPrice"],
            dtes=hist["dte"],
            strike=K, r=r, q=q, option_type=option_type,
        )

        # Inférence TimesFM
        horizon = forecast_days
        try:
            iv_fc_raw, _qf, iv_series_out = fl.run_iv_forecast(iv_series, horizon)
        except Exception as e_tfm:
            logger.warning("TimesFM failed: %s", e_tfm)
            return {"error": f"TimesFM non disponible: {e_tfm}"}

        iv_fc, opt_prices, deltas, iv_hist, hist_opt, hist_d, x_hist = \
            fl.process_iv_forecast_results(
                iv_fc_raw, iv_series_out, horizon,
                K=K, T_total=T_total, S=float(S), r=r, q=q,
                option_type=option_type,
            )

        def _to_list(arr):
            flat = np.array(arr).flatten()
            return [round(float(v), 6) if v is not None and np.isfinite(v) else None for v in flat]

        return {
            "iv_forecast": _to_list(iv_fc),
            "option_prices_forecast": _to_list(opt_prices),
            "deltas_forecast": _to_list(deltas),
            "iv_history": _to_list(iv_hist),
            "option_prices_history": _to_list(hist_opt),
            "deltas_history": _to_list(hist_d),
            "x_history": list(x_hist),
            "occ_symbol": occ_symbol,
            "error": None,
        }
    except Exception as e:
        logger.exception("run_forecast error")
        return {"error": str(e)}

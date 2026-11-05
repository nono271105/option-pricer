"""
health.py : Diagnostic des sources de données externes (yfinance et FRED).
Permet de valider séparément l'accessibilité réseau et la latence depuis l'environnement d'exécution (notamment Vercel).
"""

from __future__ import annotations

import os
import logging
import time
from typing import Any, Dict
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


def _source_error(source: str, started_at: float, exc: Exception) -> Dict[str, Any]:
    logger.warning(
        "Échec du contrôle de santé %s (%s)",
        source,
        type(exc).__name__,
    )
    return {
        "ok": False,
        "latency_ms": round((time.perf_counter() - started_at) * 1000, 1),
        "error": f"Erreur lors de la vérification de {source}",
    }


def check_health(ticker_symbol: str = "AAPL") -> Dict[str, Any]:
    """Teste indépendamment chaque source externe et retourne son état, sa latence et son erreur éventuelle."""
    sources: Dict[str, Dict[str, Any]] = {}
    ticker = yf.Ticker(ticker_symbol)

    # 1. yfinance : historique 1 jour
    t0 = time.perf_counter()
    try:
        hist = ticker.history(period="1d")
        ok = not hist.empty
        sources["yfinance_history"] = {
            "ok": ok,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": None if ok else "Historique 1j vide",
        }
    except Exception as e:
        sources["yfinance_history"] = _source_error("yfinance_history", t0, e)

    # 2. yfinance : .info
    t0 = time.perf_counter()
    try:
        info = ticker.info
        ok = isinstance(info, dict) and bool(info.get("shortName") or info.get("symbol"))
        sources["yfinance_info"] = {
            "ok": ok,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": None if ok else "Info vide ou introuvable",
        }
    except Exception as e:
        sources["yfinance_info"] = _source_error("yfinance_info", t0, e)

    # 3. yfinance : liste des échéances
    expiries = []
    t0 = time.perf_counter()
    try:
        expiries = list(ticker.options)
        ok = len(expiries) > 0
        sources["yfinance_expiries"] = {
            "ok": ok,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "error": None if ok else "Aucune échéance disponible",
        }
    except Exception as e:
        sources["yfinance_expiries"] = _source_error("yfinance_expiries", t0, e)

    # 4. yfinance : une chaîne d'options
    t0 = time.perf_counter()
    try:
        if expiries:
            chain = ticker.option_chain(expiries[0])
            ok = not chain.calls.empty or not chain.puts.empty
            sources["yfinance_chain"] = {
                "ok": ok,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": None if ok else "Chaîne d'options vide",
            }
        else:
            sources["yfinance_chain"] = {
                "ok": False,
                "latency_ms": 0.0,
                "error": "Aucune échéance disponible pour tester la chaîne",
            }
    except Exception as e:
        sources["yfinance_chain"] = _source_error("yfinance_chain", t0, e)

    # 5. FRED : dernière valeur SOFR
    t0 = time.perf_counter()
    try:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            sources["fred_sofr"] = {
                "ok": False,
                "latency_ms": 0.0,
                "error": "Variable FRED_API_KEY absente",
            }
        else:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": "SOFR",
                    "api_key": api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 5,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            obs = data.get("observations", [])
            valid_val = None
            for o in obs:
                try:
                    valid_val = float(o["value"])
                    break
                except (ValueError, TypeError, KeyError):
                    continue
            ok = valid_val is not None
            sources["fred_sofr"] = {
                "ok": ok,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error": None if ok else "Aucune observation SOFR valide",
            }
    except Exception as e:
        sources["fred_sofr"] = _source_error("fred_sofr", t0, e)

    all_ok = all(s["ok"] for s in sources.values())
    status = "healthy" if all_ok else ("degraded" if any(s["ok"] for s in sources.values()) else "unhealthy")

    return {
        "status": status,
        "sources": sources,
    }


if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    load_dotenv()
    print(json.dumps(check_health(), indent=2))

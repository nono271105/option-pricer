"""
backend/main.py : Point d'entrée de l'application FastAPI.
Expose les 11 endpoints métiers + health check sous le préfixe /api.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from health import check_health
import services
from services import DataSourceError
from validation import _validation_error_response
from schemas import (
    MarketDataRequest,
    ExpiriesRequest,
    OptionChainRequest,
    BsmRequest,
    CrrRequest,
    SimulationRequest,
    SmileRequest,
    SurfaceRequest,
    ExoticRequest,
    StrategyRequest,
    CustomStrategyRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Option Pricer API",
    description="API de valorisation d'options financières et analyse de volatilité",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Gestionnaires d'exceptions ────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    err = _validation_error_response(exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": err["error"], "code": err["code"]},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errs = exc.errors()
    msg = "; ".join(
        f"{'.'.join(str(loc) for loc in e.get('loc', []))}: {e.get('msg', 'Valeur invalide')}"
        for e in errs
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": msg, "code": "VALIDATION_ERROR"},
    )


@app.exception_handler(DataSourceError)
async def data_source_error_handler(request: Request, exc: DataSourceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Erreur interne non gérée : %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Erreur serveur interne."},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return check_health()


@app.get("/api/strategies")
async def get_strategies() -> List[str]:
    return services.get_strategy_names()


@app.post("/api/market-data")
async def market_data(req: MarketDataRequest) -> Dict[str, Any]:
    return services.fetch_market_data(req.ticker)


@app.post("/api/expiries")
async def expiries(req: ExpiriesRequest) -> Dict[str, Any]:
    return services.get_available_expiries(req.ticker)


@app.post("/api/option-chain")
async def option_chain(req: OptionChainRequest) -> Dict[str, Any]:
    return services.get_option_chain(
        ticker=req.ticker,
        expiry_str=req.expiry_str,
        S=req.S,
        r=req.r,
        q=req.q,
    )


@app.post("/api/bsm")
async def calculate_bsm(req: BsmRequest) -> Dict[str, Any]:
    return services.calculate_bsm(
        ticker=req.ticker,
        S=req.S,
        K=req.K,
        maturity_date=req.maturity_date,
        r=req.r,
        q=req.q,
        option_type=req.option_type,
        position=req.position,
        hist_vol=req.hist_vol,
    )


@app.post("/api/crr")
async def calculate_crr(req: CrrRequest) -> Dict[str, Any]:
    return services.calculate_crr(
        ticker=req.ticker,
        S=req.S,
        K=req.K,
        maturity_date=req.maturity_date,
        r=req.r,
        q=req.q,
        N=req.N,
        option_type=req.option_type,
        position=req.position,
        hist_vol=req.hist_vol,
    )


@app.post("/api/simulation")
async def simulation(req: SimulationRequest) -> Dict[str, Any]:
    return services.run_simulation(
        K=req.K,
        T_days=req.T_days,
        r=req.r,
        q=req.q,
        vol_min=req.vol_min,
        vol_max=req.vol_max,
        vol_step=req.vol_step,
        underlying_min=req.underlying_min,
        underlying_max=req.underlying_max,
        underlying_step=req.underlying_step,
        option_type=req.option_type,
    )


@app.post("/api/smile")
async def calculate_smile(req: SmileRequest) -> Dict[str, Any]:
    return services.calculate_smile(
        ticker=req.ticker,
        expiry_str=req.expiry_str,
        S=req.S,
        r=req.r,
        q=req.q,
    )


@app.post("/api/surface")
async def calculate_surface(req: SurfaceRequest) -> Dict[str, Any]:
    return services.calculate_surface(
        ticker=req.ticker,
        S=req.S,
        r=req.r,
        q=req.q,
    )


@app.post("/api/exotic")
async def price_exotic(req: ExoticRequest) -> Dict[str, Any]:
    return services.price_exotic(
        exotic_type=req.exotic_type,
        ticker=req.ticker,
        S=req.S,
        K=req.K,
        maturity_date=req.maturity_date,
        r=req.r,
        q=req.q,
        option_type=req.option_type,
        barrier=req.barrier,
        barrier_type=req.barrier_type or "down-and-out",
        averaging=req.averaging or "arithmetic",
        payoff_amount=req.payoff_amount,
        n_sims=req.n_sims,
        n_steps=req.n_steps,
        seed=req.seed,
        rebate=req.rebate,
        hist_vol=req.hist_vol,
    )


@app.post("/api/strategy")
async def calculate_strategy(req: StrategyRequest) -> Dict[str, Any]:
    return services.calculate_strategy(
        strategy_name=req.strategy_name,
        ticker=req.ticker,
        S=req.S,
        T_days=req.T_days,
        r=req.r,
        q=req.q,
        expiry_str=req.expiry_str,
        hist_vol=req.hist_vol,
    )


@app.post("/api/custom-strategy")
async def calculate_custom_strategy(req: CustomStrategyRequest) -> Dict[str, Any]:
    return services.calculate_custom_strategy(
        ticker=req.ticker,
        S=req.S,
        r=req.r,
        q=req.q,
        hist_vol=req.hist_vol,
        legs=[leg.model_dump() for leg in req.legs],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

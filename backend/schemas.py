"""
backend/schemas.py : Modèles Pydantic pour les requêtes de l'API FastAPI.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

OptionType = Literal["call", "put"]
Position = Literal["long", "short"]
ExoticType = Literal["barrier", "asian", "lookback", "digital"]
BarrierType = Literal["down-and-out", "down-and-in", "up-and-out", "up-and-in"]
Averaging = Literal["arithmetic", "geometric"]


class MarketDataRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Symbole boursier (ex: AAPL)")


class ExpiriesRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Symbole boursier (ex: AAPL)")


class OptionChainRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    expiry_str: str = Field(..., description="Date d'expiration souhaitée (YYYY-MM-DD ou DD/MM/YYYY)")
    S: Optional[float] = None
    r: Optional[float] = None
    q: Optional[float] = None


class BsmRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    S: float
    K: float
    maturity_date: str
    r: float
    q: float
    option_type: OptionType
    position: Position = "long"
    hist_vol: Optional[float] = None


class CrrRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    S: float
    K: float
    maturity_date: str
    r: float
    q: float
    N: int
    option_type: OptionType
    position: Position = "long"
    hist_vol: Optional[float] = None


class SimulationRequest(BaseModel):
    K: float
    T_days: float
    r: float
    q: float
    vol_min: int
    vol_max: int
    vol_step: int
    underlying_min: int
    underlying_max: int
    underlying_step: int
    option_type: OptionType = "call"


class SmileRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    expiry_str: str
    S: Optional[float] = None
    r: Optional[float] = None
    q: Optional[float] = None


class SurfaceRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    S: Optional[float] = None
    r: Optional[float] = None
    q: Optional[float] = None


class ExoticRequest(BaseModel):
    exotic_type: ExoticType
    ticker: str = Field(..., min_length=1)
    S: float
    K: float
    maturity_date: str
    r: float
    q: float
    option_type: OptionType
    barrier: Optional[float] = None
    barrier_type: Optional[BarrierType] = "down-and-out"
    averaging: Optional[Averaging] = "arithmetic"
    payoff_amount: float = 1.0
    n_sims: int = 50_000
    n_steps: int = 252
    seed: int = 42
    rebate: float = 0.0
    hist_vol: Optional[float] = None


class StrategyRequest(BaseModel):
    strategy_name: str
    ticker: str = Field(..., min_length=1)
    S: float
    T_days: float
    r: float
    q: float
    expiry_str: str
    hist_vol: Optional[float] = None


class CustomStrategyLegRequest(BaseModel):
    option_type: OptionType
    position: Position
    strike: float = Field(..., gt=0)
    expiry_str: str = Field(..., min_length=1)


class CustomStrategyRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    S: float = Field(..., gt=0)
    r: float
    q: float = Field(..., ge=0, lt=1)
    hist_vol: Optional[float] = Field(default=None, gt=0, le=10)
    legs: list[CustomStrategyLegRequest] = Field(..., min_length=1, max_length=50)

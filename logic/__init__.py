"""
logic/ — Modules de calcul (logique métier) pour chaque onglet.

Réexporte les classes publiques pour un accès simplifié :
    from logic import OptionModels, CRRModels, StrategyManager, ...
"""

from logic.bsm_logic import OptionModels
from logic.crr_logic import CRRModels
from logic.simulation_logic import SimulationLogic
from logic.volatility_smile_logic import VolatilitySmileLogic
from logic.volatility_surface_logic import ImpliedVolatilitySurface
from logic.exotic_options_logic import (
    ExoticResult,
    price_barrier_analytical, price_barrier_mc,
    price_asian_mc,
    price_lookback_mc,
    price_digital_analytical, price_digital_mc,
)
from logic.strategy_logic import StrategyManager
from logic.forecast_logic import ForecastLogic

__all__ = [
    "OptionModels", "CRRModels", "SimulationLogic",
    "VolatilitySmileLogic", "ImpliedVolatilitySurface",
    "ExoticResult",
    "price_barrier_analytical", "price_barrier_mc",
    "price_asian_mc", "price_lookback_mc",
    "price_digital_analytical", "price_digital_mc",
    "StrategyManager", "ForecastLogic",
]

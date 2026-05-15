"""
UI/ — Composants graphiques PySide6, un fichier par onglet.
"""

from UI.bsm_ui import BSMTab
from UI.crr_ui import CRRModelTab
from UI.simulation_ui import CallPriceSimulationTab
from UI.volatility_smile_ui import VolatilitySmileTab
from UI.volatility_surface_ui import VolatilitySurfaceTab
from UI.exotic_options_ui import ExoticOptionsTab
from UI.strategy_ui import StrategyTab
from UI.forecast_ui import ForecastTimesFMTab

__all__ = [
    "BSMTab", "CRRModelTab", "CallPriceSimulationTab",
    "VolatilitySmileTab", "VolatilitySurfaceTab",
    "ExoticOptionsTab", "StrategyTab", "ForecastTimesFMTab",
]

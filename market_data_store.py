# Source de vérité unique pour les données de marché partagées entre tous les onglets

from typing import Optional


class MarketDataStore:
    """Source de vérité unique pour les données de marché partagées."""

    def __init__(self):
        self.S: Optional[float] = None
        self.r: Optional[float] = None
        self.q: Optional[float] = None
        self.sigma: Optional[float] = None
        self.historical_vol: Optional[float] = None
        self.ticker: Optional[str] = None
        self.company_name: Optional[str] = None
        self.pricing_method: str = "N/A"

    def update(self, **kwargs) -> None:
        """Met à jour les attributs."""
        for k, v in kwargs.items():
            setattr(self, k, v)

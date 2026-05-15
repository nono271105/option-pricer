"""
MarketDataStore — Source de vérité unique pour les données de marché partagées.

Pattern pub/sub : chaque onglet s'abonne via subscribe() et reçoit
automatiquement les mises à jour lorsque les données de marché changent.
"""

import logging
from typing import Optional, Callable, List

logger = logging.getLogger(__name__)


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
        self._subscribers: List[Callable] = []

    def subscribe(self, callback: Callable) -> None:
        """Ajoute un callback qui sera appelé à chaque update()."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Retire un callback de la liste des abonnés."""
        self._subscribers.remove(callback)

    def update(self, **kwargs) -> None:
        """Met à jour les attributs et notifie tous les abonnés."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._notify()

    def _notify(self) -> None:
        """Appelle tous les callbacks abonnés avec self en argument."""
        for cb in self._subscribers:
            try:
                cb(self)
            except Exception as e:
                logger.warning("Subscriber error: %s", e)

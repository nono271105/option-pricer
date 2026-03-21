"""
Module de cache pour les données API avec TTL (Time To Live).
Évite les requêtes API répétées en stockant les résultats localement.
"""

from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import threading


class DataCache:
    """
    Cache thread-safe avec expiration automatique (TTL).
    
    Exemple:
        cache = DataCache(ttl_seconds=3600)  # 1 heure
        cache.set('AAPL_price', 150.25)
        price = cache.get('AAPL_price')  # Retourne 150.25 si pas expiré
    """
    
    def __init__(self, ttl_seconds: int = 3600) -> None:
        """
        Initialise le cache.
        
        Args:
            ttl_seconds: Temps avant expiration en secondes (défaut: 3600 = 1 heure)
        """
        self.ttl: int = ttl_seconds
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache si elle existe et n'a pas expiré.
        
        Args:
            key: Clé du cache
            
        Returns:
            Optional[Any]: Valeur du cache ou None si expiré/inexistant
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            elapsed = (datetime.now() - timestamp).total_seconds()
            
            if elapsed > self.ttl:
                # Valeur expirée, la supprimer
                del self._cache[key]
                return None
            
            return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Enregistre une valeur dans le cache.
        
        Args:
            key: Clé du cache
            value: Valeur à stocker
        """
        with self._lock:
            self._cache[key] = (value, datetime.now())
    
    def clear(self, key: Optional[str] = None) -> None:
        """
        Efface une entrée ou tout le cache.
        
        Args:
            key: Clé spécifique à effacer, ou None pour tout effacer
        """
        with self._lock:
            if key:
                if key in self._cache:
                    del self._cache[key]
            else:
                self._cache.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """
        Retourne les statistiques du cache.
        
        Returns:
            Dict avec nombre d'entrées et TTL
        """
        with self._lock:
            return {
                'entries': len(self._cache),
                'ttl_seconds': self.ttl
            }


# Instance globale du cache (partagée entre tous les modules)
global_cache = DataCache(ttl_seconds=3600)  # 1 heure par défaut

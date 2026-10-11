"""
conftest.py : Configuration pytest pour la suite de tests Option Pricer.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def pytest_configure(config):
    """Déclare les marqueurs personnalisés."""
    config.addinivalue_line(
        "markers",
        "slow: marque les tests lents (Monte Carlo) : exclure avec '-m \"not slow\"'",
    )
    config.addinivalue_line(
        "markers",
        "integration: tests réseau ou nécessitant une clé API : exclus par défaut",
    )


def pytest_report_header(config):
    """Affiche les tolérances en en-tête du rapport pytest."""
    return [
        "Option Pricer : Suite de régression pricing",
        "Tolérances : ±0.01$ (analytique)  |  ±0.10$ (MC)  |  ±2.00$ (lookback MC vs analytique)",
    ]


@pytest.fixture(autouse=True)
def _clear_global_caches():
    """Vide les caches globaux avant chaque test pour éviter les contaminations entre tests."""
    from cache import global_cache_price, global_cache_chain, global_cache_static
    global_cache_price.clear()
    global_cache_chain.clear()
    global_cache_static.clear()
    yield
    # Nettoyage post-test (optionnel mais symétrique)
    global_cache_price.clear()
    global_cache_chain.clear()
    global_cache_static.clear()

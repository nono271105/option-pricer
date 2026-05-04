"""
conftest.py : Configuration pytest pour la suite de tests Option Pricer.
"""

import pytest


def pytest_configure(config):
    """Déclare les marqueurs personnalisés."""
    config.addinivalue_line(
        "markers",
        "slow: marque les tests lents (Monte Carlo) — exclure avec '-m \"not slow\"'",
    )


def pytest_report_header(config):
    """Affiche les tolérances en en-tête du rapport pytest."""
    return [
        "Option Pricer : Suite de régression pricing",
        "Tolérances : ±0.01$ (analytique)  |  ±0.10$ (MC)  |  ±2.00$ (lookback MC vs analytique)",
    ]

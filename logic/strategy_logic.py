import logging
import math
import numpy as np
from typing import Literal, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class StrategyManager:
    def __init__(self) -> None:
        pass

    # =========================================================================
    # Définitions des stratégies
    # =========================================================================
    # dictionnaire de composition des stratégies, où l'offset définit l'écartement des strikes par rapport au spot

    STRATEGY_DEFINITIONS: Dict[str, List[Dict]] = {
        #  Positions de base 
        "Long Call":  [{"type": "call", "pos": "long",  "offset": 0.0}],
        "Short Call": [{"type": "call", "pos": "short", "offset": 0.0}],
        "Long Put":   [{"type": "put",  "pos": "long",  "offset": 0.0}],
        "Short Put":  [{"type": "put",  "pos": "short", "offset": 0.0}],

        # Spreads directionnels
        "Bull Call Spread": [
            {"type": "call", "pos": "long",  "offset":  0.00},
            {"type": "call", "pos": "short", "offset": +0.05},
        ],
        "Bear Call Spread": [
            {"type": "call", "pos": "short", "offset":  0.00},
            {"type": "call", "pos": "long",  "offset": +0.05},
        ],
        "Bull Put Spread": [
            {"type": "put",  "pos": "short", "offset":  0.00},
            {"type": "put",  "pos": "long",  "offset": -0.05},
        ],
        "Bear Put Spread": [
            {"type": "put",  "pos": "long",  "offset":  0.00},
            {"type": "put",  "pos": "short", "offset": -0.05},
        ],

        # Volatilité
        "Long Straddle": [
            {"type": "call", "pos": "long", "offset": 0.0},
            {"type": "put",  "pos": "long", "offset": 0.0},
        ],
        "Short Straddle": [
            {"type": "call", "pos": "short", "offset": 0.0},
            {"type": "put",  "pos": "short", "offset": 0.0},
        ],
        "Long Strangle": [
            {"type": "call", "pos": "long", "offset": +0.05},
            {"type": "put",  "pos": "long", "offset": -0.05},
        ],
        "Short Strangle": [
            {"type": "call", "pos": "short", "offset": +0.05},
            {"type": "put",  "pos": "short", "offset": -0.05},
        ],

        # Butterflies
        "Long Call Butterfly": [
            {"type": "call", "pos": "long",  "offset": -0.05},
            {"type": "call", "pos": "short", "offset":  0.00},
            {"type": "call", "pos": "short", "offset":  0.00},
            {"type": "call", "pos": "long",  "offset": +0.05},
        ],
        "Short Call Butterfly": [
            {"type": "call", "pos": "short", "offset": -0.05},
            {"type": "call", "pos": "long",  "offset":  0.00},
            {"type": "call", "pos": "long",  "offset":  0.00},
            {"type": "call", "pos": "short", "offset": +0.05},
        ],
        "Long Put Butterfly": [
            {"type": "put",  "pos": "long",  "offset": -0.05},
            {"type": "put",  "pos": "short", "offset":  0.00},
            {"type": "put",  "pos": "short", "offset":  0.00},
            {"type": "put",  "pos": "long",  "offset": +0.05},
        ],
        "Short Put Butterfly": [
            {"type": "put",  "pos": "short", "offset": -0.05},
            {"type": "put",  "pos": "long",  "offset":  0.00},
            {"type": "put",  "pos": "long",  "offset":  0.00},
            {"type": "put",  "pos": "short", "offset": +0.05},
        ],
        "Long Iron Butterfly": [
            {"type": "put",  "pos": "long",  "offset": -0.05},
            {"type": "put",  "pos": "short", "offset":  0.00},
            {"type": "call", "pos": "short", "offset":  0.00},
            {"type": "call", "pos": "long",  "offset": +0.05},
        ],
        "Short Iron Butterfly": [
            {"type": "put",  "pos": "short", "offset": -0.05},
            {"type": "put",  "pos": "long",  "offset":  0.00},
            {"type": "call", "pos": "long",  "offset":  0.00},
            {"type": "call", "pos": "short", "offset": +0.05},
        ],

        # Condors
        "Long Call Condor": [
            {"type": "call", "pos": "long",  "offset": -0.075},
            {"type": "call", "pos": "short", "offset": -0.025},
            {"type": "call", "pos": "short", "offset": +0.025},
            {"type": "call", "pos": "long",  "offset": +0.075},
        ],
        "Short Call Condor": [
            {"type": "call", "pos": "short", "offset": -0.075},
            {"type": "call", "pos": "long",  "offset": -0.025},
            {"type": "call", "pos": "long",  "offset": +0.025},
            {"type": "call", "pos": "short", "offset": +0.075},
        ],
        "Long Put Condor": [
            {"type": "put",  "pos": "long",  "offset": -0.075},
            {"type": "put",  "pos": "short", "offset": -0.025},
            {"type": "put",  "pos": "short", "offset": +0.025},
            {"type": "put",  "pos": "long",  "offset": +0.075},
        ],
        "Short Put Condor": [
            {"type": "put",  "pos": "short", "offset": -0.075},
            {"type": "put",  "pos": "long",  "offset": -0.025},
            {"type": "put",  "pos": "long",  "offset": +0.025},
            {"type": "put",  "pos": "short", "offset": +0.075},
        ],
        "Long Iron Condor": [
            {"type": "put",  "pos": "long",  "offset": -0.075},
            {"type": "put",  "pos": "short", "offset": -0.025},
            {"type": "call", "pos": "short", "offset": +0.025},
            {"type": "call", "pos": "long",  "offset": +0.075},
        ],
        "Short Iron Condor": [
            {"type": "put",  "pos": "short", "offset": -0.075},
            {"type": "put",  "pos": "long",  "offset": -0.025},
            {"type": "call", "pos": "long",  "offset": +0.025},
            {"type": "call", "pos": "short", "offset": +0.075},
        ],
    }

    # =========================================================================
    # Construction des legs
    # =========================================================================

    def build_legs(self, strategy_name: str, S: float, T: float,
                   r: float, sigma: float, q: float,
                   maturity_datetime: datetime, ticker: str,
                   data_fetcher, option_models) -> List[Dict]:
        """
        Construit la liste de legs pour une stratégie donnée.
        Chaque leg contient : option_type, position, strike, premium (marché ou BSM).

        Returns:
            List[Dict] avec clés : option_type, position, strike, premium
        """
        definition = self.STRATEGY_DEFINITIONS.get(strategy_name)
        if definition is None:
            raise ValueError(f"Stratégie inconnue : {strategy_name}")

        legs = []
        for leg_def in definition:
            strike = math.ceil(S * (1 + leg_def["offset"]))

            # valorisation hybride de la jambe : extraction de la cotation réelle ou pricing théorique si illiquide
            premium = self._get_premium(
                ticker, S, strike, T, r, sigma, q,
                leg_def["type"], maturity_datetime,
                data_fetcher, option_models
            )

            legs.append({
                "option_type": leg_def["type"],
                "position":    leg_def["pos"],
                "strike":      strike,
                "premium":     premium,
            })

        return legs

    def _get_premium(self, ticker: str, S_current: float, strike: float, T: float,
                     r: float, sigma: float, q: float,
                     option_type: str, maturity_datetime: datetime,
                     data_fetcher, option_models) -> float:
        """
        Récupère la prime d'un leg : prix de marché via yfinance si dispo,
        sinon calcul BSM.
        """
        try:
            _, market_price, _ = data_fetcher.get_implied_volatility_and_price(
                ticker, strike, maturity_datetime, option_type)
            if market_price is not None and market_price > 0:
                return float(market_price)
        except Exception as exc:
            logger.warning("Prime marché indisponible pour %s %s K=%s, fallback BSM: %s",
                           ticker, option_type, strike, exc)

        # si la cotation n'est pas fiable, on simule le prix théorique exact
        return option_models.black_scholes_price(S=S_current, K=strike, T=T,
                              r=r, sigma=sigma, q=q,
                              option_type=option_type)

    # =========================================================================
    # Calcul du payoff à maturité
    # =========================================================================

    def compute_payoff(self, legs: List[Dict], S_range: np.ndarray) -> np.ndarray:
        """
        Somme des payoffs nets de tous les legs sur la plage de prix S_range.
        """
        total = np.zeros(len(S_range))
        for leg in legs:
            total += self.calculate_single_option_payoff(
                S_range,
                leg["strike"],
                leg["premium"],
                leg["option_type"],
                leg["position"],
            )
        return total

    # =========================================================================
    # Valeur actuelle de la stratégie (avec valeur temps)
    # =========================================================================

    def compute_value_today(self, legs: List[Dict], S_range: np.ndarray,
                             S_current: float, T: float, r: float,
                             sigma: float, q: float, option_models) -> np.ndarray:
        """
        Valeur actuelle de la stratégie pour chaque S hypothétique dans S_range.
        Utilise BSM pour évaluer chaque leg au spot S_i avec les paramètres actuels.
        Soustrait le coût initial de la stratégie.
        """
        cost = sum(
            leg["premium"] if leg["position"] == "long" else -leg["premium"]
            for leg in legs
        )

        total = np.full(len(S_range), -cost, dtype=float)
        for leg in legs:
            bsm = option_models.black_scholes_price(
                S=S_range, K=leg["strike"], T=T,
                r=r, sigma=sigma, q=q,
                option_type=leg["option_type"]
            )
            if leg["position"] == "long":
                total += bsm
            else:
                total -= bsm

        return total

    # =========================================================================
    # Métriques : coût, breakevens, gain/perte max
    # =========================================================================

    def compute_metrics(self, legs: List[Dict], S_range: np.ndarray,
                         payoff: np.ndarray) -> Dict:
        """
        Calcule les métriques clés de la stratégie.

        Returns:
            Dict avec clés : cost, breakevens, max_gain, max_loss
        """
        # calcul de l'investissement initial
        cost = sum(
            leg["premium"] if leg["position"] == "long" else -leg["premium"]
            for leg in legs
        )

        # détermination des points morts
        breakevens = []
        for i in range(len(payoff) - 1):
            if payoff[i] * payoff[i + 1] < 0:
                # interpolation linéaire pour affiner la précision
                be = S_range[i] + (0 - payoff[i]) * (S_range[i+1] - S_range[i]) / (payoff[i+1] - payoff[i])
                breakevens.append(round(float(be), 2))

        breakevens = sorted(set(breakevens))

        # extraction des bornes de rentabilité théorique
        max_gain = float(np.max(payoff))
        max_loss = float(np.min(payoff))

        # détection du risque ou gain asymétrique illimité
        if payoff[-1] > payoff[-2] or payoff[0] > payoff[1]:
            max_gain = np.inf
        if payoff[-1] < payoff[-2] or payoff[0] < payoff[1]:
            max_loss = -np.inf

        return {
            "cost":       round(cost, 4),
            "breakevens": breakevens,
            "max_gain":   max_gain,
            "max_loss":   abs(max_loss) if np.isfinite(max_loss) else np.inf,
        }

    # =========================================================================
    # Grecs agrégés
    # =========================================================================

    def compute_greeks(self, legs: List[Dict], S: float, T: float,
                        r: float, sigma: float, q: float,
                        option_models) -> Dict[str, float]:
        """
        Somme des grecs BSM de tous les legs.
        Les legs short ont leurs grecs inversés.
        """
        total = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        for leg in legs:
            g = option_models.calculate_greeks(
                S=S, K=leg["strike"], T=T,
                r=r, sigma=sigma, q=q,
                option_type=leg["option_type"]
            )
            sign = 1 if leg["position"] == "long" else -1
            for key in total:
                total[key] += sign * g.get(key, 0.0)
        total["vega"] = total["vega"] / 100.0
        return {k: round(v, 6) for k, v in total.items()}

    # =========================================================================
    # Méthodes existantes (inchangées)
    # =========================================================================

    def calculate_single_option_payoff(
        self, 
        S_range: np.ndarray, 
        K: float, 
        premium: float, 
        option_type: Literal['call', 'put'], 
        position: Literal['long', 'short']
    ) -> np.ndarray:
        """
        Calcule le payoff net à maturité pour une seule option.
        """
        if option_type == 'call':
            gross_payoff = np.maximum(S_range - K, 0)
        elif option_type == 'put':
            gross_payoff = np.maximum(K - S_range, 0)
        else:
            raise ValueError("option_type doit être 'call' ou 'put'")

        if position == 'long':
            net_payoff = gross_payoff - premium
        elif position == 'short':
            net_payoff = -gross_payoff + premium
        else:
            raise ValueError("position doit être 'long' ou 'short'")
        return net_payoff
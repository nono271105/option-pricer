from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cache import global_cache_iv
from logic.bsm_logic import OptionModels
from logic.volatility_resolver import ImpliedVolatilityResolver


class FakeFetcher:
    def __init__(self, expiries=None, chains=None, historical_vol=None):
        self.expiries = expiries or []
        self.chains = chains or {}
        self.historical_vol = historical_vol

    def get_available_expiries(self, ticker):
        return self.expiries

    def get_option_chain_for_expiry(self, ticker, expiry):
        return self.chains.get(expiry)

    def get_historical_volatility(self, ticker, period):
        return self.historical_vol


def expiry_in(days):
    return (date.today() + timedelta(days=days)).isoformat()


def quote_row(strike, expiry, sigma, option_type="call", *, wide=False, no_quote=False):
    S, r, q = 100.0, 0.04, 0.01
    T = max((date.fromisoformat(expiry) - date.today()).days, 1) / 365
    price = OptionModels().black_scholes_price(
        S, strike, T, r, sigma, q, option_type
    )
    if no_quote:
        bid, ask = 0.0, 0.0
    elif wide:
        bid, ask = max(0.01, price * 0.2), price * 1.8
    else:
        bid, ask = max(0.001, price - 0.01), price + 0.01
    return {
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "impliedVolatility": sigma,
        "volume": 20,
        "openInterest": 100,
    }


def chain(calls=None, puts=None):
    return SimpleNamespace(
        calls=pd.DataFrame(calls or []),
        puts=pd.DataFrame(puts or []),
    )


@pytest.fixture(autouse=True)
def clear_iv_cache():
    global_cache_iv.clear()
    yield
    global_cache_iv.clear()


def test_exact_contract_iv_is_recomputed_from_usable_bid_ask_mid():
    expiry = expiry_in(35)
    option_chain = chain(calls=[quote_row(100, expiry, 0.31)])
    resolver = ImpliedVolatilityResolver(FakeFetcher([expiry], {expiry: option_chain}))

    result = resolver.resolve("AAPL", 100, 100, expiry, 0.04, 0.01, "call", 0.18)

    assert result.sigma == pytest.approx(0.31, abs=0.01)
    assert result.source == "IV recalculée (bid/ask)"
    assert result.market_price is not None
    assert result.expiry_used == expiry


def test_exact_yahoo_iv_is_used_when_no_bid_ask_is_available():
    expiry = expiry_in(35)
    option_chain = chain(calls=[quote_row(100, expiry, 0.29, no_quote=True)])
    resolver = ImpliedVolatilityResolver(FakeFetcher([expiry], {expiry: option_chain}))

    result = resolver.resolve("AAPL", 100, 100, expiry, 0.04, 0.01, "call", 0.18)

    assert result.sigma == 0.29
    assert result.source == "IV Yahoo (contrat exact)"
    assert result.market_price is None


def test_missing_strike_interpolates_liquid_same_expiry_quotes():
    expiry = expiry_in(35)
    option_chain = chain(calls=[
        quote_row(101, expiry, 0.20),
        quote_row(110, expiry, 0.35),
    ])
    resolver = ImpliedVolatilityResolver(FakeFetcher([expiry], {expiry: option_chain}))

    result = resolver.resolve("AAPL", 100, 105, expiry, 0.04, 0.01, "call", 0.18)

    assert 0.20 < result.sigma < 0.35
    assert result.source == "IV interpolée (même échéance)"
    assert result.market_price is None


def test_missing_expiry_interpolates_total_variance_between_neighbor_expiries():
    target = expiry_in(30)
    before, after = expiry_in(20), expiry_in(40)
    chains = {
        before: chain(calls=[
            quote_row(101, before, 0.20),
            quote_row(110, before, 0.20),
        ]),
        after: chain(calls=[
            quote_row(101, after, 0.40),
            quote_row(110, after, 0.40),
        ]),
    }
    resolver = ImpliedVolatilityResolver(FakeFetcher([before, after], chains))

    result = resolver.resolve("AAPL", 100, 105, target, 0.04, 0.01, "call", 0.18)

    expected = ((0.20**2 * 20 + 0.40**2 * 40) / 2 / 30) ** 0.5
    assert result.sigma == pytest.approx(expected, rel=0.02)
    assert result.source == "IV interpolée (maturité et moneyness)"


def test_unusable_exact_quote_falls_back_to_historical_volatility():
    expiry = expiry_in(35)
    option_chain = chain(calls=[quote_row(100, expiry, 0.40, wide=True)])
    resolver = ImpliedVolatilityResolver(FakeFetcher([expiry], {expiry: option_chain}))

    result = resolver.resolve("AAPL", 100, 100, expiry, 0.04, 0.01, "call", 0.27)

    assert result.sigma == 0.27
    assert result.source == "Volatilité historique"


def test_missing_market_and_historical_volatility_uses_twenty_percent_default():
    resolver = ImpliedVolatilityResolver(FakeFetcher(historical_vol=None))

    result = resolver.resolve(
        "AAPL", 100, 100, expiry_in(35), 0.04, 0.01, "call", None
    )

    assert result.sigma == 0.20
    assert result.source == "Défaut (20 %)"


def test_historical_volatility_is_fetched_when_not_supplied_by_request():
    resolver = ImpliedVolatilityResolver(FakeFetcher(historical_vol=0.27))

    result = resolver.resolve(
        "AAPL", 100, 100, expiry_in(35), 0.04, 0.01, "call", None
    )

    assert result.sigma == 0.27
    assert result.source == "Volatilité historique"


def test_no_maturity_extrapolation_uses_historical_volatility():
    only_expiry = expiry_in(20)
    option_chain = chain(calls=[
        quote_row(101, only_expiry, 0.20),
        quote_row(110, only_expiry, 0.25),
    ])
    resolver = ImpliedVolatilityResolver(
        FakeFetcher([only_expiry], {only_expiry: option_chain})
    )

    result = resolver.resolve(
        "AAPL", 100, 105, expiry_in(40), 0.04, 0.01, "call", 0.29
    )

    assert result.sigma == 0.29
    assert result.source == "Volatilité historique"


def test_surface_grid_does_not_extrapolate_outside_observed_quotes():
    from logic.volatility_surface_logic import ImpliedVolatilitySurface

    data = pd.DataFrame({
        "Strike": [100.0, 110.0, 100.0],
        "Days_to_Maturity": [10.0, 10.0, 20.0],
        "IV": [0.20, 0.25, 0.30],
    })

    _, _, values = ImpliedVolatilitySurface().interpolate_surface(
        data, strike_grid_size=3, maturity_grid_size=3
    )

    assert np.isnan(values[-1, -1])
    assert np.isfinite(values[0, 0])

# Commande : pytest tests/test_app.py -v

"""
test_app.py : Tests unitaires pour tous les modules non-pricing.

Notes de migration (PySide6 → React/Eel) :
    - TestUtils supprimé : get_default_maturity_date (utils.py) et PySide6 ne font
      plus partie du projet depuis la migration vers Eel.
    - TestForecastLogic adapté : process_iv_forecast_results reçoit désormais un
      array 1D directement (le découpage [0] est fait en amont dans run_iv_forecast).
"""

import sys
import os
import time
import threading
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cache import DataCache, global_cache
from data_fetcher import DataFetcher
from market_data_store import MarketDataStore
from logic.strategy_logic import StrategyManager
from logic.simulation_logic import SimulationLogic
from logic.forecast_logic import ForecastLogic
from logic.volatility_smile_logic import VolatilitySmileLogic
from logic.bsm_logic import OptionModels


# ============================================================================
#  CACHE
# ============================================================================

class TestDataCache:
    """Tests pour le module cache.py."""

    def test_set_and_get(self):
        cache = DataCache(ttl_seconds=60)
        cache.set("key1", 42)
        assert cache.get("key1") == 42

    def test_get_missing_key_returns_none(self):
        cache = DataCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        cache = DataCache(ttl_seconds=1)
        cache.set("expire_me", "value")
        assert cache.get("expire_me") == "value"
        time.sleep(1.1)
        assert cache.get("expire_me") is None

    def test_clear_specific_key(self):
        cache = DataCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear("a")
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_clear_all(self):
        cache = DataCache()
        cache.set("x", 10)
        cache.set("y", 20)
        cache.clear()
        assert cache.get("x") is None
        assert cache.get("y") is None

    def test_get_stats(self):
        cache = DataCache(ttl_seconds=300)
        cache.set("s1", "v1")
        cache.set("s2", "v2")
        stats = cache.get_stats()
        assert stats["entries"] == 2
        assert stats["ttl_seconds"] == 300

    def test_overwrite_existing_key(self):
        cache = DataCache()
        cache.set("k", "old")
        cache.set("k", "new")
        assert cache.get("k") == "new"

    def test_thread_safety(self):
        cache = DataCache(ttl_seconds=60)
        errors = []

        def writer(start):
            try:
                for i in range(100):
                    cache.set(f"key_{start + i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n * 100,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.get_stats()["entries"] == 400

    def test_global_cache_instance(self):
        """global_cache est bien une instance partagée avec TTL 1h."""
        assert isinstance(global_cache, DataCache)
        assert global_cache.ttl == 3600


# ============================================================================
#  MARKET DATA STORE
# ============================================================================

class TestMarketDataStore:
    """Tests pour market_data_store.py."""

    def test_subscribe_and_notify(self):
        store = MarketDataStore()
        received = []
        store.subscribe(lambda s: received.append(s.S))
        store.update(S=150.0)
        assert received == [150.0]

    def test_multiple_subscribers(self):
        store = MarketDataStore()
        calls = []
        store.subscribe(lambda s: calls.append("tab1"))
        store.subscribe(lambda s: calls.append("tab2"))
        store.update(S=100.0)
        assert calls == ["tab1", "tab2"]

    def test_unsubscribe(self):
        store = MarketDataStore()
        calls = []
        cb = lambda s: calls.append("called")
        store.subscribe(cb)
        store.unsubscribe(cb)
        store.update(S=100.0)
        assert calls == []

    def test_update_attributes(self):
        store = MarketDataStore()
        store.update(S=100.0, r=0.05, q=0.01, ticker="AAPL")
        assert store.S == 100.0
        assert store.r == 0.05
        assert store.q == 0.01
        assert store.ticker == "AAPL"

    def test_subscriber_error_does_not_crash(self):
        def bad_subscriber(s):
            raise ZeroDivisionError("intentional error to test resilience")

        store = MarketDataStore()
        store.subscribe(bad_subscriber)
        healthy_calls = []
        store.subscribe(lambda s: healthy_calls.append(s.S))
        store.update(S=42.0)  # should not raise
        assert healthy_calls == [42.0]


# ============================================================================
#  DATA_FETCHER  (appels réseau mockés)
# ============================================================================

class TestDataFetcher:
    """Tests pour data_fetcher.py avec mocks réseau."""

    def setup_method(self):
        global_cache.clear()
        self.fetcher = DataFetcher()

    # ── get_live_price ──────────────────────────────────────────────────

    @patch("data_fetcher.yf.Ticker")
    def test_get_live_price_success(self, mock_ticker_cls):
        mock_hist = pd.DataFrame({"Close": [150.0]})
        mock_ticker_cls.return_value.history.return_value = mock_hist
        price = self.fetcher.get_live_price("AAPL")
        assert price == 150.0

    @patch("data_fetcher.yf.Ticker")
    def test_get_live_price_empty(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()
        assert self.fetcher.get_live_price("FAKE") is None

    @patch("data_fetcher.yf.Ticker")
    def test_get_live_price_cached(self, mock_ticker_cls):
        mock_hist = pd.DataFrame({"Close": [200.0]})
        mock_ticker_cls.return_value.history.return_value = mock_hist

        p1 = self.fetcher.get_live_price("MSFT")
        p2 = self.fetcher.get_live_price("MSFT")
        assert p1 == p2 == 200.0
        # yfinance ne doit être appelé qu'une seule fois grâce au cache
        mock_ticker_cls.return_value.history.assert_called_once()

    # ── get_historical_volatility ───────────────────────────────────────

    @patch("data_fetcher.yf.Ticker")
    def test_historical_volatility_valid(self, mock_ticker_cls):
        np.random.seed(0)
        prices = pd.DataFrame({"Close": 100 + np.cumsum(np.random.randn(252))})
        mock_ticker_cls.return_value.history.return_value = prices
        vol = self.fetcher.get_historical_volatility("AAPL")
        assert vol is not None
        assert 0 < vol < 2  # volatilité raisonnable

    @patch("data_fetcher.yf.Ticker")
    def test_historical_volatility_empty(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()
        assert self.fetcher.get_historical_volatility("FAKE") is None

    # ── get_sofr_rate ───────────────────────────────────────────────────

    @patch("data_fetcher.requests.get")
    def test_get_sofr_rate_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "observations": [{"value": "5.33"}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        rate = self.fetcher.get_sofr_rate()
        assert rate == pytest.approx(0.0533, abs=1e-6)

    @patch("data_fetcher.requests.get")
    def test_get_sofr_rate_no_observations(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"observations": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        assert self.fetcher.get_sofr_rate() is None

    # ── get_dividend_yield ──────────────────────────────────────────────

    @patch("data_fetcher.yf.Ticker")
    def test_dividend_yield_trailing(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {"trailingAnnualDividendYield": 0.006}
        div = self.fetcher.get_dividend_yield("AAPL")
        assert div == pytest.approx(0.006, abs=1e-6)

    @patch("data_fetcher.yf.Ticker")
    def test_dividend_yield_fallback_zero(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {}
        div = self.fetcher.get_dividend_yield("NODIV")
        assert div == 0.0

    # ── get_company_name ────────────────────────────────────────────────

    @patch("data_fetcher.yf.Ticker")
    def test_get_company_name(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {"longName": "Apple Inc."}
        name = self.fetcher.get_company_name("AAPL")
        assert name == "Apple Inc."

    @patch("data_fetcher.yf.Ticker")
    def test_get_company_name_fallback(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {}
        name = self.fetcher.get_company_name("XYZ")
        assert name == "XYZ"


# ============================================================================
#  STRATEGY_MANAGER  (payoff, métriques, grecs — pas de réseau)
# ============================================================================

class TestStrategyManager:
    """Tests pour strategy_manager.py (logique pure, pas build_legs)."""

    def setup_method(self):
        self.sm = StrategyManager()
        self.S_range = np.linspace(80, 120, 500)

    # ── Définitions ─────────────────────────────────────────────────────

    def test_all_strategies_defined(self):
        names = list(StrategyManager.STRATEGY_DEFINITIONS.keys())
        assert len(names) >= 20  # au minimum les 20+ stratégies actuelles
        for name in ["Long Call", "Short Put", "Long Straddle",
                      "Long Iron Condor", "Bull Call Spread"]:
            assert name in names

    # ── single option payoff ────────────────────────────────────────────

    def test_long_call_payoff(self):
        S = np.array([90.0, 100.0, 110.0])
        payoff = self.sm.calculate_single_option_payoff(S, K=100, premium=5,
                                                         option_type="call", position="long")
        np.testing.assert_array_almost_equal(payoff, [-5.0, -5.0, 5.0])

    def test_short_put_payoff(self):
        S = np.array([90.0, 100.0, 110.0])
        payoff = self.sm.calculate_single_option_payoff(S, K=100, premium=5,
                                                         option_type="put", position="short")
        np.testing.assert_array_almost_equal(payoff, [-5.0, 5.0, 5.0])

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            self.sm.calculate_single_option_payoff(np.array([100.0]), 100, 5, "forward", "long")

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError):
            self.sm.calculate_single_option_payoff(np.array([100.0]), 100, 5, "call", "neutral")

    # ── compute_payoff (multi-leg) ──────────────────────────────────────

    def test_long_straddle_payoff_profile(self):
        """Un long straddle doit avoir un payoff en V."""
        legs = [
            {"option_type": "call", "position": "long", "strike": 100, "premium": 5},
            {"option_type": "put",  "position": "long", "strike": 100, "premium": 5},
        ]
        payoff = self.sm.compute_payoff(legs, self.S_range)
        # Au centre (S=100), perte maximale = -10
        idx_atm = np.argmin(np.abs(self.S_range - 100))
        assert payoff[idx_atm] == pytest.approx(-10.0, abs=0.5)
        # Aux extrémités, le payoff est positif
        assert payoff[0] > 0
        assert payoff[-1] > 0

    # ── compute_metrics ─────────────────────────────────────────────────

    def test_metrics_long_call(self):
        legs = [{"option_type": "call", "position": "long", "strike": 100, "premium": 5}]
        payoff = self.sm.compute_payoff(legs, self.S_range)
        metrics = self.sm.compute_metrics(legs, self.S_range, payoff)

        assert metrics["cost"] == pytest.approx(5.0, abs=0.01)
        assert len(metrics["breakevens"]) == 1
        assert metrics["breakevens"][0] == pytest.approx(105.0, abs=0.5)
        assert metrics["max_gain"] == np.inf   # gain illimité
        assert metrics["max_loss"] == pytest.approx(5.0, abs=0.5)

    def test_metrics_bull_call_spread(self):
        legs = [
            {"option_type": "call", "position": "long",  "strike": 100, "premium": 5},
            {"option_type": "call", "position": "short", "strike": 105, "premium": 2},
        ]
        payoff = self.sm.compute_payoff(legs, self.S_range)
        metrics = self.sm.compute_metrics(legs, self.S_range, payoff)

        # Coût net = 5 - 2 = 3
        assert metrics["cost"] == pytest.approx(3.0, abs=0.01)
        # Gain max plafonné = 5 - 3 = 2
        assert np.isfinite(metrics["max_gain"])
        assert metrics["max_gain"] == pytest.approx(2.0, abs=0.5)

    # ── compute_greeks (agrégation) ─────────────────────────────────────

    def test_straddle_greeks_near_zero_delta(self):
        """Un straddle ATM a un delta proche de 0."""
        om = OptionModels()
        legs = [
            {"option_type": "call", "position": "long", "strike": 100, "premium": 5},
            {"option_type": "put",  "position": "long", "strike": 100, "premium": 5},
        ]
        greeks = self.sm.compute_greeks(legs, S=100, T=0.25, r=0.05,
                                         sigma=0.2, q=0.0, option_models=om)
        assert abs(greeks["delta"]) < 0.15
        assert greeks["gamma"] > 0  # gamma toujours positif pour un long straddle


# ============================================================================
#  SIMULATION_LOGIC
# ============================================================================

class TestSimulationLogic:
    """Tests pour simulation_logic.py."""

    def setup_method(self):
        self.om = OptionModels()
        self.sim = SimulationLogic(self.om)

    def test_run_simulation_dimensions(self):
        vols, prices, matrix, all_p = self.sim.run_simulation(
            K=100, T=0.25, r=0.05, q=0.0,
            vol_min=10, vol_max=30, vol_step=10,
            underlying_min=90, underlying_max=110, underlying_step=10,
        )
        assert len(vols) == 3       # 10, 20, 30
        assert len(prices) == 3     # 90, 100, 110
        assert matrix.shape == (3, 3)
        assert len(all_p) == 9

    def test_run_simulation_prices_positive(self):
        _, _, matrix, _ = self.sim.run_simulation(
            K=100, T=0.25, r=0.05, q=0.0,
            vol_min=20, vol_max=40, vol_step=10,
            underlying_min=80, underlying_max=120, underlying_step=5,
        )
        assert np.all(matrix >= 0)

    def test_run_simulation_empty_range(self):
        vols, prices, matrix, all_p = self.sim.run_simulation(
            K=100, T=0.25, r=0.05, q=0.0,
            vol_min=50, vol_max=40, vol_step=10,  # range inversée
            underlying_min=90, underlying_max=110, underlying_step=10,
        )
        assert len(vols) == 0
        assert len(all_p) == 0

    def test_higher_vol_gives_higher_price_atm(self):
        """À moneyness fixe (ATM), plus de vol → plus de prix."""
        _, _, matrix, _ = self.sim.run_simulation(
            K=100, T=0.25, r=0.05, q=0.0,
            vol_min=10, vol_max=50, vol_step=10,
            underlying_min=100, underlying_max=100, underlying_step=10,
        )
        prices_col = matrix[:, 0]
        for i in range(len(prices_col) - 1):
            assert prices_col[i + 1] >= prices_col[i]


# ============================================================================
#  FORECAST_LOGIC  (inversion IV et repricing à IV prédite)
# ============================================================================

class TestForecastLogic:
    """Tests pour forecast_logic.py (inversion IV, repricing, pas l'inférence TimesFM).

    Depuis la migration React/Eel, process_iv_forecast_results attend un array
    1D directement (le découpage [0] est fait en amont dans run_iv_forecast).
    """

    def setup_method(self):
        self.om = OptionModels()
        self.fl = ForecastLogic(self.om)

    def test_compute_iv_roundtrip_call(self):
        """BSM(sigma) donne un prix, l'inversion doit retrouver sigma."""
        sigma_true = 0.25
        S, K, T_days, r, q = 100, 100, 30, 0.05, 0.01
        T = T_days / 365.0
        price = self.om.black_scholes_price(S, K, T, r, sigma_true, q, "call")

        iv_series = self.fl.compute_iv_from_prices(
            mid_prices=[price],
            underlying_prices=[S],
            dtes=[T_days],
            strike=K, r=r, q=q, option_type="call",
        )
        assert len(iv_series) == 1
        assert iv_series[0] == pytest.approx(sigma_true, abs=1e-3)

    def test_compute_iv_roundtrip_put(self):
        """Même test pour un put."""
        sigma_true = 0.30
        S, K, T_days, r, q = 100, 105, 60, 0.05, 0.0
        T = T_days / 365.0
        price = self.om.black_scholes_price(S, K, T, r, sigma_true, q, "put")

        iv_series = self.fl.compute_iv_from_prices(
            mid_prices=[price],
            underlying_prices=[S],
            dtes=[T_days],
            strike=K, r=r, q=q, option_type="put",
        )
        assert iv_series[0] == pytest.approx(sigma_true, abs=1e-3)

    def test_compute_iv_multiple_points(self):
        """Inversion sur une série de 10 points avec des IV variables."""
        S_vals = [100, 101, 99, 102, 98, 103, 97, 104, 100, 101]
        sigma_vals = [0.20, 0.22, 0.21, 0.23, 0.19, 0.24, 0.18, 0.25, 0.20, 0.22]
        K, r, q = 100, 0.05, 0.0
        dte_vals = [30, 29, 28, 27, 26, 25, 24, 23, 22, 21]

        prices = []
        for i in range(10):
            T = dte_vals[i] / 365.0
            p = self.om.black_scholes_price(S_vals[i], K, T, r, sigma_vals[i], q, "call")
            prices.append(p)

        iv_series = self.fl.compute_iv_from_prices(
            mid_prices=prices,
            underlying_prices=S_vals,
            dtes=dte_vals,
            strike=K, r=r, q=q, option_type="call",
        )
        assert len(iv_series) == 10
        for i in range(10):
            assert iv_series[i] == pytest.approx(sigma_vals[i], abs=1e-3)

    def test_compute_iv_handles_invalid_prices(self):
        """Les prix invalides (négatifs, nuls) sont interpolés."""
        K, r, q = 100, 0.05, 0.0
        valid_price = self.om.black_scholes_price(100, K, 30/365, r, 0.25, q, "call")

        iv_series = self.fl.compute_iv_from_prices(
            mid_prices=[valid_price, -1.0, 0.0, valid_price],
            underlying_prices=[100, 100, 100, 100],
            dtes=[30, 29, 28, 27],
            strike=K, r=r, q=q, option_type="call",
        )
        assert len(iv_series) == 4
        # les NaN doivent avoir été interpolés
        assert not np.any(np.isnan(iv_series))

    def test_process_iv_forecast_results_shapes(self):
        """Vérifie les dimensions des séries retournées.

        process_iv_forecast_results attend désormais un array 1D (horizon,)
        directement — le découpage [0] est fait dans run_iv_forecast.
        """
        horizon = 10
        iv_history = np.full(20, 0.25)
        # Array 1D : la méthode fait .flatten() en interne
        iv_point_forecast = np.full(horizon, 0.24)

        iv_fc, opt_prices, deltas, iv_hist, hist_opt, hist_d, x_hist = \
            self.fl.process_iv_forecast_results(
                iv_point_forecast, iv_history, horizon,
                K=100, T_total=0.25, S=100, r=0.05, q=0.0,
                option_type="call",
            )

        assert len(iv_fc) == horizon
        assert len(opt_prices) == horizon
        assert len(deltas) == horizon
        assert len(iv_hist) == 20   # min(30, 20) = 20
        assert len(hist_opt) == 20
        assert len(hist_d) == 20

    def test_process_iv_forecast_option_prices_positive(self):
        """Les prix d'options repricés doivent être strictement positifs."""
        horizon = 5
        iv_history = np.full(30, 0.25)
        iv_point_forecast = np.full(horizon, 0.25)  # 1D

        _, opt_prices, _, _, _, _, _ = self.fl.process_iv_forecast_results(
            iv_point_forecast, iv_history, horizon,
            K=100, T_total=0.5, S=100, r=0.05, q=0.0,
            option_type="call",
        )
        assert np.all(opt_prices > 0)

    def test_process_iv_forecast_delta_range(self):
        """Les deltas d'un call doivent être dans [0, 1]."""
        horizon = 10
        iv_history = np.full(30, 0.25)
        iv_point_forecast = np.linspace(0.15, 0.35, horizon)  # 1D

        _, _, deltas, _, _, _, _ = self.fl.process_iv_forecast_results(
            iv_point_forecast, iv_history, horizon,
            K=100, T_total=0.5, S=100, r=0.05, q=0.0,
            option_type="call",
        )
        assert np.all(deltas >= 0)
        assert np.all(deltas <= 1)

    def test_process_iv_forecast_negative_iv_clamped(self):
        """Les IV négatives prédites doivent être bornées à 0.01 minimum."""
        horizon = 3
        iv_history = np.full(10, 0.20)
        # IV négatives simulant un artefact du modèle — array 1D
        iv_point_forecast = np.array([-0.05, 0.0, 0.10])

        iv_fc, opt_prices, _, _, _, _, _ = self.fl.process_iv_forecast_results(
            iv_point_forecast, iv_history, horizon,
            K=100, T_total=0.5, S=100, r=0.05, q=0.0,
            option_type="call",
        )
        # les prix doivent quand même être positifs grâce au clamping
        assert np.all(opt_prices > 0)


# ============================================================================
#  VOLATILITY_SMILE_LOGIC  (IV inversion)
# ============================================================================

class TestVolatilitySmileLogic:
    """Tests pour volatility_smile_logic.py."""

    def setup_method(self):
        self.vsl = VolatilitySmileLogic()

    def test_calculate_iv_roundtrip(self):
        """BSM(sigma) → price → IV(price) doit retrouver sigma."""
        om = OptionModels()
        sigma_true = 0.25
        S, K, T, r, q = 100, 100, 0.5, 0.05, 0.01
        price = om.black_scholes_price(S, K, T, r, sigma_true, q, "call")
        iv = self.vsl.calculate_iv_from_price(price, S, K, T, r, q, "call")
        assert iv == pytest.approx(sigma_true, abs=1e-4)

    def test_iv_returns_none_for_negative_price(self):
        assert self.vsl.calculate_iv_from_price(-1, 100, 100, 0.5, 0.05, 0, "call") is None

    def test_iv_returns_none_for_zero_T(self):
        assert self.vsl.calculate_iv_from_price(5, 100, 100, 0, 0.05, 0, "call") is None

    def test_iv_put_roundtrip(self):
        om = OptionModels()
        sigma_true = 0.30
        S, K, T, r, q = 100, 105, 0.25, 0.05, 0.0
        price = om.black_scholes_price(S, K, T, r, sigma_true, q, "put")
        iv = self.vsl.calculate_iv_from_price(price, S, K, T, r, q, "put")
        assert iv == pytest.approx(sigma_true, abs=1e-4)

    def test_process_smile_data_valid(self):
        """Vérifie le pipeline complet avec des données synthétiques."""
        om = OptionModels()
        S, T, r, q = 100, 0.25, 0.05, 0.0

        # Créer une chaîne d'options synthétique
        strikes_put = np.array([85, 90, 95])
        strikes_call = np.array([100, 105, 110])

        def _make_chain(strikes, opt_type, sigma=0.25):
            prices = [om.black_scholes_price(S, k, T, r, sigma, q, opt_type)
                      for k in strikes]
            return pd.DataFrame({
                "strike": strikes,
                "bid": [p * 0.95 for p in prices],
                "ask": [p * 1.05 for p in prices],
                "lastPrice": prices,
                "impliedVolatility": [sigma] * len(strikes),
            })

        chain = MagicMock()
        chain.calls = _make_chain(strikes_call, "call")
        chain.puts = _make_chain(strikes_put, "put")

        s_interp, iv_interp, df = self.vsl.process_smile_data(chain, S, T, r, q)
        assert s_interp is not None
        assert iv_interp is not None
        assert len(s_interp) == 200
        assert len(df) >= 4  # au moins quelques points valides

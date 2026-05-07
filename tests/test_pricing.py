# Commande : pytest tests/test_pricing.py -v
# Golden values : DerivaGem (John Hull)

import os
import sys
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from option_models import OptionModels
from exotic_options_models import (
    price_barrier_analytical,
    price_digital_analytical,
    price_asian_mc,
    price_lookback_mc,
)

# ═══════════════════════════════════════════════════════════════════════════
# GOLDEN VALUES — DerivaGem (John Hull)
# ═══════════════════════════════════════════════════════════════════════════

GOLDEN = {
    # --- BSM Européen ---
    # Inputs : S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.00
    "bsm_call_atm":            10.4505835721856,
    "bsm_put_atm":             5.57352602225696,
    # Inputs : S=100, K=110, T=0.5, r=0.05, sigma=0.25, q=0.00
    "bsm_call_otm":            4.22578239296008,
    "bsm_put_itm":             11.5098727160767,
    # Inputs : S=100, K=90, T=0.25, r=0.05, sigma=0.30, q=0.00
    "bsm_call_itm":            12.8581969564938,
    "bsm_put_otm":             1.74019900094315,

    # --- Greeks BSM (mêmes inputs que bsm_call_atm) ---
    "delta_call_atm":          0.636830651175619,
    "delta_put_atm":           -0.363169348824381,
    "gamma_atm":               0.0187620173458469,
    "vega_atm":                0.375240346916938,    # DerivaGem: per 1% vol
    "theta_call_atm":          -0.0175726782094197,  # par jour
    "rho_call_atm":            0.532324815453763,    # per 1% rate

    # --- CRR Américain ---
    # Inputs : S=100, K=100, T=1.0, r=0.05, q=0.00, sigma=0.20, N=200
    "crr_call_american_atm":   10.4405912598599,
    "crr_put_american_atm":    6.0863827499161,
    # Inputs : S=100, K=110, T=0.5, r=0.05, q=0.00, sigma=0.25, N=200
    "crr_put_american_itm":    12.1205570376456,

    # --- Barrières analytiques (Rubinstein-Reiner) ---
    # Inputs communs : S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.00
    "barrier_down_out_call":   8.66547165824565,   # H=90
    "barrier_down_in_call":    1.7851119139399,    # H=90
    "barrier_up_out_call":     0.118614052789107,  # H=110
    "barrier_up_in_call":      10.3319695193964,   # H=110
    "barrier_down_out_put":    0.151220376439861,  # H=90
    "barrier_down_in_put":     5.4223056458171,    # H=90
    "barrier_up_out_put":      4.19819381092549,   # H=110
    "barrier_up_in_put":       1.37533221133146,   # H=110

    # --- Digitale cash or nothing ---
    # Inputs : S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.00, payoff=1.0
    "digital_call_atm":        0.532324815453763,
    "digital_put_atm":         0.418904609046951,

    # --- Monte Carlo (tolérance large : abs=0.10) ---
    # Asiatique arithmétique : S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.00
    "asian_call_arithmetic":   5.78283833805171,
    "asian_put_arithmetic":    3.36462978955099,
    # Lookback flottant MC : S=100, T=1.0, r=0.05, sigma=0.20, q=0.00
    "lookback_call":           19.1676252573323,
    "lookback_put":            12.3397446874323,
}

# ═══════════════════════════════════════════════════════════════════════════
# Tolérances
# ═══════════════════════════════════════════════════════════════════════════
TOL         = 0.01   # 1 centime
TOL_MC      = 0.10   # Monte Carlo
TOL_MC_LBK  = 3.00   # NB: DerivaGem utilise Goldman-Sosin-Gatto (monitoring continu),
                     # écart structurel attendu avec notre MC discret (252 steps). Tolérance élargie
TOL_PARITY  = 1e-4   # propriétés mathématiques exactes
TOL_GREEK   = 1e-6   # égalité gamma_call == gamma_put, etc.

models = OptionModels()


def _pct(result: float, expected: float) -> str:
    """Pourcentage de différence relative, safe pour expected ~ 0."""
    if abs(expected) < 1e-12:
        return "N/A"
    return f"{abs(result - expected) / abs(expected) * 100:.4f}%"


# ═══════════════════════════════════════════════════════════════════════════
# BSM
# ═══════════════════════════════════════════════════════════════════════════
class TestBSM:

    def test_bsm_call_atm(self):
        expected = GOLDEN["bsm_call_atm"]
        result = models.black_scholes_price(100, 100, 1.0, 0.05, 0.20, 0.00, "call")
        assert result == pytest.approx(expected, abs=TOL), (
            f"BSM call ATM: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_bsm_put_atm(self):
        expected = GOLDEN["bsm_put_atm"]
        result = models.black_scholes_price(100, 100, 1.0, 0.05, 0.20, 0.00, "put")
        assert result == pytest.approx(expected, abs=TOL), (
            f"BSM put ATM: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_bsm_call_otm(self):
        expected = GOLDEN["bsm_call_otm"]
        result = models.black_scholes_price(100, 110, 0.5, 0.05, 0.25, 0.00, "call")
        assert result == pytest.approx(expected, abs=TOL), (
            f"BSM call OTM: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_bsm_put_itm(self):
        expected = GOLDEN["bsm_put_itm"]
        result = models.black_scholes_price(100, 110, 0.5, 0.05, 0.25, 0.00, "put")
        assert result == pytest.approx(expected, abs=TOL), (
            f"BSM put ITM: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_bsm_call_itm(self):
        expected = GOLDEN["bsm_call_itm"]
        result = models.black_scholes_price(100, 90, 0.25, 0.05, 0.30, 0.00, "call")
        assert result == pytest.approx(expected, abs=TOL), (
            f"BSM call ITM: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_bsm_put_otm(self):
        expected = GOLDEN["bsm_put_otm"]
        result = models.black_scholes_price(100, 90, 0.25, 0.05, 0.30, 0.00, "put")
        assert result == pytest.approx(expected, abs=TOL), (
            f"BSM put OTM: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_call_put_parity(self):
        """C - P == S·exp(-qT) - K·exp(-rT)  (propriété mathématique)."""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.00
        C = models.black_scholes_price(S, K, T, r, sigma, q, "call")
        P = models.black_scholes_price(S, K, T, r, sigma, q, "put")
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert (C - P) == pytest.approx(parity, abs=TOL_PARITY), (
            f"Put-call parity: C-P={C - P:.6f}, S·e^(-qT)-K·e^(-rT)={parity:.6f}, "
            f"pct={_pct(C - P, parity)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GREEKS
# ═══════════════════════════════════════════════════════════════════════════
class TestGreeks:

    # Inputs communs : S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.00
    def _greeks(self, option_type):
        return models.calculate_greeks(100, 100, 1.0, 0.05, 0.20, 0.00, option_type)

    def test_delta_call_atm(self):
        expected = GOLDEN["delta_call_atm"]
        result = self._greeks("call")["delta"]
        assert result == pytest.approx(expected, abs=TOL_GREEK), (
            f"Delta call ATM: expected={expected:.6f}, got={result:.6f}, "
            f"diff={abs(result - expected):.6f}, pct={_pct(result, expected)}"
        )

    def test_delta_put_atm(self):
        expected = GOLDEN["delta_put_atm"]
        result = self._greeks("put")["delta"]
        assert result == pytest.approx(expected, abs=TOL_GREEK), (
            f"Delta put ATM: expected={expected:.6f}, got={result:.6f}, "
            f"diff={abs(result - expected):.6f}, pct={_pct(result, expected)}"
        )

    def test_gamma_atm(self):
        expected = GOLDEN["gamma_atm"]
        gamma_call = self._greeks("call")["gamma"]
        gamma_put = self._greeks("put")["gamma"]
        # Gamma call == Gamma put (propriété mathématique)
        assert gamma_call == pytest.approx(gamma_put, abs=TOL_GREEK), (
            f"Gamma call≠put: call={gamma_call:.8f}, put={gamma_put:.8f}, "
            f"pct={_pct(gamma_call, gamma_put)}"
        )
        assert gamma_call == pytest.approx(expected, abs=TOL_GREEK), (
            f"Gamma ATM: expected={expected:.6f}, got={gamma_call:.6f}, "
            f"diff={abs(gamma_call - expected):.6f}, pct={_pct(gamma_call, expected)}"
        )

    def test_vega_atm(self):
        # NOTE: DerivaGem rapporte le vega par 1% de vol (×0.01).
        # Le code retourne dC/dσ brut. Conversion : code_vega × 0.01 = DerivaGem.
        expected_dg = GOLDEN["vega_atm"]
        expected_code = expected_dg * 100.0  # valeur attendue du code
        vega_call = self._greeks("call")["vega"]
        vega_put = self._greeks("put")["vega"]
        # Vega call == Vega put (propriété mathématique)
        assert vega_call == pytest.approx(vega_put, abs=TOL_GREEK), (
            f"Vega call≠put: call={vega_call:.8f}, put={vega_put:.8f}, "
            f"pct={_pct(vega_call, vega_put)}"
        )
        assert vega_call == pytest.approx(expected_code, abs=TOL_GREEK), (
            f"Vega ATM: expected={expected_code:.4f} (DG×100), "
            f"got={vega_call:.4f}, diff={abs(vega_call - expected_code):.4f}, "
            f"pct={_pct(vega_call, expected_code)}"
        )

    def test_theta_call_atm(self):
        expected = GOLDEN["theta_call_atm"]
        result = self._greeks("call")["theta"]
        assert result == pytest.approx(expected, abs=TOL_GREEK), (
            f"Theta call ATM: expected={expected:.6f}, got={result:.6f}, "
            f"diff={abs(result - expected):.6f}, pct={_pct(result, expected)}"
        )

    def test_rho_call_atm(self):
        expected = GOLDEN["rho_call_atm"]
        result = self._greeks("call")["rho"]
        assert result == pytest.approx(expected, abs=TOL_GREEK), (
            f"Rho call ATM: expected={expected:.6f}, got={result:.6f}, "
            f"diff={abs(result - expected):.6f}, pct={_pct(result, expected)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# CRR (Américain)
# ═══════════════════════════════════════════════════════════════════════════
class TestCRR:

    def test_crr_call_american_atm(self):
        expected = GOLDEN["crr_call_american_atm"]
        result = models.cox_ross_rubinstein_price(100, 100, 1.0, 0.05, 0.00, 0.20, 200, "call")
        assert result == pytest.approx(expected, abs=TOL), (
            f"CRR call ATM: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_crr_put_american_atm(self):
        expected = GOLDEN["crr_put_american_atm"]
        result = models.cox_ross_rubinstein_price(100, 100, 1.0, 0.05, 0.00, 0.20, 200, "put")
        assert result == pytest.approx(expected, abs=TOL), (
            f"CRR put ATM: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_crr_put_american_itm(self):
        expected = GOLDEN["crr_put_american_itm"]
        result = models.cox_ross_rubinstein_price(100, 110, 0.5, 0.05, 0.00, 0.25, 200, "put")
        assert result == pytest.approx(expected, abs=TOL), (
            f"CRR put ITM: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_american_put_ge_european(self):
        """Le put américain vaut toujours ≥ au put européen (early exercise premium ≥ 0)."""
        bsm_put = models.black_scholes_price(100, 100, 1.0, 0.05, 0.20, 0.00, "put")
        crr_put = models.cox_ross_rubinstein_price(100, 100, 1.0, 0.05, 0.00, 0.20, 200, "put")
        premium = crr_put - bsm_put
        assert crr_put >= bsm_put - 1e-6, (
            f"American put ({crr_put:.4f}) < European put ({bsm_put:.4f}), "
            f"premium={premium:.4f}, pct={_pct(crr_put, bsm_put)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# BARRIÈRES
# ═══════════════════════════════════════════════════════════════════════════
class TestBarriers:

    # Inputs communs : S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.00
    _S, _K, _T, _r, _sig, _q = 100, 100, 1.0, 0.05, 0.20, 0.00

    def _barrier(self, barrier, option_type, barrier_type):
        res = price_barrier_analytical(
            self._S, self._K, self._T, self._r, self._sig, self._q,
            barrier, option_type, barrier_type,
        )
        return res.price

    def test_barrier_down_out_call(self):
        expected = GOLDEN["barrier_down_out_call"]
        result = self._barrier(90, "call", "down-and-out")
        assert result == pytest.approx(expected, abs=TOL), (
            f"Down-out call: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_barrier_down_in_call(self):
        expected = GOLDEN["barrier_down_in_call"]
        result = self._barrier(90, "call", "down-and-in")
        assert result == pytest.approx(expected, abs=TOL), (
            f"Down-in call: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_barrier_up_out_call(self):
        expected = GOLDEN["barrier_up_out_call"]
        result = self._barrier(110, "call", "up-and-out")
        assert result == pytest.approx(expected, abs=TOL), (
            f"Up-out call: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_barrier_up_in_call(self):
        expected = GOLDEN["barrier_up_in_call"]
        result = self._barrier(110, "call", "up-and-in")
        assert result == pytest.approx(expected, abs=TOL), (
            f"Up-in call: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_barrier_down_out_put(self):
        expected = GOLDEN["barrier_down_out_put"]
        result = self._barrier(90, "put", "down-and-out")
        assert result == pytest.approx(expected, abs=TOL), (
            f"Down-out put: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_barrier_down_in_put(self):
        expected = GOLDEN["barrier_down_in_put"]
        result = self._barrier(90, "put", "down-and-in")
        assert result == pytest.approx(expected, abs=TOL), (
            f"Down-in put: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_barrier_up_out_put(self):
        expected = GOLDEN["barrier_up_out_put"]
        result = self._barrier(110, "put", "up-and-out")
        assert result == pytest.approx(expected, abs=TOL), (
            f"Up-out put: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_barrier_up_in_put(self):
        expected = GOLDEN["barrier_up_in_put"]
        result = self._barrier(110, "put", "up-and-in")
        assert result == pytest.approx(expected, abs=TOL), (
            f"Up-in put: expected={expected:.4f}, got={result:.4f}, "
            f"diff={abs(result - expected):.4f}, pct={_pct(result, expected)}"
        )

    def test_in_out_parity_call(self):
        """down_in + down_out == vanilla BSM call."""
        di = self._barrier(90, "call", "down-and-in")
        do = self._barrier(90, "call", "down-and-out")
        vanilla = models.black_scholes_price(
            self._S, self._K, self._T, self._r, self._sig, self._q, "call"
        )
        assert (di + do) == pytest.approx(vanilla, abs=TOL), (
            f"In-out parity call: DI+DO={di + do:.4f}, vanilla={vanilla:.4f}, "
            f"pct={_pct(di + do, vanilla)}"
        )

    def test_in_out_parity_put(self):
        """down_in + down_out == vanilla BSM put."""
        di = self._barrier(90, "put", "down-and-in")
        do = self._barrier(90, "put", "down-and-out")
        vanilla = models.black_scholes_price(
            self._S, self._K, self._T, self._r, self._sig, self._q, "put"
        )
        assert (di + do) == pytest.approx(vanilla, abs=TOL), (
            f"In-out parity put: DI+DO={di + do:.4f}, vanilla={vanilla:.4f}, "
            f"pct={_pct(di + do, vanilla)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# DIGITALES
# ═══════════════════════════════════════════════════════════════════════════
class TestDigital:

    _S, _K, _T, _r, _sig, _q = 100, 100, 1.0, 0.05, 0.20, 0.00

    def _digital(self, option_type, payoff_amount=1.0):
        res = price_digital_analytical(
            self._S, self._K, self._T, self._r, self._sig, self._q,
            option_type, payoff_amount=payoff_amount,
        )
        return res.price

    def test_digital_call_atm(self):
        expected = GOLDEN["digital_call_atm"]
        result = self._digital("call")
        assert result == pytest.approx(expected, abs=TOL), (
            f"Digital call: expected={expected:.6f}, got={result:.6f}, "
            f"diff={abs(result - expected):.6f}, pct={_pct(result, expected)}"
        )

    def test_digital_put_atm(self):
        expected = GOLDEN["digital_put_atm"]
        result = self._digital("put")
        assert result == pytest.approx(expected, abs=TOL), (
            f"Digital put: expected={expected:.6f}, got={result:.6f}, "
            f"diff={abs(result - expected):.6f}, pct={_pct(result, expected)}"
        )

    def test_digital_call_put_sum(self):
        """call + put == exp(-rT)  (propriété mathématique)."""
        c = self._digital("call")
        p = self._digital("put")
        expected = np.exp(-self._r * self._T)
        assert (c + p) == pytest.approx(expected, abs=TOL_GREEK), (
            f"Digital C+P={c + p:.8f}, exp(-rT)={expected:.8f}, "
            f"pct={_pct(c + p, expected)}"
        )

    def test_digital_payoff_scaling(self):
        """prix(payoff=2) == 2 × prix(payoff=1)."""
        p1 = self._digital("call", payoff_amount=1.0)
        p2 = self._digital("call", payoff_amount=2.0)
        assert p2 == pytest.approx(2.0 * p1, abs=TOL_GREEK), (
            f"Scaling: 2×p1={2 * p1:.8f}, p2={p2:.8f}, "
            f"pct={_pct(p2, 2.0 * p1)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════
class TestMonteCarlo:

    _S, _K, _T, _r, _sig, _q = 100, 100, 1.0, 0.05, 0.20, 0.00
    _n_sims, _n_steps, _seed = 100_000, 252, 42

    @pytest.mark.slow
    def test_asian_call_arithmetic(self):
        expected = GOLDEN["asian_call_arithmetic"]
        res = price_asian_mc(
            self._S, self._K, self._T, self._r, self._sig, self._q,
            "call", "arithmetic", self._n_sims, self._n_steps,
        )
        assert res.price == pytest.approx(expected, abs=TOL_MC), (
            f"Asian call: expected={expected:.4f}, got={res.price:.4f}, "
            f"diff={abs(res.price - expected):.4f}, pct={_pct(res.price, expected)}"
        )

    @pytest.mark.slow
    def test_asian_put_arithmetic(self):
        expected = GOLDEN["asian_put_arithmetic"]
        res = price_asian_mc(
            self._S, self._K, self._T, self._r, self._sig, self._q,
            "put", "arithmetic", self._n_sims, self._n_steps,
        )
        assert res.price == pytest.approx(expected, abs=TOL_MC), (
            f"Asian put: expected={expected:.4f}, got={res.price:.4f}, "
            f"diff={abs(res.price - expected):.4f}, pct={_pct(res.price, expected)}"
        )

    @pytest.mark.slow
    def test_lookback_call(self):
        expected = GOLDEN["lookback_call"]
        res = price_lookback_mc(
            self._S, self._T, self._r, self._sig, self._q,
            "call", self._n_sims, self._n_steps,
        )
        assert res.price == pytest.approx(expected, abs=TOL_MC_LBK), (
            f"Lookback call: expected={expected:.4f}, got={res.price:.4f}, "
            f"diff={abs(res.price - expected):.4f}, pct={_pct(res.price, expected)}"
        )

    @pytest.mark.slow
    def test_lookback_put(self):
        expected = GOLDEN["lookback_put"]
        res = price_lookback_mc(
            self._S, self._T, self._r, self._sig, self._q,
            "put", self._n_sims, self._n_steps,
        )
        assert res.price == pytest.approx(expected, abs=TOL_MC_LBK), (
            f"Lookback put: expected={expected:.4f}, got={res.price:.4f}, "
            f"diff={abs(res.price - expected):.4f}, pct={_pct(res.price, expected)}"
        )

    @pytest.mark.slow
    def test_asian_lt_vanilla(self):
        """Prix asiatique call < prix BSM call (toujours vrai)."""
        asian = price_asian_mc(
            self._S, self._K, self._T, self._r, self._sig, self._q,
            "call", "arithmetic", self._n_sims, self._n_steps,
        ).price
        vanilla = models.black_scholes_price(
            self._S, self._K, self._T, self._r, self._sig, self._q, "call"
        )
        assert asian < vanilla, (
            f"Asian call ({asian:.4f}) >= Vanilla call ({vanilla:.4f}), "
            f"pct={_pct(asian, vanilla)}"
        )

    @pytest.mark.slow
    def test_lookback_gt_vanilla(self):
        """Prix lookback call > prix BSM call (toujours vrai)."""
        lookback = price_lookback_mc(
            self._S, self._T, self._r, self._sig, self._q,
            "call", self._n_sims, self._n_steps,
        ).price
        vanilla = models.black_scholes_price(
            self._S, self._K, self._T, self._r, self._sig, self._q, "call"
        )
        assert lookback > vanilla, (
            f"Lookback call ({lookback:.4f}) <= Vanilla call ({vanilla:.4f}), "
            f"pct={_pct(lookback, vanilla)}"
        )

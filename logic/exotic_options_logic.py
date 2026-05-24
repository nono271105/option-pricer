# Moteur de valorisation des produits dérivés exotiques par méthodes analytiques et stochastiques

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Data class résultat
# ---------------------------------------------------------------------------

@dataclass
class ExoticResult:
    """Résultat d'un pricing d'option exotique."""
    price: float
    method: str                      # "Analytique" ou "Monte Carlo"
    std_error: Optional[float]       # Seulement pour MC
    price_paths: Optional[np.ndarray]  # Échantillon de trajectoires pour le graphique
    payoffs: Optional[np.ndarray]    # Distribution des payoffs simulés
    ci_95: Optional[Tuple[float, float]] = None # Intervalle de confiance à 95%


# ---------------------------------------------------------------------------
# Helpers communs
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, T: float, r: float,
           sigma: float, q: float = 0.0) -> Tuple[float, float]:
    """Calcule d1 et d2 Black-Scholes."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T et sigma doivent être strictement positifs.")
    log_sk = np.log(S / K)
    d1 = (log_sk + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def _simulate_paths(S: float, T: float, r: float, sigma: float,
                    q: float, n_steps: int, n_sims: int,
                    seed: Optional[int] = None) -> np.ndarray:
    """
    Génère des trajectoires GBM (Geometric Brownian Motion) avec variables antithétiques.

    Returns:
        Array shape (n_sims, n_steps + 1) avec S[0] = S.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    half_sims = n_sims // 2
    Z_half = rng.standard_normal((half_sims, n_steps))
    Z = np.vstack([Z_half, -Z_half])
    
    if n_sims % 2 != 0:
        Z = np.vstack([Z, rng.standard_normal((1, n_steps))])

    log_returns = drift + diffusion * Z
    log_paths = np.concatenate(
        [np.zeros((n_sims, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    return S * np.exp(log_paths)


# ---------------------------------------------------------------------------
# 1. OPTIONS BARRIÈRES
# ---------------------------------------------------------------------------

def price_barrier_analytical(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float, barrier: float, option_type: str, barrier_type: str,
    rebate: float = 0.0
) -> ExoticResult:
    """
    Prix analytique d'une option barrière — Rubinstein & Reiner (1991).

    Implémentation exacte des formules du papier de Kotzé / Financial Chaos Theory.

    Blocs A–F (R = rebate, défaut 0) :

        A = φS e^{-dτ} (H/S)^{2λ}   N(ηy)  − φK e^{-rτ} (H/S)^{2λ-2} N(ηy  − ησ√τ)
        B = R e^{-rτ} [N(ηx1 − ησ√τ) − (H/S)^{2λ-2} N(ηy1 − ησ√τ)]
        C = φS e^{-dτ}               N(φx)  − φK e^{-rτ}               N(φx  − φσ√τ)
        D = φS e^{-dτ}               N(φx1) − φK e^{-rτ}               N(φx1 − φσ√τ)
        E = φS e^{-dτ} (H/S)^{2λ}   N(ηy1) − φK e^{-rτ} (H/S)^{2λ-2} N(ηy1 − ησ√τ)
        F = R [(H/S)^{a+b} N(ηz) + (H/S)^{a-b} N(ηz − 2ηbσ√τ)]

    Variables :
        μ  = r − d − σ²/2
        λ  = 1 + μ/σ²
        a  = μ/σ²
        b  = √(μ² + 2rσ²) / σ²
        x  = [ln(S/K)   + (r−d+σ²/2)τ] / (σ√τ)
        x1 = [ln(S/H)   + (r−d+σ²/2)τ] / (σ√τ)
        y  = [ln(H²/SK) + (r−d+σ²/2)τ] / (σ√τ)
        y1 = [ln(H/S)   + (r−d+σ²/2)τ] / (σ√τ)
        z  = [ln(H/S)   + bσ²τ]         / (σ√τ)

    Table 1 — combinaisons pour chaque type (φ et η selon le type) :
        Down-and-In  Call  K≥H : A+B       | Put  K≥H : D−A+E+B
        Down-and-In  Call  K<H : C−D+E+B  | Put  K<H : C+B
        Up-and-In    Call  K≥H : C+B       | Put  K≥H : C−D+E+B
        Up-and-In    Call  K<H : D−A+E+B  | Put  K<H : A+B
        Down-and-Out Call  K≥H : C−A+F    | Put  K≥H : C−D+A−E+F
        Down-and-Out Call  K<H : D−E+F    | Put  K<H : F
        Up-and-Out   Call  K≥H : F         | Put  K≥H : D−E+F
        Up-and-Out   Call  K<H : C−D+A−E+F| Put  K<H : C−A+F

    Args:
        option_type  : "call" ou "put"
        barrier_type : "up-and-out", "up-and-in", "down-and-out", "down-and-in"
        rebate       : montant versé si l'option est désactivée (défaut 0)
    """
    H   = barrier
    tau = T
    d   = q          # dividend yield noté d dans le papier
    R   = rebate
    sqT = np.sqrt(tau)

    if "up" in barrier_type and H <= S:
        raise ValueError(f"Pour une barrière Up, le niveau H ({H}) doit être strictement supérieur au spot S ({S}).")
    if "down" in barrier_type and H >= S:
        raise ValueError(f"Pour une barrière Down, le niveau H ({H}) doit être strictement inférieur au spot S ({S}).")

    # initialisation des intermédiaires de calcul du modèle de Rubinstein
    mu  = r - d - 0.5 * sigma ** 2
    lam = 1.0 + mu / sigma ** 2
    a   = mu / sigma ** 2
    b   = np.sqrt(mu ** 2 + 2.0 * r * sigma ** 2) / sigma ** 2

    drift = (r - d + 0.5 * sigma ** 2) * tau   # (r−d+σ²/2)τ

    x  = (np.log(S / K)          + drift) / (sigma * sqT)
    x1 = (np.log(S / H)          + drift) / (sigma * sqT)
    y  = (np.log(H ** 2 / (S*K)) + drift) / (sigma * sqT)
    y1 = (np.log(H / S)          + drift) / (sigma * sqT)
    z  = (np.log(H / S) + b * sigma ** 2 * tau) / (sigma * sqT)

    # indicateurs binaires définissant le sens de la barrière et le type de l'option
    phi = 1  if option_type == "call" else -1
    eta = 1  if "down" in barrier_type else -1

    HS_2lam   = (H / S) ** (2 * lam)        # (H/S)^{2λ}
    HS_2lam2  = (H / S) ** (2 * lam - 2)    # (H/S)^{2λ-2}

    N = norm.cdf   # raccourci

    # évaluation des probabilités composées selon les six sous-blocs structurels
    def _A() -> float:
        return (  phi * S * np.exp(-d * tau) * HS_2lam   * N(eta * y)
                - phi * K * np.exp(-r * tau) * HS_2lam2  * N(eta * y  - eta * sigma * sqT))

    def _B() -> float:
        return R * np.exp(-r * tau) * (
              N(eta * x1 - eta * sigma * sqT)
            - HS_2lam2 * N(eta * y1 - eta * sigma * sqT))

    def _C() -> float:
        return (  phi * S * np.exp(-d * tau) * N(phi * x)
                - phi * K * np.exp(-r * tau) * N(phi * x  - phi * sigma * sqT))

    def _D() -> float:
        return (  phi * S * np.exp(-d * tau) * N(phi * x1)
                - phi * K * np.exp(-r * tau) * N(phi * x1 - phi * sigma * sqT))

    def _E() -> float:
        return (  phi * S * np.exp(-d * tau) * HS_2lam   * N(eta * y1)
                - phi * K * np.exp(-r * tau) * HS_2lam2  * N(eta * y1 - eta * sigma * sqT))

    def _F() -> float:
        return R * (  (H / S) ** (a + b) * N(eta * z)
                    + (H / S) ** (a - b) * N(eta * z - 2 * eta * b * sigma * sqT))

    A, B, C, D, E, F = _A(), _B(), _C(), _D(), _E(), _F()

    # agrégation algébrique des composantes selon le profil spécifique du contrat
    if option_type == "call":   # φ = +1
        if barrier_type == "down-and-in":
            price = (A + B)             if K >= H else (C - D + E + B)
        elif barrier_type == "down-and-out":
            price = (C - A + F)         if K >= H else (D - E + F)
        elif barrier_type == "up-and-in":
            price = (C + B)             if K >= H else (D - A + E + B)
        else:   # up-and-out
            price = F                   if K >= H else (C - D + A - E + F)
    else:       # φ = -1  put
        if barrier_type == "down-and-in":
            price = (D - A + E + B)     if K >= H else (C + B)
        elif barrier_type == "down-and-out":
            price = (C - D + A - E + F) if K >= H else F
        elif barrier_type == "up-and-in":
            price = (C - D + E + B)     if K >= H else (A + B)
        else:   # up-and-out
            price = (D - E + F)         if K >= H else (C - A + F)

    price = max(float(price), 0.0)
    return ExoticResult(price=round(price, 6), method="Analytique",
                        std_error=None, price_paths=None, payoffs=None)


def price_barrier_mc(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float, barrier: float, option_type: str, barrier_type: str,
    n_sims: int = 50_000, n_steps: int = 252, seed: Optional[int] = None
) -> ExoticResult:
    """Monte Carlo pour options barrières avec correction Broadie-Glasserman-Kou."""
    if "up" in barrier_type and barrier <= S:
        raise ValueError(f"Pour une barrière Up, le niveau H ({barrier}) doit être strictement supérieur au spot S ({S}).")
    if "down" in barrier_type and barrier >= S:
        raise ValueError(f"Pour une barrière Down, le niveau H ({barrier}) doit être strictement inférieur au spot S ({S}).")

    paths = _simulate_paths(S, T, r, sigma, q, n_steps, n_sims, seed)
    S_T = paths[:, -1]
    phi = 1 if option_type == "call" else -1
    intrinsic = np.maximum(phi * (S_T - K), 0)

    path_max = paths.max(axis=1)
    path_min = paths.min(axis=1)

    is_up = "up" in barrier_type
    is_out = "out" in barrier_type

    beta = 0.5825971579
    if is_up:
        adj_barrier = barrier * np.exp(-beta * sigma * np.sqrt(T / n_steps))
        breached = path_max >= adj_barrier
    else:
        adj_barrier = barrier * np.exp(+beta * sigma * np.sqrt(T / n_steps))
        breached = path_min <= adj_barrier

    if is_out:
        alive = ~breached
    else:
        alive = breached

    payoffs = np.where(alive, intrinsic, 0.0)
    discount = np.exp(-r * T)
    price = discount * payoffs.mean()
    std_err = discount * payoffs.std() / np.sqrt(n_sims)
    ci_95 = (float(price - 1.96 * std_err), float(price + 1.96 * std_err))

    # extraction d'un sous-ensemble représentatif pour l'affichage visuel
    sample_idx = np.random.default_rng(seed).choice(n_sims, size=min(200, n_sims), replace=False)
    return ExoticResult(
        price=round(float(price), 6),
        method="Monte Carlo",
        std_error=round(float(std_err), 6),
        price_paths=paths[sample_idx],
        payoffs=payoffs,
        ci_95=ci_95,
    )


# ---------------------------------------------------------------------------
# 2. OPTIONS ASIATIQUES
# ---------------------------------------------------------------------------

def price_asian_mc(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float, option_type: str, averaging: str = "arithmetic",
    n_sims: int = 50_000, n_steps: int = 252, seed: Optional[int] = None
) -> ExoticResult:
    """Monte Carlo pour options asiatiques."""
    paths = _simulate_paths(S, T, r, sigma, q, n_steps, n_sims, seed)
    phi = 1 if option_type == "call" else -1

    if averaging == "arithmetic":
        avg = paths[:, 1:].mean(axis=1)
    else:
        avg = np.exp(np.log(paths[:, 1:]).mean(axis=1))

    payoffs = np.maximum(phi * (avg - K), 0)
    discount = np.exp(-r * T)
    price = discount * payoffs.mean()
    std_err = discount * payoffs.std() / np.sqrt(n_sims)
    ci_95 = (float(price - 1.96 * std_err), float(price + 1.96 * std_err))

    sample_idx = np.random.default_rng(seed).choice(n_sims, size=min(200, n_sims), replace=False)
    return ExoticResult(
        price=round(float(price), 6),
        method="Monte Carlo",
        std_error=round(float(std_err), 6),
        price_paths=paths[sample_idx],
        payoffs=payoffs,
        ci_95=ci_95,
    )


# ---------------------------------------------------------------------------
# 3. OPTIONS LOOKBACK
# ---------------------------------------------------------------------------

def price_lookback_mc(
    S: float, T: float, r: float, sigma: float,
    q: float, option_type: str,
    n_sims: int = 50_000, n_steps: int = 252, seed: Optional[int] = None
) -> ExoticResult:
    """Monte Carlo pour options lookback à strike flottant."""
    paths = _simulate_paths(S, T, r, sigma, q, n_steps, n_sims, seed)
    S_T = paths[:, -1]

    beta = 0.5825971579
    if option_type == "call":
        S_min = paths.min(axis=1)
        S_min_adj = S_min * np.exp(-beta * sigma * np.sqrt(T / n_steps))
        payoffs = np.maximum(S_T - S_min_adj, 0)
    else:
        S_max = paths.max(axis=1)
        S_max_adj = S_max * np.exp(beta * sigma * np.sqrt(T / n_steps))
        payoffs = np.maximum(S_max_adj - S_T, 0)

    discount = np.exp(-r * T)
    price = discount * payoffs.mean()
    std_err = discount * payoffs.std() / np.sqrt(n_sims)
    ci_95 = (float(price - 1.96 * std_err), float(price + 1.96 * std_err))

    sample_idx = np.random.default_rng(seed).choice(n_sims, size=min(200, n_sims), replace=False)
    return ExoticResult(
        price=round(float(price), 6),
        method="Monte Carlo",
        std_error=round(float(std_err), 6),
        price_paths=paths[sample_idx],
        payoffs=payoffs,
        ci_95=ci_95,
    )


# ---------------------------------------------------------------------------
# 4. OPTIONS DIGITALES / BINAIRES
# ---------------------------------------------------------------------------

def price_digital_analytical(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float, option_type: str, payoff_amount: float = 1.0
) -> ExoticResult:
    """
    Prix analytique d'une option digitale Cash-or-Nothing.

    Payoff : payoff_amount si S_T > K (call) ou S_T < K (put), sinon 0.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        price = payoff_amount * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = payoff_amount * np.exp(-r * T) * norm.cdf(-d2)

    return ExoticResult(price=round(price, 6), method="Analytique",
                        std_error=None, price_paths=None, payoffs=None)


def price_digital_mc(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float, option_type: str, payoff_amount: float = 1.0,
    n_sims: int = 50_000, n_steps: int = 252, seed: Optional[int] = None
) -> ExoticResult:
    """Monte Carlo pour options digitales."""
    paths = _simulate_paths(S, T, r, sigma, q, n_steps, n_sims, seed)
    S_T = paths[:, -1]

    if option_type == "call":
        payoffs = np.where(S_T > K, payoff_amount, 0.0)
    else:
        payoffs = np.where(S_T < K, payoff_amount, 0.0)

    discount = np.exp(-r * T)
    price = discount * payoffs.mean()
    std_err = discount * payoffs.std() / np.sqrt(n_sims)
    ci_95 = (float(price - 1.96 * std_err), float(price + 1.96 * std_err))

    sample_idx = np.random.default_rng(seed).choice(n_sims, size=min(200, n_sims), replace=False)
    return ExoticResult(
        price=round(float(price), 6),
        method="Monte Carlo",
        std_error=round(float(std_err), 6),
        price_paths=paths[sample_idx],
        payoffs=payoffs,
        ci_95=ci_95,
    )
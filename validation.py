"""
validation.py : Validation centralisée des paramètres financiers.

Module pur Python, sans dépendance à Eel.
Réutilisable tel quel par Pydantic (FastAPI) ou tout autre framework.

Chaque fonction retourne None si les paramètres sont valides,
ou lève une ValueError avec un message stable et lisible.

Les limites sont toutes des constantes nommées en majuscules
pour faciliter la revue et l'ajustement futur.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de limites — décision prudente commentée
# ─────────────────────────────────────────────────────────────────────────────

# Sous-jacent / strike : pas de borne supérieure technique, borne inférieure stricte
S_MIN: float = 1e-6          # S > 0 requis par log(S/K)
K_MIN: float = 1e-6          # idem

# Maturité : zéro ou négatif interdit (position déjà à maturité ou expirée)
T_MIN: float = 1e-6          # en années

# Volatilité : borne inf prudente (sigma = 0 ↔ division par zéro dans BSM)
SIGMA_MIN: float = 1e-6
SIGMA_MAX: float = 10.0      # 1 000 % — valeur extrême mais défendable (penny stocks)

# Taux sans risque : peut être légèrement négatif (politique monétaire)
R_MIN: float = -0.20         # −20 % : borne prudente
R_MAX: float = 2.00          # 200 % : hyperinflation extrême

# Dividende : strictement borné [0, 1) — au-delà c'est une erreur de donnée
Q_MIN: float = 0.0
Q_MAX: float = 0.99          # 99 % : valeur absurde mais non infinie

# Monte Carlo
N_SIMS_MIN: int = 1
N_SIMS_MAX: int = 200_000    # décision prudente : limite mémoire/temps

N_STEPS_MIN: int = 1
N_STEPS_MAX: int = 1_000     # décision prudente

# Arbre CRR
N_CRR_MIN: int = 2
N_CRR_MAX: int = 5_000       # déjà validé dans crr_logic.py

# Ensembles d'énumérations valides
VALID_OPTION_TYPES: frozenset = frozenset({"call", "put"})
VALID_POSITIONS: frozenset = frozenset({"long", "short"})
VALID_BARRIER_TYPES: frozenset = frozenset({
    "down-and-out", "down-and-in", "up-and-out", "up-and-in"
})
VALID_AVERAGING: frozenset = frozenset({"arithmetic", "geometric"})
VALID_EXOTIC_TYPES: frozenset = frozenset({
    "barrier", "asian", "lookback", "digital"
})

# Seuil d'écart de strike max acceptable dans get_implied_volatility_and_price
MAX_STRIKE_GAP_RATIO: float = 0.10   # 10 % du strike demandé — décision prudente


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions de validation des scalaires
# ─────────────────────────────────────────────────────────────────────────────

def _check_finite(name: str, value: float) -> None:
    """Vérifie qu'une valeur est un float fini (ni NaN ni ±inf)."""
    if not math.isfinite(value):
        raise ValueError(
            f"VALIDATION_INVALID_VALUE: '{name}' doit être fini, reçu {value!r}.",
            "VALIDATION_INVALID_VALUE",
        )


def validate_S(S: float) -> None:
    """Prix du sous-jacent strictement positif et fini."""
    _check_finite("S", S)
    if S <= S_MIN:
        raise ValueError(
            f"VALIDATION_S_NON_POSITIVE: S={S} doit être > {S_MIN}.",
            "VALIDATION_S_NON_POSITIVE",
        )


def validate_K(K: float) -> None:
    """Strike strictement positif et fini."""
    _check_finite("K", K)
    if K <= K_MIN:
        raise ValueError(
            f"VALIDATION_K_NON_POSITIVE: K={K} doit être > {K_MIN}.",
            "VALIDATION_K_NON_POSITIVE",
        )


def validate_T(T: float) -> None:
    """Maturité strictement positive et finie (en années)."""
    _check_finite("T", T)
    if T <= T_MIN:
        raise ValueError(
            f"VALIDATION_T_NON_POSITIVE: T={T:.6f} an(s) doit être > {T_MIN}.",
            "VALIDATION_T_NON_POSITIVE",
        )


def validate_sigma(sigma: float) -> None:
    """Volatilité dans (0, SIGMA_MAX] et finie."""
    _check_finite("sigma", sigma)
    if sigma <= SIGMA_MIN or sigma > SIGMA_MAX:
        raise ValueError(
            f"VALIDATION_SIGMA_OUT_OF_RANGE: sigma={sigma:.6f} doit être dans "
            f"({SIGMA_MIN}, {SIGMA_MAX}].",
            "VALIDATION_SIGMA_OUT_OF_RANGE",
        )


def validate_r(r: float) -> None:
    """Taux sans risque dans [R_MIN, R_MAX] et fini."""
    _check_finite("r", r)
    if not (R_MIN <= r <= R_MAX):
        raise ValueError(
            f"VALIDATION_R_OUT_OF_RANGE: r={r:.4f} doit être dans [{R_MIN}, {R_MAX}].",
            "VALIDATION_R_OUT_OF_RANGE",
        )


def validate_q(q: float) -> None:
    """Dividende dans [Q_MIN, Q_MAX] et fini."""
    _check_finite("q", q)
    if not (Q_MIN <= q <= Q_MAX):
        raise ValueError(
            f"VALIDATION_Q_OUT_OF_RANGE: q={q:.4f} doit être dans [{Q_MIN}, {Q_MAX}].",
            "VALIDATION_Q_OUT_OF_RANGE",
        )


def validate_option_type(option_type: str) -> None:
    """option_type doit être 'call' ou 'put'."""
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(
            f"VALIDATION_INVALID_OPTION_TYPE: option_type='{option_type}' invalide. "
            f"Valeurs acceptées : {sorted(VALID_OPTION_TYPES)}.",
            "VALIDATION_INVALID_OPTION_TYPE",
        )


def validate_position(position: str) -> None:
    """position doit être 'long' ou 'short'."""
    if position not in VALID_POSITIONS:
        raise ValueError(
            f"VALIDATION_INVALID_POSITION: position='{position}' invalide. "
            f"Valeurs acceptées : {sorted(VALID_POSITIONS)}.",
            "VALIDATION_INVALID_POSITION",
        )


def validate_barrier_type(barrier_type: str) -> None:
    """barrier_type doit être l'une des 4 valeurs OCC standard."""
    if barrier_type not in VALID_BARRIER_TYPES:
        raise ValueError(
            f"VALIDATION_INVALID_BARRIER_TYPE: barrier_type='{barrier_type}' invalide. "
            f"Valeurs acceptées : {sorted(VALID_BARRIER_TYPES)}.",
            "VALIDATION_INVALID_BARRIER_TYPE",
        )


def validate_averaging(averaging: str) -> None:
    """averaging doit être 'arithmetic' ou 'geometric'."""
    if averaging not in VALID_AVERAGING:
        raise ValueError(
            f"VALIDATION_INVALID_AVERAGING: averaging='{averaging}' invalide. "
            f"Valeurs acceptées : {sorted(VALID_AVERAGING)}.",
            "VALIDATION_INVALID_AVERAGING",
        )


def validate_exotic_type(exotic_type: str) -> None:
    """exotic_type doit être l'un des types supportés."""
    if exotic_type not in VALID_EXOTIC_TYPES:
        raise ValueError(
            f"VALIDATION_INVALID_EXOTIC_TYPE: exotic_type='{exotic_type}' invalide. "
            f"Valeurs acceptées : {sorted(VALID_EXOTIC_TYPES)}.",
            "VALIDATION_INVALID_EXOTIC_TYPE",
        )


def validate_n_sims(n_sims: int) -> None:
    """n_sims dans [N_SIMS_MIN, N_SIMS_MAX]."""
    if not isinstance(n_sims, int):
        n_sims = int(n_sims)
    if not (N_SIMS_MIN <= n_sims <= N_SIMS_MAX):
        raise ValueError(
            f"VALIDATION_N_SIMS_OUT_OF_RANGE: n_sims={n_sims} doit être dans "
            f"[{N_SIMS_MIN}, {N_SIMS_MAX}].",
            "VALIDATION_N_SIMS_OUT_OF_RANGE",
        )


def validate_n_steps(n_steps: int) -> None:
    """n_steps dans [N_STEPS_MIN, N_STEPS_MAX]."""
    if not (N_STEPS_MIN <= n_steps <= N_STEPS_MAX):
        raise ValueError(
            f"VALIDATION_N_STEPS_OUT_OF_RANGE: n_steps={n_steps} doit être dans "
            f"[{N_STEPS_MIN}, {N_STEPS_MAX}].",
            "VALIDATION_N_STEPS_OUT_OF_RANGE",
        )


def validate_N_crr(N: int) -> None:
    """N (nombre de pas CRR) dans [N_CRR_MIN, N_CRR_MAX]."""
    if not (N_CRR_MIN <= N <= N_CRR_MAX):
        raise ValueError(
            f"VALIDATION_N_CRR_OUT_OF_RANGE: N={N} doit être dans "
            f"[{N_CRR_MIN}, {N_CRR_MAX}].",
            "VALIDATION_N_CRR_OUT_OF_RANGE",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Validation groupée : paramètres BSM standard
# ─────────────────────────────────────────────────────────────────────────────

def validate_pricing_params(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: Optional[str] = None,
    sigma: Optional[float] = None,
) -> None:
    """
    Valide les paramètres BSM/CRR standards.
    sigma est optionnel (peut être dérivé de l'IV de marché).
    """
    validate_S(S)
    validate_K(K)
    validate_T(T)
    validate_r(r)
    validate_q(q)
    if option_type is not None:
        validate_option_type(option_type)
    if sigma is not None:
        validate_sigma(sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Validation de date
# ─────────────────────────────────────────────────────────────────────────────

def validate_date_str(
    date_str: str,
    field_name: str = "maturity_date",
    allow_past: bool = False,
) -> datetime:
    """
    Parse et valide une date YYYY-MM-DD ou DD/MM/YYYY.

    - Lève ValueError si le format est invalide.
    - Lève ValueError si la date est dans le passé (sauf allow_past=True).
    - Renvoie le datetime parsé.
    """
    if not date_str or not date_str.strip():
        raise ValueError(
            f"VALIDATION_DATE_EMPTY: '{field_name}' ne peut pas être vide.",
            "VALIDATION_DATE_EMPTY",
        )
    dt: Optional[datetime] = None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            break
        except ValueError:
            continue
    if dt is None:
        raise ValueError(
            f"VALIDATION_DATE_INVALID: '{field_name}'='{date_str}' n'est pas "
            f"au format YYYY-MM-DD ou DD/MM/YYYY.",
            "VALIDATION_DATE_INVALID",
        )
    if not allow_past and dt.date() <= date.today():
        raise ValueError(
            f"VALIDATION_DATE_EXPIRED: '{field_name}'='{date_str}' est dans le passé "
            f"ou aujourd'hui. Fournissez une date d'expiration future.",
            "VALIDATION_DATE_EXPIRED",
        )
    return dt


# ─────────────────────────────────────────────────────────────────────────────
# Validation du résultat de pricing
# ─────────────────────────────────────────────────────────────────────────────

def validate_price_result(price: float, label: str = "price") -> float:
    """Vérifie qu'un prix de sortie est fini et non négatif."""
    if not math.isfinite(price):
        raise ValueError(
            f"VALIDATION_PRICE_NON_FINITE: {label}={price!r} n'est pas fini.",
            "VALIDATION_PRICE_NON_FINITE",
        )
    if price < 0:
        raise ValueError(
            f"VALIDATION_PRICE_NEGATIVE: {label}={price:.6f} est négatif.",
            "VALIDATION_PRICE_NEGATIVE",
        )
    return price


# ─────────────────────────────────────────────────────────────────────────────
# Helper de formatage d'erreur pour les réponses API Eel
# ─────────────────────────────────────────────────────────────────────────────

def _validation_error_response(exc: ValueError) -> dict:
    """
    Transforme une ValueError de validation en réponse Eel standardisée.

    Format : {"error": message_lisible, "code": CODE_STABLE}
    Le code est le premier élément de exc.args si c'est un tuple, sinon dérivé du message.
    """
    args = exc.args
    if len(args) >= 2:
        message, code = args[0], args[1]
    else:
        message = args[0] if args else str(exc)
        # Extraire le code stable du début du message (format CODE: ...)
        if ":" in message:
            code = message.split(":")[0].strip()
        else:
            code = "VALIDATION_ERROR"
    return {"error": message.split(":", 1)[-1].strip() if ":" in message else message,
            "code": code}

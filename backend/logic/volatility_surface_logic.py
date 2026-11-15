# Pipeline de construction et d'interpolation de la surface 3D de volatilité implicite

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from data_fetcher import DataFetcher
from logic.bsm_logic import OptionModels
from logic.volatility_resolver import (
    implied_volatility_from_price,
    valid_mid_quote,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Constantes de configuration
# ─────────────────────────────────────────────────────────────────────────
MAX_SURFACE_EXPIRIES = 6            # Plafond d'échéances (contrainte de latence)
NUM_EXPIRATIONS = MAX_SURFACE_EXPIRIES  # Alias conservé pour compatibilité des imports
SURFACE_TIME_BUDGET_SECONDS = 12.0  # Budget de temps global pour le téléchargement
MAX_DAYS_TO_MATURITY = 100
MIN_DAYS_TO_MATURITY = 7
MONEYNESS_MIN = 0.70                # K/S minimum
MONEYNESS_MAX = 1.30                # K/S maximum
STRIKE_GRID_SIZE = 30
MATURITY_GRID_SIZE = 12
INTERPOLATION_METHOD = 'linear'
MIN_POINTS_FOR_INTERPOLATION = 3  # un triangle suffit pour une triangulation 2D
DAYS_IN_YEAR = 365.0
MARKET_TZ = ZoneInfo("America/New_York")


def _market_today():
    """Date de marché (New York) pour éviter la dépendance au fuseau local."""
    return datetime.now(MARKET_TZ).date()


class ImpliedVolatilitySurface:
    """
    Calcule la surface de volatilité implicite (IV Surface) pour un ticker.

    Surface 3D (sortie, pour plotly) :
        X : Strike (K)
        Y : Time to Maturity (T) en jours
        Z : Implied Volatility (sigma)

    Méthodologie :
        - Convention OTM-only par rapport au forward F = S * exp((r - q) T) :
          puts pour K < F, calls pour K >= F.
        - IV calculée par inversion BSM depuis le mid-price (bid + ask) / 2.
        - Filtres de liquidité et de moneyness (K/S), sans extrapolation.
        - Interpolation en (log-moneyness k = ln(K/F), T) sur la variance
          totale w = sigma^2 * T, puis retour à sigma = sqrt(w / T).

    Limites connues :
        - BSM ignore la prime d'exercice anticipé des options américaines
          (biais léger sur les puts OTM d'actions US).
        - r et q sont supposés constants sur toutes les maturités.
    """

    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.option_models = OptionModels()

    # ─────────────────────────────────────────────────────────────────────
    # Calcul de l'IV par inversion BSM
    # ─────────────────────────────────────────────────────────────────────

    def _compute_iv_from_mid(
        self,
        mid_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        option_type: str,
    ) -> Optional[float]:
        """
        Calcule l'IV par inversion du modèle BSM à partir du mid-price.

        Résout BSM(S, K, T, r, sigma, q) = mid_price (méthode de Brent), avec la
        même méthodologie que le smile 2D pour garantir la cohérence avec la
        surface 3D.

        Returns:
            IV en décimal (0.25 pour 25 %) ou None si l'inversion échoue.
        """
        return implied_volatility_from_price(
            mid_price, S, K, T, r, q, option_type, self.option_models
        )

    # ─────────────────────────────────────────────────────────────────────
    # Récupération des chaînes d'options
    # ─────────────────────────────────────────────────────────────────────

    def get_option_chains_multiple_expirations(
        self,
        ticker_symbol: str,
        max_expiries: int = MAX_SURFACE_EXPIRIES,
        time_budget: float = SURFACE_TIME_BUDGET_SECONDS,
    ) -> Tuple[Dict[str, object], Optional[str]]:
        """
        Récupère les chaînes d'options pour plusieurs échéances (max_expiries).

        Un budget de temps global est imposé : en cas de dépassement, les chaînes
        déjà reçues sont renvoyées avec un avertissement. Les téléchargements en
        cours ne sont pas attendus (les threads ne peuvent pas être interrompus,
        mais ils ne bloquent plus l'appelant).

        Returns:
            (option_chains, warning | None)
        """
        try:
            expirations = self.data_fetcher.get_available_expiries(ticker_symbol)
            if not expirations:
                logger.warning("Aucune date d'expiration trouvée pour %s", ticker_symbol)
                return {}, None

            today = _market_today()
            valid_expiries = []
            for exp_str in expirations:
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                except ValueError:
                    continue
                days = (exp_date - today).days
                if MIN_DAYS_TO_MATURITY <= days <= MAX_DAYS_TO_MATURITY:
                    valid_expiries.append(exp_str)

            valid_expiries = sorted(valid_expiries)[:max_expiries]
            if not valid_expiries:
                return {}, None

            option_chains: Dict[str, object] = {}
            timed_out = False

            executor = ThreadPoolExecutor(max_workers=min(len(valid_expiries), 4))
            try:
                future_to_exp = {
                    executor.submit(
                        self.data_fetcher.get_option_chain_for_expiry, ticker_symbol, exp
                    ): exp
                    for exp in valid_expiries
                }
                try:
                    # Le timeout global est appliqué ici : as_completed lève
                    # TimeoutError si le budget est dépassé.
                    for fut in as_completed(future_to_exp, timeout=time_budget):
                        exp = future_to_exp[fut]
                        try:
                            chain = fut.result()
                            if chain is not None:
                                option_chains[exp] = chain
                        except Exception as e:
                            timed_out = True
                            logger.warning("Erreur pour l'échéance %s : %s", exp, e)
                except FuturesTimeout:
                    timed_out = True
                    logger.warning(
                        "Budget de %.0fs dépassé : %d/%d échéances récupérées",
                        time_budget, len(option_chains), len(valid_expiries),
                    )
            finally:
                # Ne pas attendre les threads en retard, annuler ceux non démarrés.
                executor.shutdown(wait=False, cancel_futures=True)

            warning = None
            if timed_out or len(option_chains) < len(valid_expiries):
                warning = (
                    f"Surface IV partielle : {len(option_chains)}/{len(valid_expiries)} "
                    f"échéances récupérées dans le budget imparti de {time_budget:.0f}s."
                )

            return option_chains, warning

        except Exception as e:
            logger.warning("Erreur lors de la récupération des chaînes d'options: %s", e)
            return {}, None

    # ─────────────────────────────────────────────────────────────────────
    # Extraction des données IV (pipeline complet)
    # ─────────────────────────────────────────────────────────────────────

    def extract_iv_surface_data(
        self,
        ticker_symbol: str,
        current_price: Optional[float] = None,
        current_rate: float = 0.05,
        current_dividend: float = 0.0,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Extrait les données pour la surface de volatilité implicite.

        Pipeline de filtrage :
          1. Convention OTM-only vs forward : puts pour K < F, calls pour K >= F
          2. Filtre de moneyness (MONEYNESS_MIN <= K/S <= MONEYNESS_MAX)
          3. Filtres de liquidité (bid/ask valides, via valid_mid_quote)
          4. IV par inversion BSM depuis le mid-price avec r et q

        Returns:
            (DataFrame, warning). DataFrame avec colonnes :
            ['Strike', 'Days_to_Maturity', 'IV', 'Option_Type', 'Mid_Price',
             'Forward', 'Log_Moneyness', 'Total_Variance']
            ou (None, warning) si aucune donnée exploitable.
        """
        if current_price is None:
            current_price = self.data_fetcher.get_live_price(ticker_symbol)
            if current_price is None:
                logger.warning("Impossible de récupérer le prix pour %s", ticker_symbol)
                return None, None

        logger.info("Prix actuel de %s: $%.2f", ticker_symbol, current_price)
        logger.info("Paramètres: r=%.4f, q=%.4f", current_rate, current_dividend)

        K_min = current_price * MONEYNESS_MIN
        K_max = current_price * MONEYNESS_MAX
        logger.info(
            "Filtre moneyness: K ∈ [%.1f, %.1f] (%.0f%% : %.0f%% du spot)",
            K_min, K_max, MONEYNESS_MIN * 100, MONEYNESS_MAX * 100,
        )

        option_chains, warning = self.get_option_chains_multiple_expirations(ticker_symbol)
        if not option_chains:
            logger.warning("Aucune chaîne d'options récupérée")
            return None, warning

        surface_data = []
        today = _market_today()

        # Les compteurs se somment : total = otm + moneyness + liquidité + iv_failed + accepté
        stats = {
            'total': 0,
            'expired': 0,
            'otm_filtered': 0,
            'moneyness_filtered': 0,
            'liquidity_filtered': 0,
            'iv_failed': 0,
            'accepted': 0,
        }

        for exp_date_str, opt_chain in option_chains.items():
            try:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                days_to_maturity = (exp_date - today).days

                if days_to_maturity < MIN_DAYS_TO_MATURITY:
                    stats['expired'] += 1
                    continue

                T = days_to_maturity / DAYS_IN_YEAR
                forward = current_price * np.exp((current_rate - current_dividend) * T)

                otm_segments = [
                    ('put', opt_chain.puts, lambda s: s < forward),
                    ('call', opt_chain.calls, lambda s: s >= forward),
                ]

                for option_type, data, is_otm in otm_segments:
                    if data is None or data.empty:
                        continue

                    strikes = data['strike']
                    otm_mask = is_otm(strikes)
                    money_mask = (strikes >= K_min) & (strikes <= K_max)

                    stats['total'] += len(data)
                    stats['otm_filtered'] += int((~otm_mask).sum())
                    stats['moneyness_filtered'] += int((otm_mask & ~money_mask).sum())

                    candidates = data[otm_mask & money_mask]

                    for _, row in candidates.iterrows():
                        K = float(row['strike'])

                        mid = valid_mid_quote(row, require_liquidity=True)
                        if mid is None:
                            stats['liquidity_filtered'] += 1
                            continue

                        iv = self._compute_iv_from_mid(
                            mid, current_price, K, T,
                            current_rate, current_dividend, option_type,
                        )
                        if iv is None or not np.isfinite(iv) or iv <= 0:
                            stats['iv_failed'] += 1
                            continue

                        stats['accepted'] += 1
                        surface_data.append({
                            'Strike': K,
                            'Days_to_Maturity': days_to_maturity,
                            'IV': iv,
                            'Option_Type': option_type,
                            'Mid_Price': mid,
                            'Forward': forward,
                            'Log_Moneyness': float(np.log(K / forward)),
                            'Total_Variance': iv ** 2 * T,
                        })

            except Exception as e:
                logger.warning("Erreur lors du traitement de %s: %s", exp_date_str, e)
                continue

        logger.info(
            "Pipeline IV : Total: %d | Expiré: %d | OTM: -%d | Moneyness: -%d | "
            "Liquidité: -%d | IV échouée: -%d | Accepté: %d",
            stats['total'], stats['expired'], stats['otm_filtered'],
            stats['moneyness_filtered'], stats['liquidity_filtered'],
            stats['iv_failed'], stats['accepted'],
        )

        if not surface_data:
            logger.warning("Aucune donnée IV valide extraite après filtrage")
            return None, warning

        df = pd.DataFrame(surface_data)
        logger.info("Données extraites: %s points", len(df))
        logger.info("  Strikes: %.2f : %.2f", df['Strike'].min(), df['Strike'].max())
        logger.info("  Maturité: %s : %s jours",
                    df['Days_to_Maturity'].min(), df['Days_to_Maturity'].max())
        logger.info("  IV: %.4f : %.4f (%.1f%% : %.1f%%)",
                    df['IV'].min(), df['IV'].max(),
                    df['IV'].min() * 100, df['IV'].max() * 100)

        return df, warning

    # ─────────────────────────────────────────────────────────────────────
    # Interpolation de la surface
    # ─────────────────────────────────────────────────────────────────────

    def interpolate_surface(
        self,
        surface_data: pd.DataFrame,
        spot: Optional[float] = None,
        rate: float = 0.05,
        dividend: float = 0.0,
        strike_grid_size: int = STRIKE_GRID_SIZE,
        maturity_grid_size: int = MATURITY_GRID_SIZE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpole la surface sur une grille régulière (Strike x Maturité).

        L'interpolation est faite dans l'espace (k, T), avec k = ln(K/F(T)), sur
        la variance totale w = sigma^2 * T, avec axes normalisés pour une
        triangulation de Delaunay bien conditionnée. Les points de grille hors
        de l'enveloppe convexe des cotations restent NaN (pas d'extrapolation).

        Args:
            surface_data: DataFrame issu de extract_iv_surface_data
            spot, rate, dividend: paramètres servant à calculer F(T) sur la grille
            strike_grid_size, maturity_grid_size: résolution de la grille

        Returns:
            (X_mesh, Y_mesh, Z_mesh) : strikes, jours, IV (pour plotly)
        """
        if len(surface_data) < MIN_POINTS_FOR_INTERPOLATION:
            raise ValueError(
                f"Pas assez de points pour interpoler ({len(surface_data)} "
                f"< {MIN_POINTS_FOR_INTERPOLATION})"
            )
        if surface_data['Days_to_Maturity'].nunique() < 2:
            raise ValueError("Au moins deux échéances sont nécessaires pour interpoler")

        # Spot de référence : à défaut, médiane des strikes (proxy ATM)
        if spot is None:
            spot = float(surface_data['Strike'].median())

        K_obs = surface_data['Strike'].to_numpy(dtype=float)
        T_obs = surface_data['Days_to_Maturity'].to_numpy(dtype=float) / DAYS_IN_YEAR
        iv_obs = surface_data['IV'].to_numpy(dtype=float)

        # Même transformation que pour la grille : k = ln(K / F(T)), w = sigma^2 * T
        F_obs = spot * np.exp((rate - dividend) * T_obs)
        k_obs = np.log(K_obs / F_obs)
        w_obs = iv_obs ** 2 * T_obs

        # Grille de sortie en (Strike, jours)
        strike_min, strike_max = surface_data['Strike'].min(), surface_data['Strike'].max()
        day_min, day_max = (
            surface_data['Days_to_Maturity'].min(),
            surface_data['Days_to_Maturity'].max(),
        )
        X_grid = np.linspace(strike_min, strike_max, strike_grid_size)
        Y_grid = np.linspace(day_min, day_max, maturity_grid_size)
        X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)

        # Conversion de la grille en (k, T) via le forward de chaque maturité
        T_mesh = Y_mesh / DAYS_IN_YEAR
        F_mesh = spot * np.exp((rate - dividend) * T_mesh)
        k_mesh = np.log(X_mesh / F_mesh)

        # Normalisation des axes (échelles très différentes entre k et T)
        k_range = max(np.ptp(k_obs), 1e-12)
        T_range = max(np.ptp(T_obs), 1e-12)
        k0, T0 = k_obs.min(), T_obs.min()

        points = np.column_stack(((k_obs - k0) / k_range, (T_obs - T0) / T_range))
        targets = (
            (k_mesh - k0) / k_range,
            (T_mesh - T0) / T_range,
        )

        w_mesh = griddata(points, w_obs, targets, method=INTERPOLATION_METHOD)

        # Variance totale strictement positive requise
        w_mesh = np.where(w_mesh > 0, w_mesh, np.nan)

        with np.errstate(invalid='ignore', divide='ignore'):
            Z_mesh = np.sqrt(w_mesh / T_mesh)

        return X_mesh, Y_mesh, Z_mesh

    # ─────────────────────────────────────────────────────────────────────
    # Point d'entrée principal
    # ─────────────────────────────────────────────────────────────────────

    def get_surface_for_ticker(
        self,
        ticker_symbol: str,
        current_price: Optional[float] = None,
        current_rate: float = 0.05,
        current_dividend: float = 0.0,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Tuple], Optional[str]]:
        """
        Récupère et interpole la surface IV pour un ticker.

        Pipeline : sélection OTM (vs forward), filtres liquidité/moneyness,
        inversion BSM, puis interpolation linéaire de la variance totale en
        (log-moneyness, maturité), sans extrapolation.

        Returns:
            (raw_data_df, (X_grid, Y_grid, Z_grid), warning)
            Les éléments valent None si l'étape correspondante échoue.
        """
        logger.info("Extraction de la surface IV pour %s...", ticker_symbol)

        if current_price is None:
            current_price = self.data_fetcher.get_live_price(ticker_symbol)
            if current_price is None:
                logger.warning("Impossible de récupérer le prix pour %s", ticker_symbol)
                return None, None, None

        surface_data, warning = self.extract_iv_surface_data(
            ticker_symbol, current_price, current_rate, current_dividend,
        )
        if surface_data is None or surface_data.empty:
            return None, None, warning

        logger.info("Interpolation de la surface (%d points)...", len(surface_data))
        try:
            X_grid, Y_grid, Z_grid = self.interpolate_surface(
                surface_data,
                spot=current_price,
                rate=current_rate,
                dividend=current_dividend,
            )
            logger.info("Surface interpolée avec succès")
            return surface_data, (X_grid, Y_grid, Z_grid), warning
        except Exception as e:
            logger.warning("Erreur lors de l'interpolation: %s", e)
            return surface_data, None, warning
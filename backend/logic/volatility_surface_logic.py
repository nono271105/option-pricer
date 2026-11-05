# Pipeline de construction et d'interpolation de la surface 3D de volatilité implicite

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from datetime import datetime
from scipy.interpolate import griddata

from data_fetcher import DataFetcher
from logic.bsm_logic import OptionModels
from logic.volatility_resolver import (
    implied_volatility_from_price,
    valid_mid_quote,
)

logger = logging.getLogger(__name__)

# constantes de configuration du modèle d'extraction
MAX_SURFACE_EXPIRIES = 6  # Plafonné à 6 échéances pour respecter les contraintes de latence
SURFACE_TIME_BUDGET_SECONDS = 12.0  # Budget de temps global pour le téléchargement des chaînes
NUM_EXPIRATIONS = MAX_SURFACE_EXPIRIES
MAX_DAYS_TO_MATURITY = 100
MIN_DAYS_TO_MATURITY = 7
MONEYNESS_MIN = 0.70
MONEYNESS_MAX = 1.30
STRIKE_GRID_SIZE = 30
MATURITY_GRID_SIZE = 12
INTERPOLATION_METHOD = 'linear'


class ImpliedVolatilitySurface:
    """
    Calcule la surface de volatilité implicite (IV Surface) pour un ticker.
    
    Surface 3D:
        X-axis: Strike prices (K)
        Y-axis: Time to Maturity (T) en jours
        Z-axis: Implied Volatility (σ)

    Méthodologie :
        - Convention OTM-only : puts pour K < S, calls pour K ≥ S
        - IV calculée par inversion BSM depuis le mid-price (bid+ask)/2
        -         Filtres de liquidité et de moneyness prudents, sans extrapolation
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

        Utilise l'algorithme de Brent (brentq) pour résoudre :
            BSM(S, K, T, r, σ, q) = mid_price

        Même méthodologie que volatility_smile_logic.calculate_iv_from_price
        pour garantir la cohérence entre le smile 2D et la surface 3D.
        
        Args:
            mid_price: Prix mid (bid+ask)/2
            S: Prix spot
            K: Strike
            T: Temps à maturité (années)
            r: Taux sans risque
            q: Dividend yield
            option_type: 'call' ou 'put'
            
        Returns:
            IV en décimal (ex: 0.25 pour 25%) ou None si impossible
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
        Récupère les chaînes d'options pour plusieurs dates d'expiration (plafonné à max_expiries).
        Impose un budget de temps global : en cas de dépassement, renvoie les chaînes partielles et un avertissement.
        
        Args:
            ticker_symbol: Symbole du titre (ex: 'AAPL')
            max_expiries: Nombre maximum d'échéances à télécharger (défaut 6)
            time_budget: Temps maximum en secondes
            
        Returns:
            Tuple: (option_chains, warning_str|None)
        """
        try:
            expirations = self.data_fetcher.get_available_expiries(ticker_symbol)
            if not expirations:
                logger.warning("Aucune date d'expiration trouvée pour %s", ticker_symbol)
                return {}, None

            today = datetime.now().date()
            valid_expiries = []
            for exp_str in expirations:
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                    days = (exp_date - today).days
                    if MIN_DAYS_TO_MATURITY <= days <= MAX_DAYS_TO_MATURITY:
                        valid_expiries.append(exp_str)
                except ValueError:
                    continue

            valid_expiries = valid_expiries[:max_expiries]
            if not valid_expiries:
                return {}, None

            start_time = time.time()
            option_chains = {}
            timed_out = False

            with ThreadPoolExecutor(max_workers=min(len(valid_expiries), 4)) as executor:
                future_to_exp = {
                    executor.submit(self.data_fetcher.get_option_chain_for_expiry, ticker_symbol, exp): exp
                    for exp in valid_expiries
                }
                for fut in as_completed(future_to_exp):
                    exp = future_to_exp[fut]
                    elapsed = time.time() - start_time
                    remaining = max(0.1, time_budget - elapsed)
                    try:
                        chain = fut.result(timeout=remaining)
                        if chain is not None:
                            option_chains[exp] = chain
                    except Exception as e:
                        timed_out = True
                        logger.warning("Dépassement de budget ou erreur pour l'échéance %s : %s", exp, e)

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
        current_dividend: float = 0.0
    ) -> Optional[pd.DataFrame]:
        """
        Extrait les données pour la surface de volatilité implicite.
        
        Pipeline de filtrage et d'interpolation :
          1. Convention OTM-only : puts pour K < S, calls pour K ≥ S
          2. Filtres de liquidité (bid/ask valides, spread maximal de 50 %)
          3. Filtrage par moneyness (MONEYNESS_MIN ≤ K/S ≤ MONEYNESS_MAX)
          4. IV calculée par inversion BSM depuis mid-price (bid+ask)/2 avec r et q
        
        Args:
            ticker_symbol: Symbole du titre
            current_price: Prix actuel (récupéré automatiquement si None)
            current_rate: Taux sans risque (SOFR)
            current_dividend: Rendement de dividende
            
        Returns:
            DataFrame avec colonnes: ['Strike', 'Days_to_Maturity', 'IV', 'Option_Type', 'Mid_Price']
        """
        # acquisition dynamique du spot en cas d'absence
        if current_price is None:
            current_price = self.data_fetcher.get_live_price(ticker_symbol)
            if current_price is None:
                logger.warning("Impossible de récupérer le prix pour %s", ticker_symbol)
                return None
        
        logger.info("Prix actuel de %s: $%.2f", ticker_symbol, current_price)
        logger.info("Paramètres: r=%.4f, q=%.4f", current_rate, current_dividend)
        
        # définition de la fenêtre de pertinence autour de la monnaie
        K_min = current_price * MONEYNESS_MIN
        K_max = current_price * MONEYNESS_MAX
        logger.info(
            "Filtre moneyness: K ∈ [%.1f, %.1f] (%.0f%% : %.0f%% du spot)",
            K_min, K_max, MONEYNESS_MIN * 100, MONEYNESS_MAX * 100,
        )
        
        # requête globale sur l'ensemble de la nappe d'échéances
        option_chains, warning = self.get_option_chains_multiple_expirations(ticker_symbol)
        if not option_chains:
            logger.warning("Aucune chaîne d'options récupérée")
            return None, warning
        
        surface_data = []
        today = datetime.now().date()
        
        stats = {
            'total': 0,
            'expired': 0,
            'moneyness_filtered': 0,
            'liquidity_filtered': 0,
            'iv_failed': 0,
            'accepted': 0,
        }
        
        for exp_date_str, opt_chain in option_chains.items():
            try:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                days_to_maturity = (exp_date - today).days
                
                # filtrage des maturités expirées ou trop proches
                if days_to_maturity < MIN_DAYS_TO_MATURITY:
                    stats['expired'] += 1
                    continue
                
                T = days_to_maturity / 365.0
                forward = current_price * np.exp((current_rate - current_dividend) * T)

                # isolation systématique des options hors de la monnaie pour maximiser la liquidité
                otm_segments = [
                    ('put',  opt_chain.puts,  lambda k: k < forward),
                    ('call', opt_chain.calls, lambda k: k >= forward),
                ]
                
                for option_type, data, is_otm_fn in otm_segments:
                    if data.empty:
                        continue
                    
                    for _, row in data.iterrows():
                        K = row['strike']
                        stats['total'] += 1
                        
                        # ── Filtre OTM ──────────────────────────────────
                        if not is_otm_fn(K):
                            continue
                        
                        # ── Filtre moneyness ────────────────────────────
                        if K < K_min or K > K_max:
                            stats['moneyness_filtered'] += 1
                            continue
                        
                        # ── Filtres de liquidité ────────────────────────
                        mid = valid_mid_quote(row, require_liquidity=True)
                        if mid is None:
                            stats['liquidity_filtered'] += 1
                            continue
                        
                        # ── Calcul IV par inversion BSM ─────────────────
                        iv = self._compute_iv_from_mid(
                            mid, current_price, K, T,
                            current_rate, current_dividend, option_type,
                        )
                        
                        if iv is None:
                            stats['iv_failed'] += 1
                            continue
                        
                        stats['accepted'] += 1
                        surface_data.append({
                            'Strike': K,
                            'Days_to_Maturity': days_to_maturity,
                            'IV': iv,
                            'Option_Type': option_type,
                            'Mid_Price': mid,
                        })
            
            except Exception as e:
                logger.warning("Erreur lors du traitement de %s: %s", exp_date_str, e)
                continue
        
        logger.info(
            "Pipeline IV : Total: %d | Expiré: %d | Moneyness: -%d | "
            "Liquidité: -%d | IV échouée: -%d | Accepté: %d",
            stats['total'], stats['expired'], stats['moneyness_filtered'],
            stats['liquidity_filtered'], stats['iv_failed'], stats['accepted'],
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
        strike_grid_size: int = STRIKE_GRID_SIZE,
        maturity_grid_size: int = MATURITY_GRID_SIZE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpole les données de surface pour créer une grille lisse 3D.
        
        Args:
            surface_data: DataFrame avec ['Strike', 'Days_to_Maturity', 'IV']
            strike_grid_size: Nombre de points pour l'axe Strike
            maturity_grid_size: Nombre de points pour l'axe Maturity
            
        Returns:
            Tuple: (X_grid, Y_grid, Z_grid) pour plotly
        """
        # extraction de l'échantillon brut
        points = surface_data[['Strike', 'Days_to_Maturity']].values
        values = surface_data['IV'].values
        
        # identification de l'enveloppe spatiale
        strike_min, strike_max = surface_data['Strike'].min(), surface_data['Strike'].max()
        maturity_min, maturity_max = (
            surface_data['Days_to_Maturity'].min(),
            surface_data['Days_to_Maturity'].max()
        )
        
        X_grid = np.linspace(strike_min, strike_max, strike_grid_size)
        Y_grid = np.linspace(maturity_min, maturity_max, maturity_grid_size)
        X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)
        
        # Restrict interpolation to the convex hull of observed quotes.
        Z_mesh = griddata(
            points, 
            values, 
            (X_mesh, Y_mesh),
            method=INTERPOLATION_METHOD,
        )

        mask_invalid = Z_mesh <= 0
        if mask_invalid.any():
            Z_mesh[mask_invalid] = np.nan

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
    ) -> Tuple[Optional[pd.DataFrame], Optional[Tuple]]:
        """
        Récupère et interpole la surface IV pour un ticker.
        
        Le pipeline complet applique :
          - Sélection OTM-only (puts K<S, calls K≥S)
          - Filtres de liquidité et moneyness
          - Calcul IV par inversion BSM depuis mid-price
          - Interpolation cubique avec fallback nearest
        
        Args:
            ticker_symbol: Symbole du titre
            current_price: Prix actuel (optionnel, récupéré si None)
            current_rate: Taux sans risque (SOFR)
            current_dividend: Dividend yield
            
        Returns:
            Tuple: (raw_data_df, (X_grid, Y_grid, Z_grid)) ou (None, None) si erreur
        """
        logger.info("Extraction de la surface IV pour %s...", ticker_symbol)

        if current_price is None:
            current_price = self.data_fetcher.get_live_price(ticker_symbol)
            if current_price is None:
                logger.warning("Impossible de récupérer le prix pour %s", ticker_symbol)
                return None, None, None

        # récupération de l'échantillon pré-traité
        surface_data, warning = self.extract_iv_surface_data(
            ticker_symbol, current_price, current_rate, current_dividend,
        )
        if surface_data is None or surface_data.empty:
            return None, None, warning
        
        # application du seuil de troncature temporelle
        surface_data = surface_data[
            surface_data['Days_to_Maturity'] <= MAX_DAYS_TO_MATURITY
        ].copy()
        
        if surface_data.empty:
            logger.warning("Aucune donnée dans la plage de %s jours", MAX_DAYS_TO_MATURITY)
            return None, None, warning
        
        logger.info("Interpolation de la surface (%d points)...", len(surface_data))
        try:
            X_grid, Y_grid, Z_grid = self.interpolate_surface(surface_data)
            logger.info("Surface interpolée avec succès")
            return surface_data, (X_grid, Y_grid, Z_grid), warning
        except Exception as e:
            logger.warning("Erreur lors de l'interpolation: %s", e)
            return surface_data, None, warning
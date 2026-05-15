"""
Module pour calculer et extraire la surface de volatilité implicite.
Surface 3D : X = Strike, Y = Maturity, Z = Implied Volatility

Méthodologie :
  1. Convention OTM-only (puts K<S, calls K≥S) — options les plus liquides
  2. Filtrage de liquidité (bid>0, ask>0, spread raisonnable, volume/OI)
  3. Filtrage par moneyness (±30% autour du spot)
  4. Calcul de l'IV par inversion BSM depuis le mid-price (bid+ask)/2
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from datetime import datetime
from scipy.interpolate import griddata
from scipy.optimize import brentq

from data_fetcher import DataFetcher
from logic.bsm_logic import OptionModels

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION (ex iv_surface_config.py — constantes inlinées)
# ═══════════════════════════════════════════════════════════════════
NUM_EXPIRATIONS = 20
MAX_DAYS_TO_MATURITY = 400
MIN_STRIKES_REQUIRED = 5
MAX_STRIKES_PER_EXPIRATION = 100
MONEYNESS_MIN = 0.70
MONEYNESS_MAX = 1.30
MAX_SPREAD_PCT = 0.50
MIN_OPEN_INTEREST = 10
IV_MIN_THRESHOLD = 0.01
IV_MAX_THRESHOLD = 3.0
STRIKE_GRID_SIZE = 30
MATURITY_GRID_SIZE = 12
DATA_PADDING_PERCENT = 0.05
INTERPOLATION_METHOD = 'cubic'
INTERPOLATION_FALLBACK = 'nearest'


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
        - Filtres de liquidité et de moneyness conformes aux standards Bloomberg
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
        if mid_price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return None

        # Vérifier que le prix est supérieur à la valeur intrinsèque
        if option_type == 'call':
            intrinsic = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
        else:
            intrinsic = max(0, K * np.exp(-r * T) - S * np.exp(-q * T))

        if mid_price < intrinsic * 0.95:  # marge de 5% pour le bruit de marché
            return None

        def objective(sigma):
            try:
                return self.option_models.black_scholes_price(
                    S, K, T, r, sigma, q, option_type
                ) - mid_price
            except Exception:
                return 1e10

        try:
            iv = brentq(
                objective,
                IV_MIN_THRESHOLD, IV_MAX_THRESHOLD,
                xtol=1e-6, maxiter=100,
            )
            # Rejeter les IV aberrantes
            if iv < IV_MIN_THRESHOLD or iv > IV_MAX_THRESHOLD:
                return None
            return iv
        except (ValueError, RuntimeError):
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Récupération des chaînes d'options
    # ─────────────────────────────────────────────────────────────────────

    def get_option_chains_multiple_expirations(
        self, 
        ticker_symbol: str
    ) -> Dict[str, object]:
        """
        Récupère les chaînes d'options pour plusieurs dates d'expiration.
        
        Args:
            ticker_symbol: Symbole du titre (ex: 'AAPL')
            
        Returns:
            Dict: {'expiration_date': option_chain, ...}
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(ticker_symbol)
            expirations = ticker.options
            
            if not expirations:
                logger.warning("Aucune date d'expiration trouvée pour %s", ticker_symbol)
                return {}
            
            # Limiter au nombre configuré d'expirations
            expirations = expirations[:NUM_EXPIRATIONS]
            
            option_chains = {}
            for exp_date in expirations:
                try:
                    opt_chain = ticker.option_chain(exp_date)
                    option_chains[exp_date] = opt_chain
                except Exception as e:
                    logger.warning("Erreur lors de la récupération de %s: %s", exp_date, e)
                    continue
            
            return option_chains
        
        except Exception as e:
            logger.warning("Erreur lors de la récupération des chaînes d'options: %s", e)
            return {}

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
        
        Pipeline conforme Bloomberg :
          1. Convention OTM-only : puts pour K < S, calls pour K ≥ S
          2. Filtres de liquidité (bid > 0, ask > 0, spread < MAX_SPREAD_PCT)
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
        # Récupérer le prix actuel si non fourni
        if current_price is None:
            current_price = self.data_fetcher.get_live_price(ticker_symbol)
            if current_price is None:
                logger.warning("Impossible de récupérer le prix pour %s", ticker_symbol)
                return None
        
        logger.info("Prix actuel de %s: $%.2f", ticker_symbol, current_price)
        logger.info("Paramètres: r=%.4f, q=%.4f", current_rate, current_dividend)
        
        # Bornes de moneyness
        K_min = current_price * MONEYNESS_MIN
        K_max = current_price * MONEYNESS_MAX
        logger.info(
            "Filtre moneyness: K ∈ [%.1f, %.1f] (%.0f%% — %.0f%% du spot)",
            K_min, K_max, MONEYNESS_MIN * 100, MONEYNESS_MAX * 100,
        )
        
        # Récupérer les chaînes d'options
        option_chains = self.get_option_chains_multiple_expirations(ticker_symbol)
        if not option_chains:
            logger.warning("Aucune chaîne d'options récupérée")
            return None
        
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
                
                # Ignorer les expirations du jour ou passées
                if days_to_maturity <= 1:
                    stats['expired'] += 1
                    continue
                
                T = days_to_maturity / 365.0
                
                # ── Convention OTM-only ─────────────────────────────────
                # Puts OTM (K < S) : plus liquides que les calls ITM
                # Calls OTM (K ≥ S) : plus liquides que les puts ITM
                otm_segments = [
                    ('put',  opt_chain.puts,  lambda k: k < current_price),
                    ('call', opt_chain.calls, lambda k: k >= current_price),
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
                        bid = row.get('bid', 0)
                        ask = row.get('ask', 0)
                        volume = row.get('volume', 0)
                        oi = row.get('openInterest', 0)
                        
                        # Coerce NaN to 0
                        bid = float(bid) if pd.notna(bid) else 0.0
                        ask = float(ask) if pd.notna(ask) else 0.0
                        volume = int(volume) if pd.notna(volume) else 0
                        oi = int(oi) if pd.notna(oi) else 0
                        
                        if bid <= 0 or ask <= 0:
                            stats['liquidity_filtered'] += 1
                            continue
                        
                        mid = (bid + ask) / 2.0
                        spread_pct = (ask - bid) / mid if mid > 0 else 999.0
                        
                        if spread_pct > MAX_SPREAD_PCT:
                            stats['liquidity_filtered'] += 1
                            continue
                        
                        if volume <= 0 and oi < MIN_OPEN_INTEREST:
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
            "Pipeline IV — Total: %d | Expiré: %d | Moneyness: -%d | "
            "Liquidité: -%d | IV échouée: -%d | Accepté: %d",
            stats['total'], stats['expired'], stats['moneyness_filtered'],
            stats['liquidity_filtered'], stats['iv_failed'], stats['accepted'],
        )
        
        if not surface_data:
            logger.warning("Aucune donnée IV valide extraite après filtrage")
            return None
        
        df = pd.DataFrame(surface_data)
        logger.info("Données extraites: %s points", len(df))
        logger.info("  Strikes: %.2f — %.2f", df['Strike'].min(), df['Strike'].max())
        logger.info("  Maturité: %s — %s jours",
                     df['Days_to_Maturity'].min(), df['Days_to_Maturity'].max())
        logger.info("  IV: %.4f — %.4f (%.1f%% — %.1f%%)",
                     df['IV'].min(), df['IV'].max(),
                     df['IV'].min() * 100, df['IV'].max() * 100)
        
        return df

    # ─────────────────────────────────────────────────────────────────────
    # Interpolation de la surface
    # ─────────────────────────────────────────────────────────────────────

    def interpolate_surface(
        self, 
        surface_data: pd.DataFrame,
        current_price: Optional[float] = None,
        strike_grid_size: int = 30,
        maturity_grid_size: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpole les données de surface pour créer une grille lisse 3D.
        
        Args:
            surface_data: DataFrame avec ['Strike', 'Days_to_Maturity', 'IV']
            current_price: Prix spot (pour clamper le strike_min)
            strike_grid_size: Nombre de points pour l'axe Strike
            maturity_grid_size: Nombre de points pour l'axe Maturity
            
        Returns:
            Tuple: (X_grid, Y_grid, Z_grid) pour plotly
        """
        # Points source
        points = surface_data[['Strike', 'Days_to_Maturity']].values
        values = surface_data['IV'].values
        
        # Bornes des données
        strike_min, strike_max = surface_data['Strike'].min(), surface_data['Strike'].max()
        maturity_min, maturity_max = (
            surface_data['Days_to_Maturity'].min(),
            surface_data['Days_to_Maturity'].max()
        )
        
        # Padding léger (5%) avec clamp sur des valeurs financièrement raisonnables
        strike_range = strike_max - strike_min
        maturity_range = maturity_max - maturity_min
        
        strike_min -= strike_range * 0.05
        strike_max += strike_range * 0.05

        # Clamper strike_min à une valeur raisonnable (>0, ≥ 50% du spot)
        floor = current_price * 0.5 if current_price else 1.0
        strike_min = max(floor, strike_min)
        
        maturity_min = max(1, maturity_min - maturity_range * 0.1)
        maturity_max += maturity_range * 0.1
        
        X_grid = np.linspace(strike_min, strike_max, strike_grid_size)
        Y_grid = np.linspace(maturity_min, maturity_max, maturity_grid_size)
        X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)
        
        # Interpoler avec griddata (cubique pour une surface lisse)
        Z_mesh = griddata(
            points, 
            values, 
            (X_mesh, Y_mesh),
            method='cubic'
        )
        
        # Remplir les NaN avec nearest neighbor
        mask_nan = np.isnan(Z_mesh)
        if mask_nan.any():
            Z_mesh[mask_nan] = griddata(
                points,
                values,
                (X_mesh[mask_nan], Y_mesh[mask_nan]),
                method='nearest'
            )

        # Remplacer les valeurs <= 0 par nearest neighbor
        # Une IV négative ou nulle est impossible financièrement — elle résulte
        # d'une extrapolation cubique en dehors de la convexe des données.
        # On corrige par le point de données réel le plus proche plutôt que
        # de laisser un trou ou une valeur aberrante dans la surface.
        mask_invalid = Z_mesh <= 0
        if mask_invalid.any():
            Z_mesh[mask_invalid] = griddata(
                points,
                values,
                (X_mesh[mask_invalid], Y_mesh[mask_invalid]),
                method='nearest'
            )

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
        
        # Extraire les données brutes (déjà filtrées et nettoyées par le pipeline)
        surface_data = self.extract_iv_surface_data(
            ticker_symbol, current_price, current_rate, current_dividend,
        )
        if surface_data is None or surface_data.empty:
            return None, None
        
        # Filtrer par nombre max de jours (configurable)
        surface_data = surface_data[
            surface_data['Days_to_Maturity'] <= MAX_DAYS_TO_MATURITY
        ].copy()
        
        if surface_data.empty:
            logger.warning("Aucune donnée dans la plage de %s jours", MAX_DAYS_TO_MATURITY)
            return None, None
        
        logger.info("Interpolation de la surface (%d points)...", len(surface_data))
        try:
            X_grid, Y_grid, Z_grid = self.interpolate_surface(
                surface_data, current_price,
            )
            logger.info("Surface interpolée avec succès")
            return surface_data, (X_grid, Y_grid, Z_grid)
        except Exception as e:
            logger.warning("Erreur lors de l'interpolation: %s", e)
            return surface_data, None
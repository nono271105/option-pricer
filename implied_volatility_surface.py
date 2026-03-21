"""
Module pour calculer et extraire la surface de volatilité implicite.
Surface 3D : X = Strike, Y = Maturity, Z = Implied Volatility
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta
from scipy.interpolate import griddata
import warnings

from data_fetcher import DataFetcher
from option_models import OptionModels
import iv_surface_config as config


class ImpliedVolatilitySurface:
    """
    Calcule la surface de volatilité implicite (IV Surface) pour un ticker.
    
    Surface 3D:
        X-axis: Strike prices (K)
        Y-axis: Time to Maturity (T) en jours
        Z-axis: Implied Volatility (σ)
    """
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.option_models = OptionModels()
    
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
                print(f"Aucune date d'expiration trouvée pour {ticker_symbol}")
                return {}
            
            # Limiter au nombre configuré d'expirations
            expirations = expirations[:config.NUM_EXPIRATIONS]
            
            option_chains = {}
            for exp_date in expirations:
                try:
                    opt_chain = ticker.option_chain(exp_date)
                    option_chains[exp_date] = opt_chain
                except Exception as e:
                    print(f"Erreur lors de la récupération de {exp_date}: {e}")
                    continue
            
            return option_chains
        
        except Exception as e:
            print(f"Erreur lors de la récupération des chaînes d'options: {e}")
            return {}
    
    def extract_iv_surface_data(
        self, 
        ticker_symbol: str,
        current_price: Optional[float] = None,
        current_rate: float = 0.05,
        current_dividend: float = 0.0
    ) -> Optional[pd.DataFrame]:
        """
        Extrait les données pour la surface de volatilité implicite.
        
        Args:
            ticker_symbol: Symbole du titre
            current_price: Prix actuel (récupéré si None)
            current_rate: Taux sans risque
            current_dividend: Rendement de dividende
            
        Returns:
            DataFrame avec colonnes: ['Strike', 'Days_to_Maturity', 'IV', 'Option_Type']
        """
        # Récupérer le prix actuel si non fourni
        if current_price is None:
            current_price = self.data_fetcher.get_live_price(ticker_symbol)
            if current_price is None:
                print(f"Impossible de récupérer le prix pour {ticker_symbol}")
                return None
        
        print(f"Prix actuel de {ticker_symbol}: ${current_price:.2f}")
        
        # Récupérer les chaînes d'options
        option_chains = self.get_option_chains_multiple_expirations(ticker_symbol)
        if not option_chains:
            print("Aucune chaîne d'options récupérée")
            return None
        
        # Extrai les données IV pour tous les strikes et expirations
        surface_data = []
        today = datetime.now().date()
        
        for exp_date_str, opt_chain in option_chains.items():
            try:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                days_to_maturity = (exp_date - today).days
                
                if days_to_maturity <= 0:
                    continue
                
                # Traiter les Calls et Puts
                for option_type, data in [('call', opt_chain.calls), ('put', opt_chain.puts)]:
                    # Filtrer les données valides
                    data = data[
                        (data['impliedVolatility'] > 0) & 
                        (data['lastPrice'] > 0) &
                        (data['strike'] > 0)
                    ].copy()
                    
                    for _, row in data.iterrows():
                        strike = float(row['strike'])
                        iv = float(row['impliedVolatility'])
                        
                        # Filtrer les IV aberrantes (< 0.001 ou > 5)
                        if iv < 0.001 or iv > 5.0:
                            continue
                        
                        surface_data.append({
                            'Strike': strike,
                            'Days_to_Maturity': days_to_maturity,
                            'IV': iv,
                            'Option_Type': option_type
                        })
            
            except Exception as e:
                print(f"Erreur lors du traitement de {exp_date_str}: {e}")
                continue
        
        if not surface_data:
            print("Aucune donnée IV valide extraite")
            return None
        
        df = pd.DataFrame(surface_data)
        print(f"Données extraites: {len(df)} points")
        print(f"  Strikes: {df['Strike'].min():.2f} - {df['Strike'].max():.2f}")
        print(f"  Maturité: {df['Days_to_Maturity'].min()} - {df['Days_to_Maturity'].max()} jours")
        print(f"  IV: {df['IV'].min():.4f} - {df['IV'].max():.4f}")
        
        return df
    
    def interpolate_surface(
        self, 
        surface_data: pd.DataFrame,
        strike_grid_size: int = 30,
        maturity_grid_size: int = 20
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
        # Points source
        points = surface_data[['Strike', 'Days_to_Maturity']].values
        values = surface_data['IV'].values
        
        # Créer la grille
        strike_min, strike_max = surface_data['Strike'].min(), surface_data['Strike'].max()
        maturity_min, maturity_max = (
            surface_data['Days_to_Maturity'].min(),
            surface_data['Days_to_Maturity'].max()
        )
        
        # Ajouter 5% de padding
        strike_range = strike_max - strike_min
        maturity_range = maturity_max - maturity_min
        
        strike_min -= strike_range * 0.05
        strike_max += strike_range * 0.05
        maturity_min = max(0, maturity_min - maturity_range * 0.2)
        maturity_max += maturity_range * 0.2
        
        X_grid = np.linspace(strike_min, strike_max, strike_grid_size)
        Y_grid = np.linspace(maturity_min, maturity_max, maturity_grid_size)
        X_mesh, Y_mesh = np.meshgrid(X_grid, Y_grid)
        
        # Interpoler avec griddata
        Z_mesh = griddata(
            points, 
            values, 
            (X_mesh, Y_mesh),
            method='cubic'
        )
        
        # Remplir les NaN avec nearest neighbor
        mask = np.isnan(Z_mesh)
        if mask.any():
            Z_mesh[mask] = griddata(
                points,
                values,
                (X_mesh[mask], Y_mesh[mask]),
                method='nearest'
            )
        
        return X_mesh, Y_mesh, Z_mesh
    
    def get_surface_for_ticker(
        self,
        ticker_symbol: str,
        current_price: Optional[float] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[Tuple]]:
        """
        Récupère et interpole la surface IV pour un ticker.
        
        Args:
            ticker_symbol: Symbole du titre
            current_price: Prix actuel (optionnel)
            
        Returns:
            Tuple: (raw_data_df, (X_grid, Y_grid, Z_grid)) ou (None, None) si erreur
        """
        print(f"Extraction de la surface IV pour {ticker_symbol}...")
        
        # Extraire les données brutes
        surface_data = self.extract_iv_surface_data(ticker_symbol, current_price)
        if surface_data is None or surface_data.empty:
            return None, None
        
        # Prendre uniquement les calls pour réduire les données et améliorer la cohérence
        surface_data = surface_data[surface_data['Option_Type'] == 'call'].copy()
        
        if surface_data.empty:
            print("Aucun Call disponible pour interpolation")
            return None, None
        
        # Filtrer les données au-delà du nombre maximum de jours configuré
        surface_data = surface_data[surface_data['Days_to_Maturity'] <= config.MAX_DAYS_TO_MATURITY].copy()
        
        if surface_data.empty:
            print(f"Aucune donnée dans la plage de {config.MAX_DAYS_TO_MATURITY} jours")
            return None, None
        
        print("Interpolation de la surface...")
        try:
            X_grid, Y_grid, Z_grid = self.interpolate_surface(surface_data)
            print("✓ Surface interpolée avec succès")
            return surface_data, (X_grid, Y_grid, Z_grid)
        except Exception as e:
            print(f"Erreur lors de l'interpolation: {e}")
            return surface_data, None

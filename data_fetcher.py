import yfinance as yf
import requests
from datetime import datetime, date
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from cache import global_cache

load_dotenv()

class DataFetcher:
    def __init__(self) -> None:
        self.fred_api_key: Optional[str] = os.getenv("FRED_API_KEY")

    def get_live_price(self, ticker_symbol: str) -> Optional[float]:
        """Récupère le prix en direct du dernier jour de trading."""
        # Vérifier le cache d'abord
        cache_key = f"live_price_{ticker_symbol}"
        cached_price = global_cache.get(cache_key)
        if cached_price is not None:
            return cached_price
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            todays_data = ticker.history(period='1d')
            if not todays_data.empty:
                price = float(todays_data['Close'].iloc[-1])
                # Stocker dans le cache
                global_cache.set(cache_key, price)
                return price
            return None
        except Exception as e:
            print(f"Erreur lors de la récupération du prix en direct pour {ticker_symbol}: {e}")
            return None

    def get_historical_volatility(self, ticker_symbol: str, period: str = "1y") -> Optional[float]:
        """Récupère la volatilité historique annualisée."""
        # Vérifier le cache
        cache_key = f"vol_{ticker_symbol}_{period}"
        cached_vol = global_cache.get(cache_key)
        if cached_vol is not None:
            return cached_vol
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=period)
            if hist.empty:
                return None
            
            # Calcul de la volatilité annuelle
            returns = hist['Close'].pct_change().dropna()
            if len(returns) < 2: 
                return None
            
            annual_volatility = returns.std() * np.sqrt(252)
            # Stocker dans le cache
            global_cache.set(cache_key, annual_volatility)
            return annual_volatility
        except Exception as e:
            print(f"Erreur lors de la récupération de la volatilité historique pour {ticker_symbol}: {e}")
            return None

    def get_sofr_rate(self) -> Optional[float]:
        """
        Récupère le taux SOFR le plus récent depuis l'API FRED.
        
        Returns:
            Optional[float]: Taux SOFR décimalisé (ex: 0.05 pour 5%) ou None
        """
        # Vérifier le cache (SOFR change rarement en 1h)
        cache_key = "sofr_rate"
        cached_rate = global_cache.get(cache_key)
        if cached_rate is not None:
            return cached_rate
        
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=SOFR&api_key={self.fred_api_key}&file_type=json"
        try:
            response = requests.get(url)
            response.raise_for_status() 
            data = response.json()
            
            observations = data.get('observations', [])
            if observations:
                latest_observation = observations[-1]
                sofr_value = float(latest_observation['value'])
                sofr_decimal = sofr_value / 100.0
                # Stocker dans le cache
                global_cache.set(cache_key, sofr_decimal)
                return sofr_decimal
            else:
                print("Aucune observation SOFR trouvée dans la réponse de l'API.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Erreur de requête HTTP lors de la récupération du SOFR : {e}")
            return None
        except ValueError as e:
            print(f"Erreur de parsing JSON ou de conversion de valeur pour le SOFR : {e}")
            return None
        except Exception as e:
            print(f"Une erreur inattendue est survenue lors de la récupération du SOFR : {e}")
            return None

    def get_dividend_yield(self, ticker_symbol: str) -> float:
        """
        Récupère le rendement de dividende annuel.
        
        Args:
            ticker_symbol: Symbole du titre
            
        Returns:
            float: Rendement de dividende annuel décimalisé (défaut: 0.0)
        """
        # Vérifier le cache
        cache_key = f"dividend_{ticker_symbol}"
        cached_div = global_cache.get(cache_key)
        if cached_div is not None:
            return cached_div
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            dividend_yield = info.get("dividendYield")
            
            if dividend_yield is not None:
                div_decimal = float(dividend_yield) / 100.0
                # Stocker dans le cache
                global_cache.set(cache_key, div_decimal)
                return div_decimal
            else:
                # Stocker 0.0 aussi dans le cache pour éviter des requêtes répétées
                global_cache.set(cache_key, 0.0)
                return 0.0
        except Exception as e:
            print(f"Erreur lors de la récupération du rendement de dividende pour {ticker_symbol}: {e}")
            return 0.0

    def get_option_data_chain(
        self, 
        ticker_symbol: str, 
        maturity_datetime: datetime
    ) -> Tuple[Optional[object], Optional[str]]:
        """
        Récupère la chaîne d'options pour la date d'expiration la plus proche.
        
        Args:
            ticker_symbol: Symbole du titre
            maturity_datetime: Date d'expiration souhaitée
            
        Returns:
            Tuple[Optional[OptionChain], Optional[str]]: (option_chain, date_expiration_réelle)
        """
        try:
            ticker = yf.Ticker(ticker_symbol)
            expirations = ticker.options
            
            if not expirations:
                print("Aucune date d'expiration trouvée.")
                return None, None

            # Trouver la date d'expiration disponible la plus proche
            closest_date = min(expirations, 
                               key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d').date() - maturity_datetime.date()))
            
            # Récupérer la chaîne d'options pour cette date
            opt_chain = ticker.option_chain(closest_date)
            
            return opt_chain, closest_date

        except Exception as e:
            print(f"Erreur lors de la récupération de la chaîne d'options: {e}")
            return None, None
            
    def get_implied_volatility_and_price(
        self, 
        ticker_symbol: str, 
        strike: float, 
        maturity_datetime: datetime, 
        option_type: str
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Récupère l'IV et le prix du marché pour un strike donné.
        
        Args:
            ticker_symbol: Symbole du titre
            strike: Prix d'exercice
            maturity_datetime: Date d'expiration
            option_type: 'call' ou 'put'
            
        Returns:
            Tuple[Optional[float], Optional[float], Optional[str]]: (IV, prix, date_expiration)
        """
        opt_chain, closest_date = self.get_option_data_chain(ticker_symbol, maturity_datetime)

        if opt_chain is None or closest_date is None:
            return None, None, None

        option_type = option_type.lower()
        
        # 1. Sélectionner les Calls ou les Puts
        if option_type == 'call':
            data = opt_chain.calls
        elif option_type == 'put':
            data = opt_chain.puts
        else:
            print(f"Type d'option non reconnu: {option_type}")
            return None, None, None
        
        # 2. Trouver le strike le plus proche dans la liste
        if data.empty:
            print("Aucune donnée d'option pour cette expiration.")
            return None, None, closest_date
            
        # Assurer que 'strike' est une colonne numérique et calculer la différence absolue
        data['abs_diff'] = abs(data['strike'] - strike)
        closest_row = data.sort_values('abs_diff').iloc[0]
        
        iv = closest_row['impliedVolatility']
        price = closest_row['lastPrice']
        
        # Vérification de sécurité et s'assurer que l'IV et le prix sont valides
        if iv is None or iv <= 0.001 or price is None or price <= 0:
            print(f"IV ({iv}) ou Prix ({price}) non valide ou nul pour K={strike} et type={option_type}.")
            return None, None, closest_date
            
        return iv, price, closest_date

    # Récupération des données pour le Sourire de Volatilité 
    def get_volatility_smile_data(
        self, 
        ticker_symbol: str, 
        maturity_datetime: datetime
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Récupère les Strikes et IV pour les Calls et Puts d'une expiration donnée.
        
        Args:
            ticker_symbol: Symbole du titre
            maturity_datetime: Date d'expiration
            
        Returns:
            Tuple[Optional[DataFrame], Optional[str]]: (DataFrame avec ['strike', 'impliedVolatility', 'type'], date_expiration)
        """
        opt_chain, closest_date = self.get_option_data_chain(ticker_symbol, maturity_datetime)

        if opt_chain is None or closest_date is None:
            return None, None

        # Nettoyage et combinaison des données Call et Put pour le tracé du sourire
        calls_df = opt_chain.calls[['strike', 'impliedVolatility', 'lastPrice']].dropna()
        puts_df = opt_chain.puts[['strike', 'impliedVolatility', 'lastPrice']].dropna()

        # Filtrer les IVs non significatives ou nulles
        calls_df = calls_df[calls_df['impliedVolatility'] > 1e-6]
        puts_df = puts_df[puts_df['impliedVolatility'] > 1e-6]
        
        # Ajouter une colonne 'type' pour distinguer les points
        calls_df['type'] = 'call'
        puts_df['type'] = 'put'

        # Correction de l'erreur 'append' en utilisant pd.concat
        combined_df = pd.concat([
            calls_df[['strike', 'impliedVolatility', 'type']],
            puts_df[['strike', 'impliedVolatility', 'type']]
        ]).drop_duplicates(subset=['strike', 'impliedVolatility']).reset_index(drop=True)
        
        return combined_df, closest_date
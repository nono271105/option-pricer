import logging
import yfinance as yf
import requests
from datetime import datetime
from dotenv import load_dotenv
import os
import numpy as np
from typing import Optional, Tuple
from cache import global_cache_price, global_cache_chain, global_cache_static

load_dotenv()

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self) -> None:
        self.fred_api_key: Optional[str] = os.getenv("FRED_API_KEY")

    def get_live_price(self, ticker_symbol: str) -> Optional[float]:
        """Récupère le prix en direct du dernier jour de trading."""
        # cache consulté en premier pour éviter un appel réseau inutile
        cache_key = f"live_price_{ticker_symbol}"
        cached_price = global_cache_price.get(cache_key)
        if cached_price is not None:
            return cached_price
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            todays_data = ticker.history(period='1d')
            if not todays_data.empty:
                price = float(todays_data['Close'].iloc[-1])
                # on met en cache avant de retourner
                global_cache_price.set(cache_key, price)
                return price
            return None
        except Exception as e:
            logger.warning("Erreur prix en direct pour %s : %s", ticker_symbol, e)
            return None

    def get_historical_volatility(self, ticker_symbol: str, period: str = "1y") -> Optional[float]:
        """Récupère la volatilité historique annualisée."""
        # même logique de cache
        cache_key = f"vol_{ticker_symbol}_{period}"
        cached_vol = global_cache_static.get(cache_key)
        if cached_vol is not None:
            return cached_vol
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=period)
            if hist.empty:
                return None
            
            # volatilité annualisée par écart-type des rendements quotidiens
            returns = hist['Close'].pct_change(fill_method=None).dropna()
            if len(returns) < 2: 
                return None
            
            annual_volatility = returns.std() * np.sqrt(252)
            # on met en cache avant de retourner
            global_cache_static.set(cache_key, annual_volatility)
            return annual_volatility
        except Exception as e:
            logger.warning("Erreur volatilité historique pour %s : %s", ticker_symbol, e)
            return None

    def get_sofr_rate(self) -> Optional[float]:
        """
        Récupère le taux SOFR le plus récent depuis l'API FRED.
        
        Returns:
            Optional[float]: Taux SOFR décimalisé (ex: 0.05 pour 5%) ou None
        """
        # le SOFR varie peu en 1h, le cache est particulièrement adapté ici
        cache_key = "sofr_rate"
        cached_rate = global_cache_static.get(cache_key)
        if cached_rate is not None:
            return cached_rate
        
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=SOFR&api_key={self.fred_api_key}&file_type=json"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status() 
            data = response.json()
            
            observations = data.get('observations', [])
            if observations:
                latest_observation = observations[-1]
                sofr_value = float(latest_observation['value'])
                sofr_decimal = sofr_value / 100.0
                # on met en cache avant de retourner
                global_cache_static.set(cache_key, sofr_decimal)
                return sofr_decimal
            else:
                logger.warning("Aucune observation SOFR trouvée dans la réponse de l'API.")
                return None
        except requests.exceptions.RequestException as e:
            logger.warning("Erreur HTTP SOFR : %s", e)
            return None
        except ValueError as e:
            logger.warning("Erreur parsing SOFR : %s", e)
            return None
        except Exception as e:
            logger.warning("Erreur inattendue SOFR : %s", e)
            return None

    def get_dividend_yield(self, ticker_symbol: str) -> float:
        """
        Récupère le rendement de dividende annuel.
        
        Args:
            ticker_symbol: Symbole du titre
            
        Returns:
            float: Rendement de dividende annuel décimalisé (défaut: 0.0)
        """
        cache_key = f"dividend_{ticker_symbol}"
        cached_div = global_cache_static.get(cache_key)
        if cached_div is not None:
            return cached_div

        result = 0.0
        try:
            info = yf.Ticker(ticker_symbol).info

            # 1. trailingAnnualDividendYield est renvoyé en décimal par yfinance (ex: 0.0034)
            tr_yield = info.get("trailingAnnualDividendYield")
            if tr_yield is not None:
                result = float(tr_yield)
            else:
                # 2. fallback : calcul manuel (dividende / cours)
                rate = info.get("trailingAnnualDividendRate") or info.get("dividendRate")
                price = info.get("previousClose") or info.get("regularMarketPrice") or self.get_live_price(ticker_symbol)
                
                if rate is not None and price is not None:
                    price_f = float(price)
                    if price_f > 0:
                        result = float(rate) / price_f
                else:
                    # 3. dernier recours : dividendYield est renvoyé en pourcentage par yfinance (ex: 0.35 pour 0.35%)
                    div_yield = info.get("dividendYield")
                    if div_yield is not None:
                        result = float(div_yield) / 100.0

            # Validation du résultat pour éviter des valeurs absurdes
            if not (0.0 <= result < 1.0):
                result = 0.0

        except (TypeError, ValueError) as e:
            logger.warning("Données dividende mal formatées pour %s : %s", ticker_symbol, e)
            result = 0.0
        except Exception as e:
            logger.warning("Erreur dividende pour %s : %s", ticker_symbol, e)
            result = 0.0

        global_cache_static.set(cache_key, result)
        return result

    def get_company_name(self, ticker_symbol: str) -> Optional[str]:
        """
        Récupère le nom de la société à partir du symbole du ticker.
        
        Args:
            ticker_symbol: Symbole du titre
            
        Returns:
            Optional[str]: Nom de la société ou None en cas d'erreur
        """
        # même logique de cache
        cache_key = f"company_name_{ticker_symbol}"
        cached_name = global_cache_static.get(cache_key)
        if cached_name is not None:
            return cached_name
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            company_name = info.get("longName") or info.get("shortName") or ticker_symbol
            # on met en cache avant de retourner
            global_cache_static.set(cache_key, company_name)
            return company_name
        except Exception as e:
            logger.warning("Erreur nom société pour %s : %s", ticker_symbol, e)
            return None

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
                logger.warning("Aucune date d'expiration trouvée pour %s.", ticker_symbol)
                return None, None

            # on cherche la date d'expiration disponible la plus proche de la date demandée
            closest_date = min(expirations, 
                               key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d').date() - maturity_datetime.date()))
            
            # récupération de la chaîne pour cette date
            opt_chain = ticker.option_chain(closest_date)
            
            return opt_chain, closest_date

        except Exception as e:
            logger.warning("Erreur chaîne d'options pour %s : %s", ticker_symbol, e)
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
        
        # sélection de la table calls ou puts selon le type demandé
        if option_type == 'call':
            data = opt_chain.calls
        elif option_type == 'put':
            data = opt_chain.puts
        else:
            logger.warning("Type d'option non reconnu : %s", option_type)
            return None, None, None
        
        # on cherche le strike coté le plus proche du strike théorique demandé
        if data.empty:
            logger.warning("Aucune donnée d'option pour cette expiration : %s", ticker_symbol)
            return None, None, closest_date
            
        # différence absolue pour identifier la ligne la plus proche
        data = data.assign(abs_diff=lambda df: abs(df['strike'] - strike))
        closest_row = data.sort_values('abs_diff').iloc[0]
        
        iv = closest_row['impliedVolatility']
        
        bid = float(closest_row.get('bid', 0) or 0)
        ask = float(closest_row.get('ask', 0) or 0)
        if bid > 0 and ask > 0:
            price = (bid + ask) / 2.0
        else:
            price = closest_row.get('lastPrice')
        
        # une IV nulle ou négative n'est pas exploitable pour le pricing
        if iv is None or iv <= 0.001 or price is None or price <= 0:
            logger.debug(
                "IV (%s) ou Prix (%s) non valide pour K=%s type=%s", iv, price, strike, option_type
            )
            return None, None, closest_date
            
        return iv, price, closest_date

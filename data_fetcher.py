import yfinance as yf
import requests
from datetime import datetime
from dotenv import load_dotenv
import os
import numpy as np
from typing import Optional, Tuple
from cache import global_cache

load_dotenv()

class DataFetcher:
    def __init__(self) -> None:
        self.fred_api_key: Optional[str] = os.getenv("FRED_API_KEY")
        self.market_data_token: Optional[str] = os.getenv("MARKET_DATA_TOKEN")

    @staticmethod
    def build_occ_symbol(ticker: str, expiration_date: str, strike: float, option_type: str) -> str:
        """
        Construit un symbole OCC standard à partir des paramètres du contrat.

        Format OCC : {TICKER}{YYMMDD}{C|P}{STRIKE*1000:08d}
        Exemple : AAPL, 2026-06-18, 300, call -> AAPL260618C00300000

        Args:
            ticker: Symbole du sous-jacent (ex: AAPL)
            expiration_date: Date d'expiration au format YYYY-MM-DD
            strike: Prix d'exercice
            option_type: 'call' ou 'put'

        Returns:
            str: Symbole OCC formaté
        """
        from datetime import datetime as dt
        exp = dt.strptime(expiration_date, "%Y-%m-%d")
        date_part = exp.strftime("%y%m%d")
        side = "C" if option_type.lower() == "call" else "P"
        strike_int = int(strike * 1000)
        return f"{ticker.upper()}{date_part}{side}{strike_int:08d}"

    def get_option_history_marketdata(
        self,
        ticker: str,
        expiration_date: str,
        strike: float,
        option_type: str,
        history_days: int = 30,
    ) -> Optional[dict]:
        """
        Recupere l'historique des prix d'un contrat d'option via marketdata.app.

        Utilise le endpoint quotes avec from/to pour obtenir les prix mid,
        le spot et le DTE de chaque jour de trading sur la periode demandee.

        Args:
            ticker: Symbole du sous-jacent
            expiration_date: Date d'expiration au format YYYY-MM-DD
            strike: Prix d'exercice
            option_type: 'call' ou 'put'
            history_days: Nombre de jours calendaires d'historique

        Returns:
            Optional[dict]: Dictionnaire avec les cles mid, underlyingPrice, dte,
                            strike, expiration, updated. None en cas d'erreur.
        """
        if not self.market_data_token:
            print("MARKET_DATA_TOKEN non configure dans le fichier .env.")
            return None

        occ_symbol = self.build_occ_symbol(ticker, expiration_date, strike, option_type)

        cache_key = f"mktdata_hist_{occ_symbol}_{history_days}"
        cached = global_cache.get(cache_key)
        if cached is not None:
            return cached

        from datetime import timedelta
        today = datetime.now()
        date_from = (today - timedelta(days=history_days)).strftime("%Y-%m-%d")
        date_to = today.strftime("%Y-%m-%d")

        url = f"https://api.marketdata.app/v1/options/quotes/{occ_symbol}/"
        headers = {"Authorization": f"Bearer {self.market_data_token}"}
        params = {"from": date_from, "to": date_to}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data.get("s") != "ok":
                print(f"marketdata.app : pas de donnees pour {occ_symbol} ({data.get('s')})")
                return None

            # filtrage des points avec prix mid valide (non null, positif)
            mid_list = data.get("mid", [])
            underlying_list = data.get("underlyingPrice", [])
            dte_list = data.get("dte", [])
            updated_list = data.get("updated", [])

            valid_indices = [
                i for i in range(len(mid_list))
                if mid_list[i] is not None
                and mid_list[i] > 0
                and underlying_list[i] is not None
            ]

            if not valid_indices:
                print(f"Aucun prix mid valide pour {occ_symbol}.")
                return None

            result = {
                "mid": [mid_list[i] for i in valid_indices],
                "underlyingPrice": [underlying_list[i] for i in valid_indices],
                "dte": [dte_list[i] for i in valid_indices],
                "updated": [updated_list[i] for i in valid_indices],
                "strike": strike,
                "expiration": expiration_date,
                "option_type": option_type,
                "occ_symbol": occ_symbol,
            }

            global_cache.set(cache_key, result)
            return result

        except requests.exceptions.RequestException as e:
            print(f"Erreur HTTP marketdata.app pour {occ_symbol} : {e}")
            return None
        except Exception as e:
            print(f"Erreur inattendue marketdata.app pour {occ_symbol} : {e}")
            return None

    def get_live_price(self, ticker_symbol: str) -> Optional[float]:
        """Récupère le prix en direct du dernier jour de trading."""
        # cache consulté en premier pour éviter un appel réseau inutile
        cache_key = f"live_price_{ticker_symbol}"
        cached_price = global_cache.get(cache_key)
        if cached_price is not None:
            return cached_price
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            todays_data = ticker.history(period='1d')
            if not todays_data.empty:
                price = float(todays_data['Close'].iloc[-1])
                # on met en cache avant de retourner
                global_cache.set(cache_key, price)
                return price
            return None
        except Exception as e:
            print(f"Erreur lors de la récupération du prix en direct pour {ticker_symbol}: {e}")
            return None

    def get_historical_volatility(self, ticker_symbol: str, period: str = "1y") -> Optional[float]:
        """Récupère la volatilité historique annualisée."""
        # même logique de cache
        cache_key = f"vol_{ticker_symbol}_{period}"
        cached_vol = global_cache.get(cache_key)
        if cached_vol is not None:
            return cached_vol
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=period)
            if hist.empty:
                return None
            
            # volatilité annualisée par écart-type des rendements quotidiens
            returns = hist['Close'].pct_change().dropna()
            if len(returns) < 2: 
                return None
            
            annual_volatility = returns.std() * np.sqrt(252)
            # on met en cache avant de retourner
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
        # le SOFR varie peu en 1h, le cache est particulièrement adapté ici
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
                # on met en cache avant de retourner
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
        cache_key = f"dividend_{ticker_symbol}"
        cached_div = global_cache.get(cache_key)
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
            # Erreur lors de la conversion des valeurs en float
            print(f"Données de dividende mal formatées pour {ticker_symbol}: {e}")
            result = 0.0
        except Exception as e:
            # Erreur réseau, ticker introuvable, etc.
            print(f"Erreur lors de la récupération du rendement de dividende pour {ticker_symbol}: {e}")
            result = 0.0

        global_cache.set(cache_key, result)
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
        cached_name = global_cache.get(cache_key)
        if cached_name is not None:
            return cached_name
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            company_name = info.get("longName") or info.get("shortName") or ticker_symbol
            # on met en cache avant de retourner
            global_cache.set(cache_key, company_name)
            return company_name
        except Exception as e:
            print(f"Erreur lors de la récupération du nom de la société pour {ticker_symbol}: {e}")
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
                print("Aucune date d'expiration trouvée.")
                return None, None

            # on cherche la date d'expiration disponible la plus proche de la date demandée
            closest_date = min(expirations, 
                               key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d').date() - maturity_datetime.date()))
            
            # récupération de la chaîne pour cette date
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
        
        # sélection de la table calls ou puts selon le type demandé
        if option_type == 'call':
            data = opt_chain.calls
        elif option_type == 'put':
            data = opt_chain.puts
        else:
            print(f"Type d'option non reconnu: {option_type}")
            return None, None, None
        
        # on cherche le strike coté le plus proche du strike théorique demandé
        if data.empty:
            print("Aucune donnée d'option pour cette expiration.")
            return None, None, closest_date
            
        # différence absolue pour identifier la ligne la plus proche
        data['abs_diff'] = abs(data['strike'] - strike)
        closest_row = data.sort_values('abs_diff').iloc[0]
        
        iv = closest_row['impliedVolatility']
        price = closest_row['lastPrice']
        
        # une IV nulle ou négative n'est pas exploitable pour le pricing
        if iv is None or iv <= 0.001 or price is None or price <= 0:
            print(f"IV ({iv}) ou Prix ({price}) non valide ou nul pour K={strike} et type={option_type}.")
            return None, None, closest_date
            
        return iv, price, closest_date

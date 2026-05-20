# Commande : pytest tests/test_data.py -v

"""
test_data.py : Tests d'intégration pour les APIs FRED (SOFR) et yfinance.
Ces tests effectuent de vraies requêtes réseau, une connexion internet est requise.
"""

import sys
import os
import pytest
import requests
import yfinance as yf
from datetime import datetime, timedelta

# ── Résolution du chemin racine pour les imports projet ────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from data_fetcher import DataFetcher

# ── Constantes de test ─────────────────────────────────────────────────────────
FRED_API_KEY   = os.getenv("FRED_API_KEY")
FRED_BASE_URL  = "https://api.stlouisfed.org/fred/series/observations"
SOFR_SERIES_ID = "SOFR"

TEST_TICKER_VALID   = "AAPL"   # Ticker liquide, toujours disponible
TEST_TICKER_INVALID = "XXXXINVALID999"
SOFR_REASONABLE_MIN = 0.0001   # 0.01 %
SOFR_REASONABLE_MAX = 0.25     # 25 %


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 : FRED API — SOFR
# ══════════════════════════════════════════════════════════════════════════════

class TestFREDApiKey:
    """Vérifie que la clé FRED est bien configurée."""

    def test_fred_api_key_present(self):
        """La variable d'environnement FRED_API_KEY doit être définie et non vide."""
        assert FRED_API_KEY is not None, (
            "FRED_API_KEY manquante. Vérifier le fichier .env à la racine du projet."
        )
        assert len(FRED_API_KEY.strip()) > 0, "FRED_API_KEY est vide."

    def test_fred_api_key_loaded_in_datafetcher(self):
        """DataFetcher doit charger la clé FRED depuis l'environnement."""
        fetcher = DataFetcher()
        assert fetcher.fred_api_key is not None, (
            "DataFetcher n'a pas chargé FRED_API_KEY. Vérifier load_dotenv()."
        )
        assert fetcher.fred_api_key == FRED_API_KEY


class TestFREDConnectivity:
    """Tests de connectivité brute à l'API FRED."""

    def test_fred_endpoint_reachable(self):
        """L'endpoint FRED doit répondre avec HTTP 200."""
        params = {
            "series_id": SOFR_SERIES_ID,
            "api_key":   FRED_API_KEY,
            "file_type": "json",
            "limit":     1,
            "sort_order": "desc",
        }
        response = requests.get(FRED_BASE_URL, params=params, timeout=10)
        assert response.status_code == 200, (
            f"FRED a répondu avec le code {response.status_code}. "
            "Vérifier la clé API et la connectivité réseau."
        )

    def test_fred_response_is_json(self):
        """La réponse FRED doit être du JSON valide."""
        params = {
            "series_id": SOFR_SERIES_ID,
            "api_key":   FRED_API_KEY,
            "file_type": "json",
            "limit":     1,
            "sort_order": "desc",
        }
        response = requests.get(FRED_BASE_URL, params=params, timeout=10)
        try:
            data = response.json()
        except ValueError:
            pytest.fail("La réponse de l'API FRED n'est pas du JSON valide.")
        assert isinstance(data, dict), "La réponse FRED devrait être un objet JSON (dict)."

    def test_fred_invalid_api_key_returns_error(self):
        """Une clé API invalide doit retourner une erreur de l'API FRED."""
        params = {
            "series_id": SOFR_SERIES_ID,
            "api_key":   "INVALID_KEY_00000",
            "file_type": "json",
            "limit":     1,
        }
        response = requests.get(FRED_BASE_URL, params=params, timeout=10)
        # FRED retourne 200 même en cas d'erreur, mais avec un message d'erreur dans le JSON
        data = response.json()
        # Une clé invalide génère soit un code d'erreur HTTP ≥ 400, soit un champ "error_message"
        has_http_error  = response.status_code >= 400
        has_json_error  = "error_message" in data or "error_code" in data
        assert has_http_error or has_json_error, (
            "L'API FRED aurait dû signaler une erreur pour une clé invalide."
        )


class TestSOFRData:
    """Tests sur les données SOFR retournées par l'API FRED."""

    def _fetch_raw_sofr(self):
        params = {
            "series_id": SOFR_SERIES_ID,
            "api_key":   FRED_API_KEY,
            "file_type": "json",
        }
        response = requests.get(FRED_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def test_sofr_observations_not_empty(self):
        """La série SOFR doit contenir au moins une observation."""
        data = self._fetch_raw_sofr()
        observations = data.get("observations", [])
        assert len(observations) > 0, "Aucune observation SOFR retournée par FRED."

    def test_sofr_latest_observation_has_expected_fields(self):
        """La dernière observation SOFR doit avoir les champs 'date' et 'value'."""
        data   = self._fetch_raw_sofr()
        latest = data["observations"][-1]
        assert "date"  in latest, "Champ 'date' manquant dans l'observation SOFR."
        assert "value" in latest, "Champ 'value' manquant dans l'observation SOFR."

    def test_sofr_date_format_is_valid(self):
        """La date de la dernière observation doit être au format YYYY-MM-DD."""
        data   = self._fetch_raw_sofr()
        latest = data["observations"][-1]
        try:
            parsed_date = datetime.strptime(latest["date"], "%Y-%m-%d")
        except ValueError:
            pytest.fail(f"Format de date SOFR invalide : '{latest['date']}'. Attendu YYYY-MM-DD.")
        # La date ne doit pas être dans le futur
        assert parsed_date.date() <= datetime.today().date(), (
            f"La date SOFR ({latest['date']}) est dans le futur."
        )

    def test_sofr_value_is_numeric(self):
        """La valeur SOFR doit être convertible en float."""
        data   = self._fetch_raw_sofr()
        latest = data["observations"][-1]
        try:
            sofr_float = float(latest["value"])
        except (ValueError, TypeError):
            pytest.fail(f"Impossible de convertir la valeur SOFR en float : '{latest['value']}'.")
        assert sofr_float >= 0, "La valeur SOFR brute ne peut pas être négative."

    def test_sofr_value_in_reasonable_range(self):
        """Le taux SOFR décimalisé doit être dans une fourchette réaliste (0.01 % – 25 %)."""
        data      = self._fetch_raw_sofr()
        latest    = data["observations"][-1]
        sofr_pct  = float(latest["value"])          # en pourcentage (ex: 5.30)
        sofr_dec  = sofr_pct / 100.0               # converti en décimal (ex: 0.053)
        assert SOFR_REASONABLE_MIN <= sofr_dec <= SOFR_REASONABLE_MAX, (
            f"Taux SOFR hors plage réaliste : {sofr_dec:.4f} "
            f"(attendu entre {SOFR_REASONABLE_MIN} et {SOFR_REASONABLE_MAX})."
        )

    def test_sofr_data_not_too_stale(self):
        """Les données SOFR ne doivent pas dater de plus de 10 jours ouvrés (~14 jours calendaires)."""
        data   = self._fetch_raw_sofr()
        latest = data["observations"][-1]
        obs_date = datetime.strptime(latest["date"], "%Y-%m-%d").date()
        age_days = (datetime.today().date() - obs_date).days
        assert age_days <= 14, (
            f"Les données SOFR datent de {age_days} jours (dernière obs. : {latest['date']}). "
            "Possible interruption de publication par FRED."
        )


class TestDataFetcherSOFR:
    """Tests de get_sofr_rate() via DataFetcher."""

    def test_get_sofr_rate_returns_float(self):
        """get_sofr_rate() doit retourner un float non None."""
        fetcher = DataFetcher()
        rate = fetcher.get_sofr_rate()
        assert rate is not None, "get_sofr_rate() a retourné None. Vérifier la clé FRED et la connectivité."
        assert isinstance(rate, float), f"get_sofr_rate() doit retourner un float, reçu : {type(rate)}."

    def test_get_sofr_rate_in_reasonable_range(self):
        """Le taux SOFR de DataFetcher doit être entre 0.01 % et 25 %."""
        fetcher = DataFetcher()
        rate = fetcher.get_sofr_rate()
        assert rate is not None
        assert SOFR_REASONABLE_MIN <= rate <= SOFR_REASONABLE_MAX, (
            f"Taux SOFR DataFetcher hors plage : {rate:.4f}."
        )

    def test_get_sofr_rate_is_decimal_not_percent(self):
        """Le taux retourné doit être décimalisé (ex: 0.053 et non 5.3)."""
        fetcher = DataFetcher()
        rate = fetcher.get_sofr_rate()
        assert rate is not None
        assert rate < 1.0, (
            f"get_sofr_rate() semble retourner un pourcentage ({rate:.4f}) au lieu d'un décimal."
        )

    def test_get_sofr_rate_uses_cache_on_second_call(self):
        """Le second appel doit retourner la même valeur (via le cache)."""
        fetcher = DataFetcher()
        rate1 = fetcher.get_sofr_rate()
        rate2 = fetcher.get_sofr_rate()
        assert rate1 == rate2, (
            f"Les deux appels consécutifs retournent des valeurs différentes : {rate1} vs {rate2}."
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 : yfinance
# ══════════════════════════════════════════════════════════════════════════════

class TestYfinanceConnectivity:
    """Tests de connectivité basique à yfinance."""

    def test_yfinance_ticker_object_created(self):
        """yf.Ticker() doit créer un objet sans exception."""
        try:
            ticker = yf.Ticker(TEST_TICKER_VALID)
        except Exception as e:
            pytest.fail(f"yf.Ticker() a levé une exception : {e}")
        assert ticker is not None

    def test_yfinance_history_returns_dataframe(self):
        """ticker.history() doit retourner un DataFrame pandas non vide pour AAPL."""
        ticker = yf.Ticker(TEST_TICKER_VALID)
        hist   = ticker.history(period="5d")
        assert not hist.empty, (
            f"yfinance n'a retourné aucune donnée pour {TEST_TICKER_VALID}. "
            "Vérifier la connectivité réseau ou la disponibilité de yfinance."
        )

    def test_yfinance_history_has_required_columns(self):
        """Le DataFrame doit contenir les colonnes essentielles : Open, High, Low, Close, Volume."""
        ticker   = yf.Ticker(TEST_TICKER_VALID)
        hist     = ticker.history(period="5d")
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing  = required - set(hist.columns)
        assert not missing, f"Colonnes manquantes dans l'historique yfinance : {missing}."

    def test_yfinance_close_prices_are_positive(self):
        """Tous les prix de clôture doivent être strictement positifs."""
        ticker = yf.Ticker(TEST_TICKER_VALID)
        hist   = ticker.history(period="5d")
        assert (hist["Close"] > 0).all(), "Des prix de clôture nuls ou négatifs détectés."

    def test_yfinance_invalid_ticker_returns_empty(self):
        """Un ticker inexistant ne doit pas lever d'exception mais retourner un DataFrame vide."""
        try:
            ticker = yf.Ticker(TEST_TICKER_INVALID)
            hist   = ticker.history(period="5d")
        except Exception as e:
            pytest.fail(f"yfinance a levé une exception pour un ticker invalide : {e}")
        assert hist.empty, (
            f"yfinance a retourné des données pour le ticker invalide '{TEST_TICKER_INVALID}'."
        )


class TestYfinancePriceData:
    """Tests sur les données de prix retournées par yfinance."""

    def test_history_1y_sufficient_rows(self):
        """L'historique sur 1 an doit contenir au moins 200 lignes (jours de trading)."""
        ticker = yf.Ticker(TEST_TICKER_VALID)
        hist   = ticker.history(period="1y")
        assert len(hist) >= 200, (
            f"Trop peu de données historiques sur 1 an : {len(hist)} lignes."
        )

    def test_history_dates_are_ascending(self):
        """Les dates de l'historique doivent être triées dans l'ordre croissant."""
        ticker = yf.Ticker(TEST_TICKER_VALID)
        hist   = ticker.history(period="1mo")
        dates  = hist.index.tolist()
        assert dates == sorted(dates), "Les dates de l'historique yfinance ne sont pas triées."

    def test_history_no_null_close_prices(self):
        """Aucun prix de clôture ne doit être NaN sur 1 mois."""
        ticker    = yf.Ticker(TEST_TICKER_VALID)
        hist      = ticker.history(period="1mo")
        null_count = hist["Close"].isna().sum()
        assert null_count == 0, f"{null_count} valeur(s) NaN détectée(s) dans les prix de clôture."

    def test_history_latest_date_recent(self):
        """La dernière date de l'historique doit être récente (≤ 5 jours ouvrés ≈ 7 jours calendaires)."""
        ticker     = yf.Ticker(TEST_TICKER_VALID)
        hist       = ticker.history(period="5d")
        last_date  = hist.index[-1].date() if not hist.empty else None
        assert last_date is not None, "Historique vide, impossible de vérifier la fraîcheur des données."
        age = (datetime.today().date() - last_date).days
        assert age <= 7, (
            f"La dernière donnée yfinance date de {age} jours ({last_date}). "
            "yfinance ne répond peut-être plus correctement."
        )


class TestYfinanceInfo:
    """Tests sur les métadonnées retournées par ticker.info."""

    def test_ticker_info_returns_dict(self):
        """ticker.info doit retourner un dictionnaire non vide."""
        ticker = yf.Ticker(TEST_TICKER_VALID)
        info   = ticker.info
        assert isinstance(info, dict), "ticker.info doit être un dict."
        assert len(info) > 0, "ticker.info est vide."

    def test_ticker_info_has_company_name(self):
        """ticker.info doit contenir 'longName' ou 'shortName'."""
        ticker = yf.Ticker(TEST_TICKER_VALID)
        info   = ticker.info
        has_name = info.get("longName") or info.get("shortName")
        assert has_name, "Ni 'longName' ni 'shortName' présent dans ticker.info."

    def test_ticker_info_market_price_positive(self):
        """Le prix de marché retourné par ticker.info doit être positif."""
        ticker = yf.Ticker(TEST_TICKER_VALID)
        info   = ticker.info
        price  = info.get("regularMarketPrice") or info.get("previousClose")
        assert price is not None, "Aucun prix de marché trouvé dans ticker.info."
        assert float(price) > 0, f"Prix de marché non positif : {price}."


class TestDataFetcherYfinance:
    """Tests de get_live_price() et get_historical_volatility() via DataFetcher."""

    def test_get_live_price_valid_ticker(self):
        """get_live_price() doit retourner un float positif pour AAPL."""
        fetcher = DataFetcher()
        price   = fetcher.get_live_price(TEST_TICKER_VALID)
        assert price is not None, (
            f"get_live_price('{TEST_TICKER_VALID}') a retourné None."
        )
        assert isinstance(price, float), f"Le prix doit être un float, reçu : {type(price)}."
        assert price > 0, f"Le prix doit être positif, reçu : {price}."

    def test_get_live_price_invalid_ticker_returns_none(self):
        """get_live_price() doit retourner None pour un ticker inexistant."""
        fetcher = DataFetcher()
        price   = fetcher.get_live_price(TEST_TICKER_INVALID)
        assert price is None, (
            f"get_live_price('{TEST_TICKER_INVALID}') aurait dû retourner None, reçu : {price}."
        )

    def test_get_historical_volatility_valid_ticker(self):
        """get_historical_volatility() doit retourner un float positif pour AAPL."""
        fetcher = DataFetcher()
        vol     = fetcher.get_historical_volatility(TEST_TICKER_VALID, period="1y")
        assert vol is not None, (
            f"get_historical_volatility('{TEST_TICKER_VALID}') a retourné None."
        )
        assert isinstance(vol, float), f"La volatilité doit être un float, reçu : {type(vol)}."
        assert vol > 0, f"La volatilité doit être positive, reçue : {vol}."

    def test_get_historical_volatility_in_realistic_range(self):
        """La volatilité historique annualisée d'AAPL doit être entre 5 % et 200 %."""
        fetcher = DataFetcher()
        vol     = fetcher.get_historical_volatility(TEST_TICKER_VALID, period="1y")
        assert vol is not None
        assert 0.05 <= vol <= 2.0, (
            f"Volatilité historique hors plage réaliste : {vol:.4f} "
            "(attendu entre 0.05 et 2.0)."
        )

    def test_get_historical_volatility_invalid_ticker(self):
        """get_historical_volatility() doit retourner None pour un ticker inexistant."""
        fetcher = DataFetcher()
        vol     = fetcher.get_historical_volatility(TEST_TICKER_INVALID, period="1y")
        assert vol is None, (
            f"get_historical_volatility('{TEST_TICKER_INVALID}') aurait dû retourner None, reçu : {vol}."
        )

    def test_get_company_name_valid_ticker(self):
        """get_company_name() doit retourner une chaîne non vide pour AAPL."""
        fetcher = DataFetcher()
        name    = fetcher.get_company_name(TEST_TICKER_VALID)
        assert name is not None, "get_company_name() a retourné None pour AAPL."
        assert isinstance(name, str) and len(name) > 0, "Le nom de la société doit être une chaîne non vide."

    @pytest.mark.slow
    def test_get_option_chain_valid_ticker(self):
        """get_option_data_chain() doit retourner une chaîne d'options valide pour AAPL."""
        fetcher         = DataFetcher()
        target_date     = datetime.now() + timedelta(days=60)
        chain, exp_date = fetcher.get_option_data_chain(TEST_TICKER_VALID, target_date)
        assert chain is not None, (
            f"get_option_data_chain('{TEST_TICKER_VALID}') a retourné None. "
            "yfinance ne répond pas ou n'a pas de données d'options."
        )
        assert exp_date is not None, "La date d'expiration retournée est None."
        # Vérifier que calls et puts existent
        assert hasattr(chain, "calls"), "L'objet option_chain n'a pas l'attribut 'calls'."
        assert hasattr(chain, "puts"),  "L'objet option_chain n'a pas l'attribut 'puts'."
        assert not chain.calls.empty, "Le DataFrame des calls est vide."
        assert not chain.puts.empty,  "Le DataFrame des puts est vide."


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 : Tests d'intégration croisés
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Tests croisés FRED + yfinance — scénarios de pricing réalistes."""

    def test_sofr_and_live_price_both_available(self):
        """
        Un scénario de pricing complet nécessite à la fois le SOFR et le prix spot.
        Ce test vérifie que les deux sources sont opérationnelles simultanément.
        """
        fetcher = DataFetcher()
        sofr  = fetcher.get_sofr_rate()
        price = fetcher.get_live_price(TEST_TICKER_VALID)
        assert sofr  is not None, "SOFR non disponible — impossible de pricer une option."
        assert price is not None, "Prix spot non disponible — impossible de pricer une option."
        # Les deux valeurs doivent être dans des plages cohérentes
        assert 0 < sofr  < 1,     f"SOFR incohérent : {sofr}."
        assert price > 1,          f"Prix spot incohérent pour AAPL : {price}."

    def test_sofr_and_volatility_both_available(self):
        """
        Les modèles BSM/CRR nécessitent le taux sans risque (SOFR) ET la volatilité.
        Ce test vérifie leur disponibilité conjointe.
        """
        fetcher = DataFetcher()
        sofr = fetcher.get_sofr_rate()
        vol  = fetcher.get_historical_volatility(TEST_TICKER_VALID, period="1y")
        assert sofr is not None, "SOFR non disponible."
        assert vol  is not None, "Volatilité historique non disponible."
        # Cohérence : le ratio vol/sofr doit être > 1 pour la plupart des actions
        assert vol > sofr, (
            f"Volatilité ({vol:.4f}) inférieure au taux sans risque ({sofr:.4f}) — "
            "valeurs suspectes."
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 : marketdata.app — IV historique
# ══════════════════════════════════════════════════════════════════════════════

MARKET_DATA_TOKEN = os.getenv("MARKET_DATA_TOKEN")


class TestOCCSymbolBuilder:
    """Tests pour la construction des symboles OCC standardises."""

    def test_call_symbol(self):
        """Format OCC pour un call AAPL K=300 exp 2026-06-18."""
        symbol = DataFetcher.build_occ_symbol("AAPL", "2026-06-18", 300, "call")
        assert symbol == "AAPL260618C00300000"

    def test_put_symbol(self):
        """Format OCC pour un put AAPL K=150 exp 2026-12-18."""
        symbol = DataFetcher.build_occ_symbol("AAPL", "2026-12-18", 150, "put")
        assert symbol == "AAPL261218P00150000"

    def test_fractional_strike(self):
        """Les strikes avec decimales doivent etre correctement convertis."""
        symbol = DataFetcher.build_occ_symbol("SPY", "2026-06-18", 450.5, "call")
        assert symbol == "SPY260618C00450500"

    def test_ticker_uppercase(self):
        """Le ticker doit etre force en majuscules."""
        symbol = DataFetcher.build_occ_symbol("aapl", "2026-06-18", 300, "call")
        assert symbol.startswith("AAPL")

    def test_small_strike(self):
        """Un strike petit (< 10) doit etre correctement padde."""
        symbol = DataFetcher.build_occ_symbol("F", "2026-06-18", 5, "call")
        assert symbol == "F260618C00005000"


class TestMarketDataToken:
    """Verifie que le token marketdata.app est configure."""

    def test_market_data_token_present(self):
        """MARKET_DATA_TOKEN doit etre defini et non vide dans .env."""
        assert MARKET_DATA_TOKEN is not None, (
            "MARKET_DATA_TOKEN manquant. Verifier le fichier .env."
        )
        assert len(MARKET_DATA_TOKEN.strip()) > 0, "MARKET_DATA_TOKEN est vide."

    def test_market_data_token_loaded_in_datafetcher(self):
        """DataFetcher doit charger le token depuis l'environnement."""
        fetcher = DataFetcher()
        assert fetcher.market_data_token is not None, (
            "DataFetcher n'a pas charge MARKET_DATA_TOKEN."
        )


class TestMarketDataConnectivity:
    """Tests de connectivite a l'API marketdata.app."""

    def test_api_endpoint_reachable(self):
        """L'endpoint marketdata.app doit repondre avec HTTP 200."""
        headers = {"Authorization": f"Bearer {MARKET_DATA_TOKEN}"}
        url = "https://api.marketdata.app/v1/options/expirations/AAPL/"
        response = requests.get(url, headers=headers, timeout=10)
        assert response.status_code == 200, (
            f"marketdata.app a repondu avec le code {response.status_code}."
        )

    def test_api_returns_valid_json(self):
        """La reponse doit etre du JSON valide avec le champ 's'."""
        headers = {"Authorization": f"Bearer {MARKET_DATA_TOKEN}"}
        url = "https://api.marketdata.app/v1/options/expirations/AAPL/"
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        assert data.get("s") == "ok", f"Status inattendu : {data.get('s')}"


class TestMarketDataOptionHistory:
    """Tests de recuperation de l'historique des prix d'options."""

    def test_get_option_history_returns_data(self):
        """get_option_history_marketdata() doit retourner des donnees pour AAPL."""
        fetcher = DataFetcher()
        # on utilise une expiration existante et un strike ATM
        headers = {"Authorization": f"Bearer {MARKET_DATA_TOKEN}"}
        exp_resp = requests.get(
            "https://api.marketdata.app/v1/options/expirations/AAPL/",
            headers=headers, timeout=10,
        )
        expirations = exp_resp.json().get("expirations", [])
        # selection de la premiere expiration a plus de 20 jours
        from datetime import datetime as dt
        today = dt.now()
        target_exp = None
        for exp in expirations:
            exp_date = dt.strptime(exp, "%Y-%m-%d")
            if (exp_date - today).days > 20:
                target_exp = exp
                break

        if target_exp is None:
            pytest.skip("Aucune expiration a plus de 20 jours disponible.")

        result = fetcher.get_option_history_marketdata(
            ticker="AAPL",
            expiration_date=target_exp,
            strike=300,
            option_type="call",
            history_days=30,
        )
        # le contrat peut ne pas avoir d'historique suffisant,
        # mais si des donnees existent elles doivent etre valides
        if result is not None:
            assert len(result["mid"]) > 0, "La liste des prix mid est vide."
            assert len(result["mid"]) == len(result["underlyingPrice"])
            assert len(result["mid"]) == len(result["dte"])
            assert all(p > 0 for p in result["mid"]), "Prix mid negatif detecte."
            assert all(p > 0 for p in result["underlyingPrice"]), "Spot negatif detecte."

    def test_get_option_history_invalid_symbol_returns_none(self):
        """Un symbole inexistant doit retourner None."""
        fetcher = DataFetcher()
        result = fetcher.get_option_history_marketdata(
            ticker="XXXXINVALID",
            expiration_date="2026-06-18",
            strike=100,
            option_type="call",
            history_days=10,
        )
        assert result is None


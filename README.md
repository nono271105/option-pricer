<h1 align="center">
  <img src="https://github.com/user-attachments/assets/8145acf4-0b8c-47e1-afa7-fd7d1b56da96" alt="logo" width="90" style="vertical-align: middle; margin-right: 8px;">
  Option Pricer
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT"></a>
  <img src="https://img.shields.io/badge/status-Active-brightgreen.svg" alt="Active">
  <br>
  <img src="https://img.shields.io/badge/PySide6-41CD52?logo=Qt&logoColor=white" alt="PySide6">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Plotly-%233F4F75.svg?logo=plotly&logoColor=white" alt="Plotly">
  <img src="https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white" alt="pandas">
</p>

<p align="center">
  Application Python complète pour l'évaluation d'options financières vanilles et exotiques.<br>
  Interface PySide6 avec données de marché en temps réel, visualisations intégrées.
</p>

---

## Fonctionnalités

### Modèles de pricing

| Modèle                                                      |            Type            |
| :----------------------------------------------------------- | :------------------------: |
| **Black-Scholes-Merton**                               |        Européenne        |
| **Cox-Ross-Rubinstein**                                |        Américaine        |
| ****Rubinstein & Reiner** (1991) + Monte Carlo** |         Barrières         |
| **Monte Carlo**                                        |         Asiatique         |
| **Monte Carlo**                                        |          Lookback          |
| **BSM fermée + Monte Carlo**                         | Digitale / Cash-or-Nothing |

### Données de marché en temps réel

- **Prix spot** via Yahoo Finance (`yfinance`)
- **Taux sans risque SOFR** depuis l'API FRED
- **Dividendes** et **volatilité implicite** extraits automatiquement depuis les chaînes d'options
- **Historique prix options** via l'API marketdata.app (utilisé pour le forecast IV)
- Cache TTL thread-safe pour limiter les appels API

### Grecs

Delta (Δ), Gamma (Γ), Theta (Θ/jour), Vega (ν), Rho (ρ), calculés analytiquement (BSM) ou par différences finies (CRR)

---

## Interface 8 onglets

### 1 · Calculateur BSM

Pricing européen en temps réel avec récupération automatique de S, r, q et IV marché.

<img width="1440" alt="Onglet BSM" src="https://github.com/user-attachments/assets/30ea8f99-96ac-49a9-ba34-484ecb03efb1"/>

---

### 2 · Modèle CRR

Pricing américain par arbre binomial. Comparaison directe avec le prix BSM européen.

<img width="1391" alt="Onglet CRR" src="https://github.com/user-attachments/assets/8940c09c-5bd6-4a8b-97fb-1cd0458ecfa4"/>

---

### 3 · Simulation

Heatmap croisée volatilité × prix sous-jacent visualise l'impact combiné de Gamma et Vega sur le prix du call.

<img width="1440" alt="Simulation" src="https://github.com/user-attachments/assets/b9099d90-6d76-47b0-b9b6-904ba023f67c"/>

---

### 4 · Smile de volatilité

Tracé IV vs Strike par inversion numérique de BSM (méthode de Brent) sur les prix mid Calls/Puts OTM.
Interpolation spline cubique.

<img width="1440" alt="Smile de volatilité" src="https://github.com/user-attachments/assets/07769499-978b-4687-9b82-67e29c1fcb3b"/>

---

### 5 · Surface IV 3D

Surface de volatilité implicite interactive axes Strike × Maturité × IV.
Interpolation Griddata cubique, export HTML.

<img width="1440" alt="Surface IV 3D" src="https://github.com/user-attachments/assets/5387adeb-0db6-4213-917b-ad3700dd6651"/>

---

### 6 · Options exotiques

Pricing analytique + Monte Carlo pour barrières, asiatiques, lookbacks et digitales.
Trajectoires GBM simulées, distribution des payoffs et profil à maturité.

<img width="1440" height="900" alt="Options Exotiques" src="https://github.com/user-attachments/assets/da66bcd2-fe51-4363-b707-d87c1c668b1a" />

---

### 7 · Stratégies

Construction et analyse de stratégies options multi-legs avec données de marché en temps réel.

**22 stratégies disponibles en 5 familles**

- Positions de base : Long/Short Call, Long/Short Put
- Spreads directionnels : Bull/Bear Call Spread, Bull/Bear Put Spread
- Volatilité : Long/Short Straddle, Long/Short Strangle
- Butterflies : Call/Put/Iron Butterfly (long et short)
- Condors : Call/Put/Iron Condor (long et short)

**Métriques calculées automatiquement** : coût total, breakevens, gain maximum, perte maximum et grecs agrégés BSM (Δ, Γ, Θ, ν, ρ) de tous les legs.

<img width="1440" height="900" alt="Stratégies" src="https://github.com/user-attachments/assets/c0396c3a-7c3c-4074-9223-34216933df86" />

---

### 8 · Forecast IV TimesFM (IA)

Prévision de la **volatilité implicite (IV)** via le modèle de fondation **Google TimesFM (2.5-200M)**.
Historique d'IV recalculé par inversion BSM (Brent) à partir des prix d'options fournis par l'API **marketdata.app**.
Repricing BSM et recalcul du delta jour par jour sur l'horizon de forecast (jusqu'à 63 jours) en injectant l'IV prédite, toutes choses étant égales par ailleurs . Exécution asynchrone (QThread) 100% compatible CPU. 3 graphiques : IV historique + forecast, prix option, delta.


---

## Benchmark vs DerivaGem (John Hull)

<img width="1440" alt="Benchmark" src="https://github.com/user-attachments/assets/ad5af14b-fa06-4c52-97e0-695cae79c2b7"/>

---

## Installation

### Prérequis

- Python 3.8+
- Clé API FRED gratuite --> https://fred.stlouisfed.org/docs/api/api_key.html

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/nono271105/option-pricer.git
cd option-pricer

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Dépendances
pip install -r requirements.txt

# 4. Variables d'environnement (créer un fichier .env à la racine)
FRED_API_KEY=VOTRE-CLÉ                  # API FRED pour le taux SOFR
MARKET_DATA_TOKEN=VOTRE-TOKEN           # API marketdata.app pour l'IV historique

# 5. Lancement
python main.py
```

---

## Structure du projet

```


option_pricer/
├── main.py                           # Point d'entrée
├── gui_app.py                        # Interface PySide6
├── market_data_store.py              # Store pub/sub centralisé : synchronisation entre onglets
├── data_fetcher.py                   # yfinance + FRED API
├── cache.py                          # Cache TTL thread-safe
├── utils.py                          # Utilitaires partagés
│
├── UI/                               # Modules d'interface utilisateur (PySide6)
│   ├── __init__.py
│   ├── bsm_ui.py                     # Onglet Black-Scholes 
│   ├── crr_ui.py                     # Onglet Cox-Ross-Rubinstein 
│   ├── exotic_options_ui.py          # Onglet Options exotiques
│   ├── forecast_ui.py                # Onglet Forecast TimesFM 
│   ├── simulation_ui.py              # Onglet Simulation
│   ├── strategy_ui.py                # Onglet Stratégies multi-legs
│   ├── volatility_smile_ui.py        # Onglet Smile de volatilité
│   └── volatility_surface_ui.py      # Onglet Surface IV 3D
│
├── logic/                            # Modules de logique métier
│   ├── __init__.py  
│   ├── bsm_logic.py                  # Modèle Black-Scholes + Grecs
│   ├── crr_logic.py                  # Modèle Cox-Ross-Rubinstein + Grecs
│   ├── exotic_options_logic.py       # Barrières, Asiatiques, Lookback, Digitales
│   ├── forecast_logic.py             # Moteur TimesFM
│   ├── simulation_logic.py           # Calcul heatmap
│   ├── strategy_logic.py             # Moteur de calcul des 22 stratégies
│   ├── volatility_smile_logic.py     # Calcul IV smile 
│   └── volatility_surface_logic.py   # Surface IV 3D
│
├── tests/                            # Tests de régression
│   ├── conftest.py                   # Fixtures pytest
│   ├── test_app.py                   # Tests interface
│   ├── test_data.py                  # Tests données
│   └── test_pricing.py               # Régression BSM, CRR, Grecs, exotiques
│
├── requirements.txt
└── README.md
```

### Patterns et Architecture

**1. MarketDataStore (Pub/Sub)**
Chaque onglet s'abonne automatiquement aux mises à jour du store centralisé. Les données (S, r, q, σ, ticker) sont synchronisées une seule fois à la source et propagées en temps réel aux 8 onglets.

**2. Séparation UI / Logique**

- `UI/*.py` : interaction PySide6, validation des inputs, affichage graphique
- `logic/*.py` : calculs purs (BSM, CRR, grecs, stratégies), aucune dépendance Qt
- Facilite les tests unitaires et la réutilisabilité du code métier

**3. QThread Worker Pattern**
Calculs lourds (CRR, Monte Carlo, TimesFM, surface IV) exécutés en arrière-plan sans bloquer l'UI.

## Dépendances principales

| Package                     | Usage                                     |
| --------------------------- | ----------------------------------------- |
| `PySide6`                 | Interface graphique (Qt6)                 |
| `yfinance`                | Prix, IV, chaînes d'options              |
| `marketdata.app` (API)    | Historique prix options (IV forecast)     |
| `matplotlib`              | Graphiques 2D                             |
| `plotly`                  | Surface IV 3D interactive                 |
| `scipy`                   | CDF normale, interpolation, optimisation  |
| `numpy`                   | Calcul numérique                         |
| `pandas`                  | Manipulation de données                  |
| `requests`                | API FRED                                  |
| `python-dotenv`           | Variables d'environnement                 |
| `pytest` / `pytest-cov` | Tests de régression                      |
| `timesfm` / `torch`     | Modèle de prévision IA (Google TimesFM) |

---

## Licence

MIT voir [`LICENSE`](LICENSE).

---

*Dernière mise à jour : mai 2026 : forecast IV TimesFM via marketdata.app, repricing BSM à IV prédite (v2.4)*

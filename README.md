<h1 align="center">
  <img src="https://github.com/user-attachments/assets/8145acf4-0b8c-47e1-afa7-fd7d1b56da96" alt="logo" width="90" style="vertical-align: middle; margin-right: 8px;">
  Option Pricer
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT"></a>
  <img src="https://img.shields.io/badge/status-Active-brightgreen.svg" alt="Active">
</p>

<p align="center">
  Application Python complète pour l'évaluation d'options financières vanilles et exotiques.<br>
  Interface PyQt5 avec données de marché en temps réel, visualisations intégrées.
</p>

---

## Fonctionnalités

### Modèles de pricing

| Modèle | Type | Méthode |
|--------|------|---------|
| **Black-Scholes-Merton** | Européenne | Formule fermée |
| **Cox-Ross-Rubinstein** | Américaine | Arbre binomial |
| **Barrières** (8 types) | Exotique | Rubinstein & Reiner (1991) + Monte Carlo |
| **Asiatique** (moyenne) | Exotique | Monte Carlo |
| **Lookback** | Exotique | Monte Carlo |
| **Digitale / Cash-or-Nothing** | Exotique | BSM fermée + Monte Carlo |

### Données de marché en temps réel
- **Prix spot** via Yahoo Finance (`yfinance`)
- **Taux sans risque SOFR** depuis l'API FRED
- **Dividendes** et **volatilité implicite** extraits automatiquement depuis les chaînes d'options
- Cache TTL thread-safe pour limiter les appels API

### Grecs
Delta (Δ), Gamma (Γ), Theta (Θ/jour), Vega (ν), Rho (ρ), calculés analytiquement (BSM) ou par différences finies (CRR)

---

## Interface 7 onglets

### 1 · Calculateur BSM

Pricing européen en temps réel avec récupération automatique de S, r, q et IV marché.

<img width="1440" alt="Onglet BSM" src="https://github.com/user-attachments/assets/30ea8f99-96ac-49a9-ba34-484ecb03efb1"/>

---

### 2 · Modèle CRR

Pricing américain par arbre binomial. Comparaison directe avec le prix BSM européen.

<img width="1391" alt="Onglet CRR" src="https://github.com/user-attachments/assets/8940c09c-5bd6-4a8b-97fb-1cd0458ecfa4"/>

---

### 3 · Simulation matricielle

Heatmap croisée volatilité × prix sous-jacent visualise l'impact combiné de Gamma et Vega sur le prix du call.

<img width="1440" alt="Simulation" src="https://github.com/user-attachments/assets/b9099d90-6d76-47b0-b9b6-904ba023f67c"/>

---

### 4 · Smile de volatilité

Tracé IV vs Strike par inversion numérique de BSM (méthode de Brent) sur les prix mid Calls/Puts OTM.
Interpolation spline cubique.

<img width="1440" alt="Smile de volatilité" src="https://github.com/user-attachments/assets/07769499-978b-4687-9b82-67e29c1fcb3b"/>

---

### 5 · Surface IV 3D (Plotly)

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

# 4. Variables d'environnement
FRED_API_KEY=VOTRE-CLÉ        # Créer un fichier .env et insérer votre clé API

# 5. Vérification
python -c "import PyQt5, yfinance, scipy; print('OK')"

# 6. Lancement
python main.py
```

---

## Structure du projet

```
option_pricer/
├── main.py                       # Point d'entrée
├── gui_app.py                    # Interface PyQt5 7 onglets
├── option_models.py              # BSM, CRR, Grecs
├── exotic_options_models.py      # Barrières, Asiatiques, Lookback, Digitales
├── exotic_options_tab.py         # Onglet options exotiques
├── strategy_manager.py           # Moteur de calcul des stratégies
├── strategy_tab.py               # Onglet stratégies
├── data_fetcher.py               # yfinance + FRED API + cache TTL
├── simulation_tab.py             # Heatmap simulation
├── volatility_smile_tab.py       # Smile de volatilité
├── volatility_surface_tab.py     # Surface IV 3D Plotly
├── implied_volatility_surface.py # Calcul surface IV
├── cache.py                      # Cache TTL thread-safe
├── requirements.txt
└── README.md
```

## Dépendances principales

| Package | Usage |
|---------|------------|
| `PyQt5` | Interface graphique |
| `PyQtWebEngine` | Rendu Plotly |
| `yfinance` | Prix, IV, chaînes d'options |
| `matplotlib` | Graphiques 2D |
| `plotly` | Surface IV 3D interactive |
| `scipy` | CDF normale, interpolation, optimisation |
| `numpy` | Calcul numérique |
| `pandas` | Manipulation de données |
| `requests` | API FRED |
| `python-dotenv` | Variables d'environnement |

---

## Licence

MIT voir [`LICENSE`](LICENSE).

---

*Dernière mise à jour : mars 2026 ajout de l'onglet Stratégies (v2.1)*

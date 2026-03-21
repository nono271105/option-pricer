"""
Configuration pour la Surface de Volatilité Implicite

Ce fichier contient les paramètres configurables pour le calcul et la 
visualisation de la surface IV 3D.
"""

# ═══════════════════════════════════════════════════════════════════
# PARAMÈTRES DE CALCUL
# ═══════════════════════════════════════════════════════════════════

# Nombre d'expirations à récupérer
# Plus de valeurs = plus de détail mais plus lent
NUM_EXPIRATIONS = 20

# Nombre maximum de jours à afficher sur la surface
# Filtre les expirations qui dépassent cette limite
MAX_DAYS_TO_MATURITY = 400

# Nombre de strikes min/max à inclure dans chaque expiration
MIN_STRIKES_REQUIRED = 5
MAX_STRIKES_PER_EXPIRATION = 100

# Filtres de volatilité implicite
IV_MIN_THRESHOLD = 0.001      # Minimum IV (0.1%)
IV_MAX_THRESHOLD = 5.0         # Maximum IV (500%)

# ═══════════════════════════════════════════════════════════════════
# PARAMÈTRES D'INTERPOLATION
# ═══════════════════════════════════════════════════════════════════

# Taille de la grille d'interpolation
STRIKE_GRID_SIZE = 30          # Nombre de points en X (Strike)
MATURITY_GRID_SIZE = 12        # Nombre de points en Y (Maturité)

# Pourcentage de padding autour des limites des données
DATA_PADDING_PERCENT = 0.05    # 5% de padding

# Méthode d'interpolation principale (voir scipy.interpolate.griddata)
INTERPOLATION_METHOD = 'cubic'  # Options: 'linear', 'cubic', 'nearest'

# Fallback si interpolation principale échoue
INTERPOLATION_FALLBACK = 'nearest'

# ═══════════════════════════════════════════════════════════════════
# PARAMÈTRES PLOTLY VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

# Colormap pour les visualisations
COLORMAP = 'Plasma'           # Options: Viridis, Plasma, Inferno, etc.

# Opacité de la surface interpolée
SURFACE_OPACITY = 0.7

# Opacité des points bruts
MARKER_OPACITY = 0

# Taille des marqueurs
MARKER_SIZE = 4

# Dimensions du graphique (en pixels)
PLOT_WIDTH = 1200
PLOT_HEIGHT = 700

# Position de la caméra (eye)
CAMERA_EYE_X = 1.5
CAMERA_EYE_Y = 1.5
CAMERA_EYE_Z = 1.3

# ═══════════════════════════════════════════════════════════════════
# PARAMÈTRES DE CACHE
# ═══════════════════════════════════════════════════════════════════

# Durée de vie du cache en secondes
CACHE_TTL = 1800               # 1/2 heure

# ═══════════════════════════════════════════════════════════════════
# PARAMÈTRES DE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════

# Nombre de workers pour ThreadPoolExecutor
NUM_WORKERS = 4

# Timeout pour les requêtes API (secondes)
API_TIMEOUT = 10

# ═══════════════════════════════════════════════════════════════════
# DONNÉES FINANCIÈRES DÉFAULT
# ═══════════════════════════════════════════════════════════════════

# Taux sans risque par défaut (si FRED API échoue)
DEFAULT_RISK_FREE_RATE = 0.05

# Rendement de dividende par défaut
DEFAULT_DIVIDEND_YIELD = 0.0

# ═══════════════════════════════════════════════════════════════════
# PARAMÈTRES UI
# ═══════════════════════════════════════════════════════════════════

# Largeur minimum du fenêtre de surface
MIN_WINDOW_WIDTH = 1200
MIN_WINDOW_HEIGHT = 700

# Style des étiquettes
LABEL_FONT_SIZE = 12
AXIS_LABEL_FONT_SIZE = 11

# Messages de statut
STATUS_MESSAGES = {
    'retrieving': 'Récupération des données de marché...',
    'computing': 'Calcul de la surface IV...',
    'interpolating': 'Interpolation de la grille...',
    'rendering': 'Rendu du graphique Plotly...',
    'success': 'OK Surface IV calculée ({} points)',
    'error': 'Erreur: {}',
}

# ═══════════════════════════════════════════════════════════════════
# VALIDATION DES ENTRÉES
# ═══════════════════════════════════════════════════════════════════

# Prix minimum acceptable
MIN_PRICE = 0.01

# Prix maximum acceptable
MAX_PRICE = 100000.0

# Regex pour validation du ticker
TICKER_PATTERN = r'^[A-Z0-9]{1,6}$'

# ═══════════════════════════════════════════════════════════════════
# CHEMINS D'EXPORT
# ═══════════════════════════════════════════════════════════════════

# Répertoire de sortie pour les exports HTML (relatif au cwd)
EXPORT_DIRECTORY = '.'

# Préfixe du nom de fichier
EXPORT_FILE_PREFIX = 'iv_surface'

# ═══════════════════════════════════════════════════════════════════
# OPTIONS DE DÉBOGAGE
# ═══════════════════════════════════════════════════════════════════

# Afficher les logs détaillés
DEBUG_MODE = False

# Sauvegarder les données brutes en CSV (pour débogage)
SAVE_RAW_DATA_CSV = False

# Afficher les temps de calcul
SHOW_TIMING = False


# ═══════════════════════════════════════════════════════════════════
# UTILISATION EN CODE
# ═══════════════════════════════════════════════════════════════════
"""
Pour utiliser cette configuration dans le code:

    import iv_surface_config as config
    
    # Accéder aux paramètres
    grid_size_x = config.STRIKE_GRID_SIZE
    grid_size_y = config.MATURITY_GRID_SIZE
    
    # Utiliser les valeurs
    X_grid = np.linspace(min_strike, max_strike, config.STRIKE_GRID_SIZE)
"""

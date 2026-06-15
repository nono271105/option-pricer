# Migration de PySide6 vers React + Eel (Option 2)

L'objectif de cette migration est de remplacer l'interface graphique native `PySide6` par la nouvelle interface web `React` tout en conservant toute ta logique mathématique et financière en Python. Nous allons utiliser **Eel** pour créer cette application hybride "Desktop Moderne", et nous migrerons l'intégralité de la suite de tests UI.

## User Review Required

> [!WARNING]
> **Suppression de PySide6 et des tests associés** : Cette architecture rend le dossier `UI/` et le fichier `tests/test_app.py` complètement obsolètes. Es-tu d'accord pour que nous supprimions tout le code lié à PySide6 (`gui_app.py`, `UI/`, `test_app.py`) au fur et à mesure que nous implémentons l'équivalent côté React ?

> [!IMPORTANT]
> **Framework de tests Frontend** : Pour remplacer les tests Qt, nous allons configurer `Vitest` et `React Testing Library` dans ton projet web. C'est le standard moderne pour tester les composants React de manière robuste.

## Proposed Changes

La migration va se dérouler en quatre grandes phases :

### 1. Le Backend (Python & Eel)

Nous allons séparer la logique pure de l'ancienne interface Qt.

#### [NEW] `api.py`
Création d'un fichier "pont" qui va exposer les fonctions de calcul et de récupération de données au Javascript.
- Importera tes modules existants (`logic.bsm_logic`, `data_fetcher`, etc.).
- Chaque fonction destinée à l'interface React aura le décorateur `@eel.expose`.
- Gérera les exceptions et renverra des dictionnaires propres (JSON-serializable) à React.

#### [MODIFY] `main.py`
Le point d'entrée du programme sera entièrement réécrit.
- Suppression de l'initialisation de `QApplication`.
- Initialisation d'Eel (`eel.init('dist')`) pour pointer vers le dossier de l'application React compilée.
- Démarrage de la fenêtre d'application (`eel.start('index.html', mode='chrome', size=(1400, 900))`).

#### [DELETE] `gui_app.py` et le dossier `UI/`
- Suppression des classes Qt (`OptionPricingApp`, `BSMTab`, `FetchDataWorker`, etc.).

---

### 2. Le Frontend (React & TypeScript)

L'interface que nous avons construite doit maintenant être branchée au moteur Python.

#### [MODIFY] `index.html` (dans le projet Vite)
- Ajout de la balise script magique d'Eel : `<script type="text/javascript" src="/eel.js"></script>`. Cela permettra à React de communiquer avec Python.

#### [MODIFY] Fichiers Composants (`BsmTab.tsx`, `CrrTab.tsx`, `App.tsx`...)
- Remplacement des `PAYOFF_DATA` et données statiques par des variables d'état (Hooks `useState` et `useEffect`).
- Lors d'un clic sur "Calculer Prix", appel de la fonction Python via Javascript : 
  `const resultat = await window.eel.calculate_bsm(S, K, r, q, t, sigma)();`
- Gérer proprement les états de chargement (loaders) lorsque Python "fetch" les données de marché.

---

### 3. Migration des Tests (Le point critique)

Pour garantir qu'il n'y ait aucune friction, la robustesse actuelle doit être transférée au web.

#### [DELETE] `tests/test_app.py`
- Ce fichier, qui teste le comportement des widgets PySide6, sera supprimé. Les fichiers `test_pricing.py` et `test_data.py` resteront intacts car la logique ne change pas.

#### [NEW] Configuration Vitest (Projet React)
- Installation de `vitest`, `@testing-library/react` et `@testing-library/user-event`.
- Configuration de l'environnement de test (JSDOM).

#### [NEW] `src/app/__tests__/` (Suite de tests Frontend)
- Recréation de la logique de `test_app.py` en Javascript/TypeScript.
- **Mock de l'API Python** : Pendant les tests, `window.eel` sera simulé (mocké) pour retourner de fausses données instantanément, ce qui permet de tester l'UI en isolation totale.
- **Tests prévus** : 
  - Vérification du rendu des composants (BSM, CRR, Option Chains).
  - Validation du fonctionnement du clic sur "Récupérer Données" (passage en état "Chargement" puis mise à jour des chiffres).
  - Vérification de l'appel correct des fonctions Eel avec les bons paramètres du formulaire.

---

### 4. Workflow de Développement & Build

Pour que tout soit fluide pendant le développement et facile à distribuer.

#### Configuration du Développement
- Ajout d'une condition dans `main.py` : si on est en mode "développement", Eel pointera vers le serveur Vite (`http://localhost:5173`) au lieu du dossier `dist`. Cela permet d'avoir le *Hot Reload* du design React tout en ayant le backend Python actif.

#### Procédure de Build
1. Lancer `npm run build` dans le projet React.
2. Copier le dossier `dist` dans le dossier de ton projet Python.
3. Exécuter `main.py` (ou packager avec PyInstaller) pour lancer l'application finale.

## Verification Plan

### Automated Tests
- Exécution de `pytest` côté Python pour valider que la refonte de l'API n'a pas cassé le pricing.
- Exécution de `npm run test` côté React pour s'assurer que 100% de la suite de tests UI migrée passe avec succès.

### Manual Verification
- Lancer le serveur hybride Python + React.
- Récupérer les données d'un Ticker (ex: AAPL) et valider la mise à jour complète du Quote Panel.
- Lancer un pricing BSM puis CRR, et vérifier que les graphiques se mettent à jour.

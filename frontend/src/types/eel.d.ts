/**
 * eel.d.ts — Déclarations TypeScript pour window.eel
 *
 * Eel injecte un objet `eel` dans window qui expose les fonctions Python
 * décorées avec @eel.expose. Chaque appel retourne une Promise via la syntaxe :
 *   const result = await window.eel.nom_fonction(args)();
 *
 * La double-invocation `()()` est la signature Eel :
 *   - Premier `()` : construit l'appel côté JS
 *   - Second `()` : retourne une Promise qui résoudra avec la valeur Python
 */

// ── Types de données retournées ────────────────────────────────────────────

export interface MarketData {
  ticker: string;
  company_name: string;
  S: number | null;
  r: number | null;
  q: number | null;
  hist_vol: number | null;
  error: string | null;
}

export interface Greeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}

export interface PricePoint {
  spot: number;
  payoff: number;
}

export interface GreekPoint {
  spot: number;
  value: number | null;
}

export interface BsmResult {
  price: number;
  greeks: Greeks;
  payoff_data: PricePoint[];
  delta_data: GreekPoint[];
  gamma_data: GreekPoint[];
  theta_data: GreekPoint[];
  vega_data: GreekPoint[];
  rho_data: GreekPoint[];
  breakeven: number;
  S: number;
  K: number;
  error: string | null;
}

export interface CrrResult {
  price: number;
  greeks: Greeks;
  payoff_data: PricePoint[];
  delta_data: GreekPoint[];
  gamma_data: GreekPoint[];
  theta_data: GreekPoint[];
  vega_data: GreekPoint[];
  rho_data: GreekPoint[];
  breakeven: number;
  S: number;
  K: number;
  error: string | null;
}

export interface SimulationResult {
  vols: number[];
  prices: number[];
  matrix: number[][];
  error: string | null;
}

export interface SmilePoint {
  strike: number;
  iv: number;
  type: string;
}

export interface SmileResult {
  expiry_used: string;
  strikes_interp: number[];
  ivs_interp: number[];
  raw_data: SmilePoint[];
  current_price: number;
  error: string | null;
}

export interface SurfaceResult {
  strikes: number[];
  maturities: number[];
  iv_surface: (number | null)[][];
  error: string | null;
}

export interface DistPoint {
  bucket: number;
  count: number;
}

export interface ExoticPriceEntry {
  price: number;
  method: string;
  std_error?: number | null;
  ci_95?: [number, number] | null;
}

export interface ExoticResult {
  price: number;
  results: {
    analytical?: ExoticPriceEntry;
    mc?: ExoticPriceEntry & { std_error: number | null; ci_95: [number, number] | null };
  } | null;
  price_paths: number[][] | null;
  payoff_distribution: DistPoint[] | null;
  S: number;
  K: number;
  error: string | null;
}

export interface OptionRow {
  strike: number;
  bid: number;
  ask: number;
  iv: number;
  volume: number;
  oi: number;
  delta: number | null;
}

export interface OptionChainResult {
  expiry_used: string;
  calls: OptionRow[];
  puts: OptionRow[];
  error: string | null;
}

export interface ExpiriesResult {
  expiries: string[];
  error: string | null;
}

export interface Leg {
  option_type: string;
  position: string;
  strike: number;
  premium: number;
}

export interface StrategyMetrics {
  cost: number | null;
  breakevens: number[];
  max_gain: number | null;
  max_loss: number | null;
}

export interface StrategyValuePoint {
  spot: number;
  value: number;
}

export interface StrategyResult {
  strategy_name: string;
  legs: Leg[];
  payoff_data: PricePoint[];
  value_today_data: StrategyValuePoint[];
  metrics: StrategyMetrics;
  greeks: Greeks;
  error: string | null;
}

export interface ForecastResult {
  iv_forecast: (number | null)[];
  option_prices_forecast: (number | null)[];
  deltas_forecast: (number | null)[];
  iv_history: (number | null)[];
  option_prices_history: (number | null)[];
  deltas_history: (number | null)[];
  x_history: number[];
  occ_symbol: string;
  error: string | null;
}

// ── Interface principale de window.eel ─────────────────────────────────────

interface EelAPI {
  // Données de marché
  fetch_market_data(ticker: string): () => Promise<MarketData>;
  get_option_chain(ticker: string, expiry_str: string): () => Promise<OptionChainResult>;
  get_available_expiries(ticker: string): () => Promise<ExpiriesResult>;

  // Modèles de pricing
  calculate_bsm(
    ticker: string, S: number, K: number, maturity_date: string,
    r: number, q: number,
    option_type: string, position: string
  ): () => Promise<BsmResult>;

  calculate_crr(
    ticker: string, S: number, K: number, maturity_date: string,
    r: number, q: number,
    N: number, option_type: string, position: string
  ): () => Promise<CrrResult>;

  // Simulation
  run_simulation(
    K: number, T_days: number, r: number, q: number,
    vol_min: number, vol_max: number, vol_step: number,
    underlying_min: number, underlying_max: number, underlying_step: number,
    option_type: string
  ): () => Promise<SimulationResult>;

  // Volatilité
  calculate_smile(ticker: string, expiry_str: string): () => Promise<SmileResult>;
  calculate_surface(ticker: string): () => Promise<SurfaceResult>;

  // Exotiques
  price_exotic(
    exotic_type: string,
    ticker: string,
    S: number, K: number, maturity_date: string, r: number, q: number,
    option_type: string,
    barrier?: number | null,
    barrier_type?: string,
    averaging?: string,
    payoff_amount?: number,
    n_sims?: number,
    n_steps?: number,
    seed?: number
  ): () => Promise<ExoticResult>;

  // Stratégies
  get_strategy_names(): () => Promise<string[]>;
  calculate_strategy(
    strategy_name: string, ticker: string, S: number, T_days: number,
    r: number, sigma: number, q: number, expiry_str: string
  ): () => Promise<StrategyResult>;

  // Forecast
  run_forecast(
    ticker: string, strike: number, T_days: number,
    option_type: string, expiry_str: string, history_days?: number
  ): () => Promise<ForecastResult>;
}

// ── Extension de l'interface Window ────────────────────────────────────────

declare global {
  interface Window {
    eel: EelAPI;
  }
}

export {};

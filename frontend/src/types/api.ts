/**
 * types/api.ts : Types TypeScript pour les échanges avec l'API FastAPI
 */

// ── Types de données retournées ────────────────────────────────────────────

export interface MarketData {
  ticker: string;
  company_name: string;
  S: number | null;
  r: number | null;
  q: number | null;
  hist_vol: number | null;
  warnings?: string[] | null;
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
  sigma: number;
  sigma_source: string;
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
  sigma: number;
  sigma_source: string;
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
  warning?: string | null;
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
  sigma: number;
  sigma_source: string;
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
  premium_source: string;
  expiry_used: string | null;
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
  sigma: number;
  sigma_source: string;
  error: string | null;
}

export interface CustomStrategyLegInput {
  option_type: 'call' | 'put';
  position: 'long' | 'short';
  strike: number;
  expiry_str: string;
}

export interface CustomStrategyLegResult extends CustomStrategyLegInput {
  expiry: string;
  premium: number;
  premium_source: string;
  expiry_used: string | null;
  sigma: number;
  sigma_source: string;
  pricing_note: string | null;
}

export interface CustomStrategyResult {
  ticker: string;
  legs: CustomStrategyLegResult[];
  initial_net_flow: number;
  current_model_value: number;
  current_model_pnl: number;
  value_today_data: StrategyValuePoint[];
  payoff_data: PricePoint[];
  has_shared_expiry: boolean;
  metrics: StrategyMetrics;
  greeks: Greeks;
  error: string | null;
}

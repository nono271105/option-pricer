/**
 * lib/api.ts : Client API HTTP vers le backend FastAPI.
 *
 * Requêtes fetch typées vers le backend FastAPI.
 * Chaque fonction renvoie l'objet typé attendu, y compris un champ `error`
 * en cas de code HTTP non-2xx ou d'erreur réseau, pour préserver les gardes `if (res.error)`.
 */

import type {
  MarketData,
  OptionChainResult,
  ExpiriesResult,
  BsmResult,
  CrrResult,
  SimulationResult,
  SmileResult,
  SurfaceResult,
  ExoticResult,
  StrategyResult,
} from '../types/api';

const API_BASE = '/api';

async function postJson<T>(endpoint: string, body: unknown): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const errMsg = (data && data.error) ? data.error : `Erreur HTTP ${res.status}`;
      return { ...data, error: errMsg } as T;
    }
    return data as T;
  } catch (err: any) {
    return { error: err?.message || 'Erreur réseau de communication avec le serveur' } as T;
  }
}

async function getJson<T>(endpoint: string): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const errMsg = (data && data.error) ? data.error : `Erreur HTTP ${res.status}`;
      return { ...data, error: errMsg } as T;
    }
    return data as T;
  } catch (err: any) {
    return { error: err?.message || 'Erreur réseau' } as T;
  }
}

// ── 1. Données de marché ─────────────────────────────────────────────────────

export async function fetchMarketData(ticker: string): Promise<MarketData> {
  return postJson<MarketData>('/market-data', { ticker });
}

export async function getAvailableExpiries(ticker: string): Promise<ExpiriesResult> {
  return postJson<ExpiriesResult>('/expiries', { ticker });
}

export async function getOptionChain(
  ticker: string,
  expiry_str: string,
  S?: number | null,
  r?: number | null,
  q?: number | null
): Promise<OptionChainResult> {
  return postJson<OptionChainResult>('/option-chain', {
    ticker,
    expiry_str,
    S: S ?? null,
    r: r ?? null,
    q: q ?? null,
  });
}

// ── 2. Pricing BSM ──────────────────────────────────────────────────────────

export async function calculateBsm(
  ticker: string,
  S: number,
  K: number,
  maturity_date: string,
  r: number,
  q: number,
  option_type: string,
  position: string = 'long',
  hist_vol?: number | null
): Promise<BsmResult> {
  return postJson<BsmResult>('/bsm', {
    ticker,
    S,
    K,
    maturity_date,
    r,
    q,
    option_type,
    position,
    hist_vol: hist_vol ?? null,
  });
}

// ── 3. Pricing CRR ──────────────────────────────────────────────────────────

export async function calculateCrr(
  ticker: string,
  S: number,
  K: number,
  maturity_date: string,
  r: number,
  q: number,
  N: number,
  option_type: string,
  position: string = 'long',
  hist_vol?: number | null
): Promise<CrrResult> {
  return postJson<CrrResult>('/crr', {
    ticker,
    S,
    K,
    maturity_date,
    r,
    q,
    N,
    option_type,
    position,
    hist_vol: hist_vol ?? null,
  });
}

// ── 4. Simulation matricielle BSM ───────────────────────────────────────────

export async function runSimulation(
  K: number,
  T_days: number,
  r: number,
  q: number,
  vol_min: number,
  vol_max: number,
  vol_step: number,
  underlying_min: number,
  underlying_max: number,
  underlying_step: number,
  option_type: string = 'call'
): Promise<SimulationResult> {
  return postJson<SimulationResult>('/simulation', {
    K,
    T_days,
    r,
    q,
    vol_min,
    vol_max,
    vol_step,
    underlying_min,
    underlying_max,
    underlying_step,
    option_type,
  });
}

// ── 5. Volatilité (Smile & Surface) ─────────────────────────────────────────

export async function calculateSmile(
  ticker: string,
  expiry_str: string,
  S?: number | null,
  r?: number | null,
  q?: number | null
): Promise<SmileResult> {
  return postJson<SmileResult>('/smile', {
    ticker,
    expiry_str,
    S: S ?? null,
    r: r ?? null,
    q: q ?? null,
  });
}

export async function calculateSurface(
  ticker: string,
  S?: number | null,
  r?: number | null,
  q?: number | null
): Promise<SurfaceResult> {
  return postJson<SurfaceResult>('/surface', {
    ticker,
    S: S ?? null,
    r: r ?? null,
    q: q ?? null,
  });
}

// ── 6. Options Exotiques ───────────────────────────────────────────────────

export interface PriceExoticParams {
  exotic_type: string;
  ticker: string;
  S: number;
  K: number;
  maturity_date: string;
  r: number;
  q: number;
  option_type: string;
  barrier?: number | null;
  barrier_type?: string;
  averaging?: string;
  payoff_amount?: number;
  n_sims?: number;
  n_steps?: number;
  seed?: number;
  rebate?: number;
  hist_vol?: number | null;
}

export async function priceExotic(params: PriceExoticParams): Promise<ExoticResult> {
  return postJson<ExoticResult>('/exotic', params);
}

// ── 7. Stratégies ───────────────────────────────────────────────────────────

export async function getStrategyNames(): Promise<string[]> {
  const res = await getJson<string[] | { error?: string }>('/strategies');
  if (Array.isArray(res)) return res;
  return [];
}

export async function calculateStrategy(
  strategy_name: string,
  ticker: string,
  S: number,
  T_days: number,
  r: number,
  q: number,
  expiry_str: string,
  hist_vol?: number | null
): Promise<StrategyResult> {
  return postJson<StrategyResult>('/strategy', {
    strategy_name,
    ticker,
    S,
    T_days,
    r,
    q,
    expiry_str,
    hist_vol: hist_vol ?? null,
  });
}


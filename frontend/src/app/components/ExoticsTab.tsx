import React, { useState, useRef, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, ReferenceLine, BarChart, Bar,
  AreaChart, Area,
} from 'recharts';
import { useMarket } from '../App';

// ── Types ────────────────────────────────────────────────────────────────

type ExoticType =
  | 'barrier_analytical' | 'barrier_mc'
  | 'asian_mc' | 'lookback_mc'
  | 'digital_analytical' | 'digital_mc';

type BarrierType = 'down-and-out' | 'down-and-in' | 'up-and-out' | 'up-and-in';

interface PathPoint { step: number; [key: string]: number | null; }
interface DistPoint { bucket: number; count: number; }
interface PayoffPoint { spot: number; payoff: number; }

interface ExoticState {
  price: number | null;
  method: string | null;
  std_error: number | null;
  ci_95: [number, number] | null;
  price_paths: PathPoint[];
  payoff_distribution: DistPoint[];
  payoff_profile: PayoffPoint[];
  S: number | null;
  K: number | null;
  loading: boolean;
  error: string | null;
}

// ── Composant principal ───────────────────────────────────────────────────

export function ExoticsTab() {
  const market = useMarket();
  const tickerRef     = useRef<HTMLInputElement>(null);
  const exoticTypeRef = useRef<HTMLSelectElement>(null);
  const optTypeRef    = useRef<HTMLSelectElement>(null);
  const strikeRef     = useRef<HTMLInputElement>(null);
  const maturityRef   = useRef<HTMLInputElement>(null);
  const sigmaRef      = useRef<HTMLInputElement>(null);
  const barrierRef    = useRef<HTMLInputElement>(null);
  const barrierTypeRef = useRef<HTMLSelectElement>(null);
  const avgRef        = useRef<HTMLSelectElement>(null);
  const payoffAmtRef  = useRef<HTMLInputElement>(null);
  const nSimsRef      = useRef<HTMLInputElement>(null);
  const nStepsRef     = useRef<HTMLInputElement>(null);

  const [state, setState] = useState<ExoticState>({
    price: null, method: null, std_error: null, ci_95: null,
    price_paths: [], payoff_distribution: [], payoff_profile: [],
    S: null, K: null, loading: false, error: null,
  });

  const [exoticType, setExoticType] = useState<ExoticType>('digital_analytical');
  const showBarrier = exoticType.startsWith('barrier');
  const showAveraging = exoticType === 'asian_mc';
  const showPayoffAmt = exoticType.includes('digital');
  const isMC = exoticType.endsWith('_mc');

  const handleFetchData = async () => {
    const ticker = tickerRef.current?.value.trim().toUpperCase() || market.ticker;
    setState(s => ({ ...s, loading: true, error: null }));
    await market.fetchMarket(ticker);
    setState(s => ({ ...s, loading: false }));
  };

  const handleCalculate = useCallback(async () => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const S = market.S ?? 100;
      const K = parseFloat(strikeRef.current?.value || String(Math.round(S)));
      const sigma = parseFloat(sigmaRef.current?.value || '20') / 100;
      const matStr = maturityRef.current?.value || '';
      const T_days = matStr ? computeDaysFromDate(matStr) : 90;
      const optType = optTypeRef.current?.value || 'call';
      const barrierVal = showBarrier ? parseFloat(barrierRef.current?.value || '0') : null;
      const barrierType = (barrierTypeRef.current?.value || 'down-and-out') as BarrierType;
      const averaging = avgRef.current?.value || 'arithmetic';
      const payoffAmt = parseFloat(payoffAmtRef.current?.value || '1');
      const nSims = parseInt(nSimsRef.current?.value || '50000');
      const nSteps = parseInt(nStepsRef.current?.value || '252');

      if (!window.eel) {
        setState(s => ({ ...s, loading: false, error: 'Eel non disponible' }));
        return;
      }

      const res = await window.eel.price_exotic(
        exoticType, S, K, T_days, market.r, sigma, market.q, optType,
        barrierVal, barrierType, averaging, payoffAmt, nSims, nSteps, 42
      )();

      if (res.error) {
        setState(s => ({ ...s, loading: false, error: res.error }));
        return;
      }

      // Transform price_paths for chart: [{step, path_0, ..., avg}]
      let pathData: PathPoint[] = [];
      if (res.price_paths) {
        const n = res.price_paths[0]?.length || 0;
        for (let i = 0; i < n; i++) {
          const pt: PathPoint = { step: i };
          let sum = 0;
          res.price_paths.forEach((p, idx) => {
            pt[`path_${idx}`] = p[i] ?? null;
            sum += p[i] ?? 0;
          });
          pt['avg'] = sum / (res.price_paths.length || 1);
          pathData.push(pt);
        }
      }

      // Payoff profile
      const profileData: PayoffPoint[] = (res.payoff_distribution ?? []).map(d => ({
        spot: d.bucket,
        payoff: d.count,
      }));

      setState(s => ({
        ...s, loading: false, error: null,
        price: res.price,
        method: res.method,
        std_error: res.std_error,
        ci_95: res.ci_95,
        price_paths: pathData,
        payoff_distribution: res.payoff_distribution ?? [],
        S: res.S,
        K: res.K,
      }));
    } catch (e: any) {
      setState(s => ({ ...s, loading: false, error: String(e) }));
    }
  }, [market, exoticType, showBarrier]);

  const numPaths = state.price_paths.length > 0 ? Object.keys(state.price_paths[0]).filter(k => k.startsWith('path_')).length : 0;
  const defaultStrike = market.S ? Math.round(market.S) : 100;
  const defaultSigma = (market.histVol * 100).toFixed(2);
  const defaultBarrier = market.S ? Math.round(market.S * 0.85) : 85;

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">

      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] rounded shrink-0">
          ⚠ {state.error}
        </div>
      )}

      {/* Top row */}
      <div className="flex gap-1">

        {/* PARAMÈTRES */}
        <div className="border border-[#222222] flex-shrink-0 w-[420px] flex flex-col">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white">▼ PARAMÈTRES EXOTIQUES</span>
          </div>
          <div className="bg-[#000000] p-1.5 space-y-1.5">
            <FR label="Ticker">
              <input ref={tickerRef} defaultValue={market.ticker} className={INP} />
            </FR>
            <FR label="Type exotique">
              <select ref={exoticTypeRef} value={exoticType}
                onChange={e => setExoticType(e.target.value as ExoticType)}
                className={SEL}>
                <option value="digital_analytical">Digitale (Analytique)</option>
                <option value="digital_mc">Digitale (Monte Carlo)</option>
                <option value="barrier_analytical">Barrière (Analytique)</option>
                <option value="barrier_mc">Barrière (Monte Carlo)</option>
                <option value="asian_mc">Asiatique (Monte Carlo)</option>
                <option value="lookback_mc">Lookback (Monte Carlo)</option>
              </select>
            </FR>
            <FR label="Type d'option">
              <select ref={optTypeRef} defaultValue="call" className={SEL}>
                <option value="call">call</option>
                <option value="put">put</option>
              </select>
            </FR>
            <FR label="Strike (K)">
              <input ref={strikeRef} key={defaultStrike} defaultValue={defaultStrike}
                type="number" step="0.5" className={INP} />
            </FR>
            <FR label="Date d'échéance">
              <input ref={maturityRef} defaultValue={getDefaultMaturity()} type="date" className={INP} />
            </FR>
            <FR label="Volatilité σ (%)">
              <input ref={sigmaRef} key={defaultSigma} defaultValue={defaultSigma}
                type="number" step="0.01" className={INP} />
            </FR>

            {/* Barrière */}
            {showBarrier && (
              <>
                <FR label="Niveau barrière (H)">
                  <input ref={barrierRef} key={defaultBarrier} defaultValue={defaultBarrier}
                    type="number" step="0.5" className={INP} />
                </FR>
                <FR label="Type barrière">
                  <select ref={barrierTypeRef} defaultValue="down-and-out" className={SEL}>
                    <option value="down-and-out">Down-and-Out</option>
                    <option value="down-and-in">Down-and-In</option>
                    <option value="up-and-out">Up-and-Out</option>
                    <option value="up-and-in">Up-and-In</option>
                  </select>
                </FR>
              </>
            )}

            {/* Asiatique */}
            {showAveraging && (
              <FR label="Moyenne">
                <select ref={avgRef} defaultValue="arithmetic" className={SEL}>
                  <option value="arithmetic">Arithmétique</option>
                  <option value="geometric">Géométrique</option>
                </select>
              </FR>
            )}

            {/* Digital */}
            {showPayoffAmt && (
              <FR label="Montant payoff ($)">
                <input ref={payoffAmtRef} defaultValue="1" type="number" step="0.1" className={INP} />
              </FR>
            )}

            {/* Monte Carlo */}
            {isMC && (
              <>
                <p className="text-[9px] text-[#4A90E2] font-bold uppercase tracking-widest pt-1">Monte Carlo</p>
                <FR label="Simulations">
                  <input ref={nSimsRef} defaultValue="50000" type="number" step="10000" className={INP} />
                </FR>
                <FR label="Pas de temps">
                  <input ref={nStepsRef} defaultValue="252" type="number" step="1" className={INP} />
                </FR>
              </>
            )}

            <div className="pt-2 flex gap-1">
              <button id="exotic-fetch-btn" onClick={handleFetchData} disabled={state.loading}
                className="flex-1 bg-[#2A2A2A] border border-[#444444] text-white py-1 hover:bg-[#3A3A3A] text-[10px] rounded-sm disabled:opacity-50">
                {state.loading ? '⏳...' : 'Récupérer Données'}
              </button>
              <button id="exotic-calc-btn" onClick={handleCalculate} disabled={state.loading}
                className="flex-1 bg-[#4A90E2] text-white py-1 hover:bg-[#357ABD] text-[10px] font-bold rounded-sm disabled:opacity-50">
                {state.loading ? '⏳ Calcul...' : 'Calculer'}
              </button>
            </div>
          </div>
        </div>

        {/* Droite: données + résultats */}
        <div className="flex-1 flex flex-col gap-1">

          {/* DONNÉES MARCHÉ */}
          <div className="border border-[#222222]">
            <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
              <span className="font-bold text-white">▼ DONNÉES MARCHÉ</span>
            </div>
            <div className="bg-[#000000] p-1.5 grid grid-cols-2 gap-x-4 gap-y-1">
              <DR label="Prix Actuel (S)" value={market.S ? `${market.S.toFixed(2)} $` : 'N/C'} />
              <DR label="Taux SOFR (r)"   value={`${(market.r * 100).toFixed(2)}%`} />
              <DR label="Dividende (q)"   value={`${(market.q * 100).toFixed(2)}%`} />
              <DR label="Vol. Hist. (σ₀)" value={`${(market.histVol * 100).toFixed(2)}%`} />
            </div>
          </div>

          {/* RÉSULTATS */}
          <div className="border border-[#222222]">
            <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
              <span className="font-bold text-white">▼ RÉSULTATS PRICING</span>
            </div>
            <div className="bg-[#000000] p-1.5 space-y-1">
              <DR label="Prix" value={state.price !== null ? `${state.price.toFixed(4)} $` : 'N/C'} highlight={state.price !== null} />
              <DR label="Méthode" value={state.method ?? 'N/C'} />
              {state.std_error !== null && (
                <DR label="Std Error MC" value={`±${state.std_error.toFixed(4)}`} />
              )}
              {state.ci_95 && (
                <DR label="IC 95%" value={`[${state.ci_95[0].toFixed(4)}, ${state.ci_95[1].toFixed(4)}]`} />
              )}
            </div>
          </div>

          {/* Monte Carlo paths */}
          {state.price_paths.length > 0 && (
            <div className="border border-[#222222] flex flex-col flex-1 min-h-[180px]">
              <div className="flex items-center justify-between px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
                <span className="font-bold text-white">▼ TRAJECTOIRES GBM</span>
                <div className="flex gap-3 text-[9px] text-[#888888]">
                  <span className="flex items-center gap-1"><span className="text-[#FFCC00]">—</span> Moyenne</span>
                  <span className="flex items-center gap-1"><span className="text-[#4A90E2]">—</span> Paths</span>
                </div>
              </div>
              <div className="flex-1 bg-[#0A0A0A] p-1">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={state.price_paths} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                    <CartesianGrid stroke="#1A1A1A" vertical={false} />
                    <XAxis dataKey="step" stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }} />
                    <YAxis stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }} domain={['auto', 'auto']} />
                    {state.S && <ReferenceLine y={state.S} stroke="#FF4444" strokeDasharray="4 4" />}
                    {state.K && <ReferenceLine y={state.K} stroke="#888888" strokeDasharray="2 4" />}
                    {Array.from({ length: numPaths }).map((_, i) => (
                      <Line key={i} type="monotone" dataKey={`path_${i}`}
                        stroke="#4A90E2" strokeOpacity={0.15} strokeWidth={1}
                        dot={false} isAnimationActive={false} />
                    ))}
                    <Line type="monotone" dataKey="avg" stroke="#FFCC00" strokeWidth={2}
                      dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom: Distribution payoffs */}
      {state.payoff_distribution.length > 0 && (
        <div className="flex gap-1 flex-1 min-h-[180px]">
          <div className="flex-1 border border-[#222222] flex flex-col">
            <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
              <span className="font-bold text-white">▼ DISTRIBUTION DES PAYOFFS</span>
            </div>
            <div className="flex-1 bg-[#0A0A0A] p-1">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={state.payoff_distribution} margin={{ top: 5, right: 5, left: -20, bottom: 15 }}>
                  <CartesianGrid stroke="#1A1A1A" vertical={false} />
                  <XAxis dataKey="bucket" stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }}
                    label={{ value: 'Payoff ($)', position: 'insideBottom', offset: -10, fill: '#888888', fontSize: 9 }} />
                  <YAxis stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }} />
                  <Bar dataKey="count" fill="#4A90E2" opacity={0.6} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* État vide */}
      {state.price === null && !state.loading && !state.error && (
        <div className="flex-1 flex items-center justify-center text-[#888888] text-[12px]">
          Configurez les paramètres et cliquez sur "Calculer"
        </div>
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────

const INP = "w-[120px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none";
const SEL = "w-[120px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none appearance-none cursor-pointer";

function FR({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-[#222222] pb-1">
      <span className="text-[10px] text-[#888888]">{label}</span>
      {children}
    </div>
  );
}

function DR({ label, value, highlight = false }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-[#1A1A1A] pb-1">
      <span className="text-[10px] text-[#888888]">{label}</span>
      <span className={`text-[11px] font-bold ${highlight ? 'text-[#4A90E2]' : 'text-[#FFFFFF]'}`}>{value}</span>
    </div>
  );
}

function computeDaysFromDate(dateStr: string): number {
  const today = new Date();
  const target = new Date(dateStr);
  return Math.max(Math.round((target.getTime() - today.getTime()) / (1000 * 60 * 60 * 24)), 1);
}

function getDefaultMaturity(): string {
  const d = new Date();
  d.setDate(d.getDate() + 90);
  return d.toISOString().split('T')[0];
}
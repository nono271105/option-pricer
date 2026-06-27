import React, { useState, useRef } from 'react';
import Plot from 'react-plotly.js';
import { useMarket } from '../App';

type GreekKey = 'delta' | 'gamma' | 'theta' | 'vega' | 'rho';
interface Greeks { delta: number; gamma: number; theta: number; vega: number; rho: number; }
interface PricePoint { spot: number; payoff: number; }
interface GreekPoint { spot: number; value: number | null; }

interface CrrState {
  price: number | null;
  greeks: Greeks | null;
  payoff_data: PricePoint[];
  allGreekData: Record<GreekKey, GreekPoint[]>;
  activeGreekData: GreekPoint[];
  activeGreek: GreekKey;
  breakeven: number | null;
  S: number | null;
  K: number | null;
  sigma?: number;
  sigma_source?: string;
  loading: boolean;
  error: string | null;
}

export function CrrTab() {
  const market = useMarket();
  const tickerRef  = useRef<HTMLInputElement>(null);
  const optTypeRef = useRef<HTMLSelectElement>(null);
  const strikeRef  = useRef<HTMLInputElement>(null);
  const maturityRef = useRef<HTMLInputElement>(null);
  const positionRef = useRef<HTMLSelectElement>(null);

  const stepsRef   = useRef<HTMLInputElement>(null);

  const [state, setState] = useState<CrrState>({
    price: null, greeks: null,
    payoff_data: [],
    allGreekData: { delta: [], gamma: [], theta: [], vega: [], rho: [] },
    activeGreekData: [],
    activeGreek: 'delta',
    breakeven: null, S: null, K: null,
    loading: false, error: null,
  });

  const handleFetchData = async () => {
    const ticker = tickerRef.current?.value.trim().toUpperCase() || market.ticker;
    setState(s => ({ ...s, loading: true, error: null }));
    await market.fetchMarket(ticker);
    setState(s => ({ ...s, loading: false }));
  };

  const handleCalculate = async () => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const ticker = tickerRef.current?.value.trim().toUpperCase() || market.ticker;
      const S = market.S ?? 100;
      const K = parseFloat(strikeRef.current?.value || String(Math.round(S)));
      const matStr = maturityRef.current?.value || '';
      const N = parseInt(stepsRef.current?.value || '200');
      const optType = optTypeRef.current?.value || 'call';
      const position = positionRef.current?.value || 'long';

      if (!window.eel) {
        setState(s => ({ ...s, loading: false, error: 'Eel non disponible' }));
        return;
      }
      const res = await window.eel.calculate_crr(ticker, S, K, matStr, market.r, market.q, N, optType, position)();

      if (res.error) {
        setState(s => ({ ...s, loading: false, error: res.error }));
        return;
      }
      const greekKey = state.activeGreek;
      const allGreekData: Record<GreekKey, GreekPoint[]> = {
        delta: (res as any).delta_data || [],
        gamma: (res as any).gamma_data || [],
        theta: (res as any).theta_data || [],
        vega:  (res as any).vega_data  || [],
        rho:   (res as any).rho_data   || [],
      };
      setState(s => ({
        ...s, loading: false, error: null,
        price: res.price, greeks: res.greeks,
        payoff_data: res.payoff_data,
        allGreekData,
        activeGreekData: allGreekData[greekKey],
        breakeven: res.breakeven, S: res.S, K: res.K,
        sigma: (res as any).sigma, sigma_source: (res as any).sigma_source,
      }));
    } catch (e: any) {
      setState(s => ({ ...s, loading: false, error: String(e) }));
    }
  };

  // Changement de grec affiché — synchrone, toutes les données sont déjà en state
  const switchGreek = (key: GreekKey) => {
    setState(s => ({
      ...s,
      activeGreek: key,
      activeGreekData: s.allGreekData[key],
    }));
  };

  const fmtG = (v?: number, d = 4) => v !== undefined ? v.toFixed(d) : 'N/C';
  // Le vega du backend est en unités de 1% de vol, on divise par 100
  const fmtVega = (v?: number) => v !== undefined ? (v / 100).toFixed(4) : 'N/C';
  const gc = (v?: number) => !v ? 'text-[#888888]' : v >= 0 ? 'text-[#00FF00]' : 'text-[#FF3333]';
  const defaultStrike = market.S ? Math.round(market.S) : 290;
  const defaultMat    = getDefaultMaturity();

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">
      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] rounded">Warning {state.error}</div>
      )}

      <div className="flex gap-1">
        {/* PARAMÈTRES CRR */}
        <div className="border border-[#222222] flex-shrink-0 w-[420px]">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white"> PARAMÈTRES CRR (Binomial Américain)</span>
          </div>
          <div className="bg-[#000000] p-1.5 space-y-1.5">
            <FR label="Ticker">
              <input ref={tickerRef} defaultValue={market.ticker} className={INPUT} />
            </FR>
            <FR label="Type d'option">
              <select ref={optTypeRef} defaultValue="call" className={SELECT}>
                <option value="call">call</option>
                <option value="put">put</option>
              </select>
            </FR>
            <FR label="Prix d'exercice (K)">
              <input ref={strikeRef} key={defaultStrike} defaultValue={defaultStrike} type="number" step="0.5" className={INPUT} />
            </FR>
            <FR label="Date d'échéance">
              <input ref={maturityRef} defaultValue={defaultMat} type="date" className={INPUT} />
            </FR>
            <FR label="Position">
              <select ref={positionRef} defaultValue="long" className={SELECT}>
                <option value="long">long</option>
                <option value="short">short</option>
              </select>
            </FR>
            <FR label="Pas de l'arbre (N)">
              <input ref={stepsRef} defaultValue="200" type="number" step="50" min="10" max="1000" className={INPUT} />
            </FR>

            <div className="pt-2 flex gap-1">
              <button id="crr-fetch-btn" onClick={handleFetchData} disabled={state.loading}
                className="flex-1 bg-[#2A2A2A] border border-[#444444] text-white py-1 hover:bg-[#3A3A3A] text-[10px] rounded-sm disabled:opacity-50">
                {state.loading ? 'Chargement...' : 'Récupérer Données'}
              </button>
              <button id="crr-calc-btn" onClick={handleCalculate} disabled={state.loading}
                className="flex-1 bg-[#4A90E2] text-white py-1 hover:bg-[#357ABD] text-[10px] font-bold rounded-sm disabled:opacity-50">
                {state.loading ? 'Calcul...' : 'Calculer Prix CRR'}
              </button>
            </div>
          </div>
        </div>

        {/* Droite : Données + Grecs */}
        <div className="flex-1 flex flex-col gap-1">
          <div className="border border-[#222222]">
            <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
              <span className="font-bold text-white"> DONNÉES MARCHÉ</span>
            </div>
            <div className="bg-[#000000] p-1.5 grid grid-cols-2 gap-x-4 gap-y-1.5">
              <DR label="Prix Actuel (S)" value={market.S ? `${market.S.toFixed(2)} $` : 'N/C'} />
              <DR label="Taux SOFR (r)"   value={`${(market.r * 100).toFixed(2)}%`} />
              <DR label="Dividende (q)"   value={`${(market.q * 100).toFixed(2)}%`} />
              <DR label={`Volatilité (σ) [${state.sigma_source || 'N/A'}]`} value={state.sigma ? `${(state.sigma * 100).toFixed(2)}%` : 'N/A'} />
              <DR label="Prix CRR (Américain)"
                value={state.price !== null ? `${state.price.toFixed(4)} $` : 'N/C'}
                highlight={state.price !== null} />
              {state.breakeven !== null && <DR label="Point Mort" value={`${state.breakeven.toFixed(2)} $`} />}
            </div>
          </div>

          {/* Grecs */}
          <div className="border border-[#222222]">
            <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
              <span className="font-bold text-white"> GRECS (CRR  différences finies)</span>
              <span className="ml-2 text-[#888888] text-[9px]">Cliquer pour courbe ↓</span>
            </div>
            <div className="bg-[#000000] overflow-auto">
              <table className="w-full text-right border-collapse text-[11px]">
                <thead>
                  <tr className="bg-[#111111] text-[9px] uppercase text-[#888888] divide-x divide-[#222222] border-b border-[#222222]">
                    {(['delta', 'gamma', 'theta', 'vega', 'rho'] as GreekKey[]).map(g => (
                      <th key={g} id={`crr-greek-${g}`} onClick={() => switchGreek(g)}
                        className={`py-1 px-2 font-normal cursor-pointer select-none transition-colors
                          ${state.activeGreek === g ? 'bg-[#4A90E2] text-white' : 'hover:bg-[#222222]'}`}>
                        {g.charAt(0).toUpperCase() + g.slice(1)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr className="divide-x divide-[#222222]">
                    <td className={`py-1.5 px-2 ${gc(state.greeks?.delta)}`}>{fmtG(state.greeks?.delta)}</td>
                    <td className="py-1.5 px-2 text-[#D4D4D4]">{fmtG(state.greeks?.gamma)}</td>
                    <td className={`py-1.5 px-2 ${gc(state.greeks?.theta)}`}>{fmtG(state.greeks?.theta)}</td>
                    <td className={`py-1.5 px-2 ${gc(state.greeks?.vega)}`}>{fmtVega(state.greeks?.vega)}</td>
                    <td className="py-1.5 px-2 text-[#D4D4D4]">{fmtG(state.greeks?.rho)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Graphiques */}
      <div className="flex gap-1 flex-1 min-h-[250px]">
        <div className="flex-1 border border-[#222222] flex flex-col">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white"> ÉVOLUTION DU {state.activeGreek.toUpperCase()} (CRR)</span>
          </div>
          <div className="bg-[#0A0A0A] flex-1 p-2">
            {state.activeGreekData.length === 0 ? (
              <div className="flex items-center justify-center h-full text-[#888888] text-[11px]">Calculer pour afficher</div>
            ) : (
              <Plot
                data={[
                  {
                    x: state.activeGreekData.map(d => d.spot),
                    y: state.activeGreekData.map(d => d.value),
                    type: 'scatter' as const,
                    mode: 'lines' as const,
                    line: { color: '#4A90E2', width: 1.5 }
                  }
                ]}
                layout={{
                  autosize: true,
                  margin: { l: 40, r: 20, t: 10, b: 30 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  xaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 } },
                  yaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 } },
                  shapes: [
                    ...(state.S ? [{
                      type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: state.S, x1: state.S,
                      y0: 0, y1: 1,
                      line: { color: '#FF4444', dash: 'dash' as const, width: 1 }
                    }] : [])
                  ]
                }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
              />
            )}
          </div>
        </div>

        <div className="flex-1 border border-[#222222] flex flex-col">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white"> PAYOFF DE L'OPTION (CRR)</span>
          </div>
          <div className="bg-[#0A0A0A] flex-1 p-2">
            {state.payoff_data.length === 0 ? (
              <div className="flex items-center justify-center h-full text-[#888888] text-[11px]">Calculer pour afficher</div>
            ) : (
              <Plot
                data={[
                  {
                    x: state.payoff_data.map(d => d.spot),
                    y: state.payoff_data.map(d => d.payoff),
                    type: 'scatter' as const,
                    mode: 'lines' as const,
                    line: { color: '#00FF00', width: 1.5 },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(0, 255, 0, 0.1)'
                  }
                ]}
                layout={{
                  autosize: true,
                  margin: { l: 40, r: 20, t: 10, b: 30 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  xaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 } },
                  yaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 } },
                  shapes: [
                    {
                      type: 'line', xref: 'paper', x0: 0, x1: 1,
                      y0: 0, y1: 0,
                      line: { color: '#444444', width: 1 }
                    },
                    ...(state.K ? [{
                      type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: state.K, x1: state.K,
                      y0: 0, y1: 1,
                      line: { color: '#888888', dash: 'dash' as const, width: 1 }
                    }] : []),
                    ...(state.breakeven ? [{
                      type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: state.breakeven, x1: state.breakeven,
                      y0: 0, y1: 1,
                      line: { color: '#D0D0D0', dash: 'dash' as const, width: 1 }
                    }] : [])
                  ],
                  annotations: [
                    ...(state.K ? [{
                      x: state.K, y: 1, xref: 'x' as const, yref: 'paper' as const,
                      text: `K=${state.K}`, showarrow: false, yanchor: 'bottom', font: { color: '#888888', size: 9 }
                    }] : []),
                    ...(state.breakeven ? [{
                      x: state.breakeven, y: 1, xref: 'x' as const, yref: 'paper' as const,
                      text: `BE=${state.breakeven.toFixed(2)}`, showarrow: false, yanchor: 'bottom', font: { color: '#D0D0D0', size: 9 }
                    }] : [])
                  ]
                }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const INPUT = "w-[120px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none";
const SELECT = "w-[120px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none appearance-none cursor-pointer";

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
    <div className="flex items-center justify-between gap-2 border-b border-[#222222] pb-1">
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

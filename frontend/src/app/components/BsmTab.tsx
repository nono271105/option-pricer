import React, { useState, useRef } from 'react';
import Plot from 'react-plotly.js';
import { useMarket } from '../App';

// ── Types ────────────────────────────────────────────────────────────────

interface Greeks { delta: number; gamma: number; theta: number; vega: number; rho: number; }
interface PricePoint { spot: number; payoff: number; }
interface GreekPoint { spot: number; value: number | null; }

type GreekKey = 'delta' | 'gamma' | 'theta' | 'vega' | 'rho';

interface BsmState {
  price: number | null;
  greeks: Greeks | null;
  payoff_data: PricePoint[];
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

// ── Composant principal ───────────────────────────────────────────────────

export function BsmTab() {
  const market = useMarket();

  // Formulaire (refs pour lire les valeurs sans re-render)
  const tickerRef = useRef<HTMLInputElement>(null);
  const optTypeRef = useRef<HTMLSelectElement>(null);
  const strikeRef = useRef<HTMLInputElement>(null);
  const maturityRef = useRef<HTMLInputElement>(null);
  const positionRef = useRef<HTMLSelectElement>(null);


  const [state, setState] = useState<BsmState>({
    price: null, greeks: null,
    payoff_data: [], activeGreekData: [],
    activeGreek: 'delta',
    breakeven: null, S: null, K: null,
    loading: false, error: null,
  });

  // ── Récupération des données de marché ──────────────────────────────

  const handleFetchData = async () => {
    const ticker = tickerRef.current?.value.trim().toUpperCase() || market.ticker;
    setState(s => ({ ...s, loading: true, error: null }));
    await market.fetchMarket(ticker);
    setState(s => ({ ...s, loading: false }));
  };

  // ── Calcul BSM ───────────────────────────────────────────────────────

  const handleCalculate = async () => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const ticker = tickerRef.current?.value.trim().toUpperCase() || market.ticker;
      const S = market.S ?? 100;
      const K = parseFloat(strikeRef.current?.value || String(Math.round(S)));
      const matStr = maturityRef.current?.value || '';
      const optType = optTypeRef.current?.value || 'call';
      const position = positionRef.current?.value || 'long';

      if (!window.eel) {
        setState(s => ({ ...s, loading: false, error: 'Eel non disponible (mode développement)' }));
        return;
      }

      const res = await window.eel.calculate_bsm(
        ticker, S, K, matStr, market.r, market.q, optType, position
      )();

      if (res.error) {
        setState(s => ({ ...s, loading: false, error: res.error }));
        return;
      }

      // Met à jour la courbe du grec actif
      const greekKey = state.activeGreek;
      const greekData = (res as any)[`${greekKey}_data`] as GreekPoint[] || [];

      setState(s => ({
        ...s, loading: false, error: null,
        price: res.price,
        greeks: res.greeks,
        payoff_data: res.payoff_data,
        activeGreekData: greekData,
        breakeven: res.breakeven,
        S: res.S,
        K: res.K,
        sigma: res.sigma,
        sigma_source: res.sigma_source,
      }));
    } catch (e: any) {
      setState(s => ({ ...s, loading: false, error: String(e) }));
    }
  };

  // ── Changement de grec affiché ───────────────────────────────────────

  const switchGreek = async (key: GreekKey) => {
    setState(s => ({ ...s, activeGreek: key }));
    if (!window.eel || !state.S || !state.K) return;
    try {
      const ticker = tickerRef.current?.value.trim().toUpperCase() || market.ticker;
      const matStr = maturityRef.current?.value || '';
      const optType = optTypeRef.current?.value || 'call';
      const position = positionRef.current?.value || 'long';
      const res = await window.eel.calculate_bsm(ticker, state.S, state.K, matStr, market.r, market.q, optType, position)();
      if (!res.error) {
        setState(s => ({ ...s, activeGreek: key, activeGreekData: (res as any)[`${key}_data`] || [] }));
      }
    } catch { /* silencieux */ }
  };

  const formatGreek = (val: number | undefined, digits = 4) =>
    val !== undefined ? val.toFixed(digits) : 'N/C';

  const greekColor = (val: number | undefined) => {
    if (val === undefined) return 'text-[#888888]';
    return val >= 0 ? 'text-[#00FF00]' : 'text-[#FF3333]';
  };

  // Valeur par défaut du strike (ATM arrondi)
  const defaultStrike = market.S ? Math.round(market.S) : 290;
  const defaultSigma = market.histVol ? (market.histVol * 100).toFixed(2) : '20.00';
  const defaultMaturity = getDefaultMaturity();

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">

      {/* Erreur globale */}
      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] rounded">
          Warning {state.error}
        </div>
      )}

      {/* Top row: Params + Données */}
      <div className="flex gap-1">

        {/* PARAMÈTRES BSM */}
        <div className="border border-[#222222] flex-shrink-0 w-[420px]">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white"> PARAMÈTRES BSM</span>
          </div>
          <div className="bg-[#000000] p-1.5 space-y-1.5">
            <FormRow label="Ticker Symbole">
              <input ref={tickerRef} defaultValue={market.ticker}
                className="w-[120px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none"
                placeholder="AAPL" />
            </FormRow>
            <FormRow label="Type d'option">
              <select ref={optTypeRef} defaultValue="call"
                className="w-[120px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none appearance-none cursor-pointer">
                <option value="call">call</option>
                <option value="put">put</option>
              </select>
            </FormRow>
            <FormRow label="Prix d'exercice (K)">
              <input ref={strikeRef} key={defaultStrike} defaultValue={defaultStrike}
                type="number" step="0.5"
                className="w-[120px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none" />
            </FormRow>
            <FormRow label="Date d'échéance">
              <input ref={maturityRef} defaultValue={defaultMaturity} type="date"
                className="w-[120px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none" />
            </FormRow>
            <FormRow label="Position">
              <select ref={positionRef} defaultValue="long"
                className="w-[120px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none appearance-none cursor-pointer">
                <option value="long">long</option>
                <option value="short">short</option>
              </select>
            </FormRow>

            <div className="pt-2 flex gap-1">
              <button id="bsm-fetch-btn" onClick={handleFetchData}
                disabled={state.loading}
                className="flex-1 bg-[#2A2A2A] border border-[#444444] text-white py-1 hover:bg-[#3A3A3A] transition-colors text-[10px] rounded-sm disabled:opacity-50">
                {state.loading ? 'Chargement...' : 'Récupérer Données'}
              </button>
              <button id="bsm-calc-btn" onClick={handleCalculate}
                disabled={state.loading}
                className="flex-1 bg-[#4A90E2] text-white py-1 hover:bg-[#357ABD] transition-colors text-[10px] font-bold rounded-sm disabled:opacity-50">
                {state.loading ? 'Calcul...' : 'Calculer Prix'}
              </button>
            </div>
          </div>
        </div>

        {/* DONNÉES MARCHÉ + Grecs */}
        <div className="flex-1 flex flex-col gap-1">

          {/* DONNÉES MARCHÉ */}
          <div className="border border-[#222222]">
            <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
              <span className="font-bold text-white"> DONNÉES MARCHÉ</span>
            </div>
            <div className="bg-[#000000] p-1.5 grid grid-cols-2 gap-x-4 gap-y-1.5">
              <DataRow label="Prix Actuel (S)" value={market.S ? `${market.S.toFixed(2)} $` : 'N/C'} />
              <DataRow label="Taux SOFR (r)"   value={`${(market.r * 100).toFixed(2)}%`} />
              <DataRow label="Dividende (q)"   value={`${(market.q * 100).toFixed(2)}%`} />
              <DataRow label={`Volatilité (σ) [${state.sigma_source || 'N/A'}]`} value={state.sigma ? `${(state.sigma * 100).toFixed(2)}%` : 'N/A'} />
              <DataRow label="Prix de l'option"
                value={state.price !== null ? `${state.price.toFixed(4)} $` : 'N/C'}
                highlight={state.price !== null} />
              {state.breakeven !== null && (
                <DataRow label="Point Mort" value={`${state.breakeven.toFixed(2)} $`} />
              )}
            </div>
          </div>

          {/* Grecs (BSM) */}
          <div className="border border-[#222222]">
            <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
              <span className="font-bold text-white"> GRECS (BSM)</span>
              <span className="ml-2 text-[#888888] text-[9px]">Cliquer pour afficher la courbe ↓</span>
            </div>
            <div className="bg-[#000000] overflow-auto">
              <table className="w-full text-right border-collapse text-[11px]">
                <thead>
                  <tr className="bg-[#111111] text-[9px] uppercase text-[#888888] divide-x divide-[#222222] border-b border-[#222222]">
                    {(['delta', 'gamma', 'theta', 'vega', 'rho'] as GreekKey[]).map(g => (
                      <th key={g}
                        id={`bsm-greek-btn-${g}`}
                        onClick={() => switchGreek(g)}
                        className={`py-1 px-2 font-normal cursor-pointer transition-colors select-none
                          ${state.activeGreek === g ? 'bg-[#4A90E2] text-white' : 'hover:bg-[#222222]'}`}>
                        {g.charAt(0).toUpperCase() + g.slice(1)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr className="divide-x divide-[#222222]">
                    <td className={`py-1.5 px-2 ${greekColor(state.greeks?.delta)}`}>
                      {formatGreek(state.greeks?.delta)}
                    </td>
                    <td className="py-1.5 px-2 text-[#D4D4D4]">
                      {formatGreek(state.greeks?.gamma)}
                    </td>
                    <td className={`py-1.5 px-2 ${greekColor(state.greeks?.theta)}`}>
                      {formatGreek(state.greeks?.theta)}
                    </td>
                    <td className={`py-1.5 px-2 ${greekColor(state.greeks?.vega)}`}>
                      {formatGreek(state.greeks?.vega)}
                    </td>
                    <td className="py-1.5 px-2 text-[#D4D4D4]">
                      {formatGreek(state.greeks?.rho)}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom row: Courbe grec + Payoff */}
      <div className="flex gap-1 flex-1 min-h-[250px]">

        {/* Courbe du grec sélectionné */}
        <div className="flex-1 border border-[#222222] flex flex-col">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white">
               ÉVOLUTION DU {state.activeGreek.toUpperCase()}
            </span>
          </div>
          <div className="bg-[#0A0A0A] flex-1 p-2 relative">
            {state.activeGreekData.length === 0 ? (
              <div className="flex items-center justify-center h-full text-[#888888] text-[11px]">
                Calculer le prix pour afficher la courbe
              </div>
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

        {/* Payoff de l'option */}
        <div className="flex-1 border border-[#222222] flex flex-col">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white"> PAYOFF DE L'OPTION</span>
          </div>
          <div className="bg-[#0A0A0A] flex-1 p-2 relative">
            {state.payoff_data.length === 0 ? (
              <div className="flex items-center justify-center h-full text-[#888888] text-[11px]">
                Calculer le prix pour afficher le payoff
              </div>
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

// ── Helpers ────────────────────────────────────────────────────────────────

function computeDaysFromDate(dateStr: string): number {
  const today = new Date();
  const target = new Date(dateStr);
  const diff = Math.round((target.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
  return Math.max(diff, 1);
}

function getDefaultMaturity(): string {
  const d = new Date();
  d.setDate(d.getDate() + 90);
  return d.toISOString().split('T')[0];
}

function FormRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-[#222222] pb-1">
      <span className="text-[10px] text-[#888888]">{label}</span>
      {children}
    </div>
  );
}

function DataRow({ label, value, highlight = false }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-[#222222] pb-1">
      <span className="text-[10px] text-[#888888]">{label}</span>
      <span className={`text-[11px] font-bold ${highlight ? 'text-[#4A90E2]' : 'text-[#FFFFFF]'}`}>{value}</span>
    </div>
  );
}

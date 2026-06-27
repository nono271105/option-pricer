import React, { useState, useRef, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { useMarket } from '../App';

// ── Types ────────────────────────────────────────────────────────────────

interface Leg {
  option_type: string;
  position: string;
  strike: number;
  premium: number;
}

interface PayoffPoint { spot: number; payoff: number; }
interface ValuePoint  { spot: number; value: number; }

interface StrategyState {
  strategy_name: string | null;
  legs: Leg[];
  payoff_data: PayoffPoint[];
  value_today_data: ValuePoint[];
  metrics: {
    cost: number | null;
    breakevens: number[];
    max_gain: number | null;
    max_loss: number | null;
  } | null;
  greeks: { delta: number; gamma: number; theta: number; vega: number; rho: number; } | null;
  sigma?: number;
  sigma_source?: string;
  loading: boolean;
  error: string | null;
}

// Familles de stratégies
const STRATEGY_FAMILIES: Record<string, string[]> = {
  'Positions de base': [
    'Long Call', 'Short Call', 'Long Put', 'Short Put',
  ],
  'Spreads directionnels': [
    'Bull Call Spread', 'Bear Call Spread', 'Bull Put Spread', 'Bear Put Spread',
  ],
  'Volatilité': [
    'Long Straddle', 'Short Straddle', 'Long Strangle', 'Short Strangle',
  ],
  'Butterflies': [
    'Long Call Butterfly', 'Short Call Butterfly',
    'Long Put Butterfly', 'Short Put Butterfly',
    'Long Iron Butterfly', 'Short Iron Butterfly',
  ],
  'Condors': [
    'Long Call Condor', 'Short Call Condor',
    'Long Put Condor', 'Short Put Condor',
    'Long Iron Condor', 'Short Iron Condor',
  ],
};

function fmtNum(v: number | null, decimals = 4): string {
  if (v === null || v === undefined) return 'N/C';
  return v.toFixed(decimals);
}

function fmtVega(v?: number): string {
  if (v === undefined || v === null) return 'N/C';
  return (v / 100).toFixed(4);
}

// ── Composant principal ───────────────────────────────────────────────────

export function StrategiesTab() {
  const market = useMarket();
  const tickerRef   = useRef<HTMLInputElement>(null);
  const maturityRef = useRef<HTMLInputElement>(null);

  const [family, setFamily]     = useState('Spreads directionnels');
  const [strategy, setStrategy] = useState('Bull Call Spread');

  const [state, setState] = useState<StrategyState>({
    strategy_name: null, legs: [],
    payoff_data: [], value_today_data: [],
    metrics: null, greeks: null,
    loading: false, error: null,
  });

  useEffect(() => {
    if (window.eel) {
      window.eel.get_strategy_names()().then((names: string[]) => {
        if (names.length > 0 && !names.includes(strategy)) {
          setStrategy(names[0]);
        }
      }).catch(() => { /* silencieux */ });
    }
  }, []);

  const handleAnalyze = async () => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const ticker = tickerRef.current?.value.trim().toUpperCase() || market.ticker;
      const S = market.S ?? 100;
      const matStr = maturityRef.current?.value || '';
      const T_days = matStr ? computeDaysFromDate(matStr) : 90;
      const expiry = matStr || getDefaultMaturity();

      if (!window.eel) {
        setState(s => ({ ...s, loading: false, error: 'Eel non disponible' }));
        return;
      }

      const res = await window.eel.calculate_strategy(
        strategy, ticker, S, T_days, market.r, market.q, expiry
      )();

      if (res.error) {
        setState(s => ({ ...s, loading: false, error: res.error }));
        return;
      }

      setState(s => ({
        ...s, loading: false, error: null,
        strategy_name: res.strategy_name,
        legs: res.legs,
        payoff_data: res.payoff_data,
        value_today_data: res.value_today_data,
        metrics: res.metrics,
        greeks: res.greeks,
        sigma: res.sigma,
        sigma_source: res.sigma_source,
      }));
    } catch (e: any) {
      setState(s => ({ ...s, loading: false, error: String(e) }));
    }
  };

  const costColor = (cost: number | null) => {
    if (cost === null) return 'text-[#D4D4D4]';
    return cost < 0 ? 'text-[#FF6B6B]' : 'text-[#6BCB77]';
  };

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">

      {/* Erreur */}
      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] rounded shrink-0">
          {state.error}
        </div>
      )}

      {/* ── Barre de sélection ── */}
      <div className="border border-[#222222] shrink-0">
        <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
          <span className="font-bold text-white"> STRATÉGIES : Sélection &amp; Analyse</span>
        </div>
        <div className="flex items-center gap-4 px-3 py-2 flex-wrap">
          <label className="flex items-center gap-1.5 text-[10px] text-[#888888]">
            Ticker
            <input ref={tickerRef} defaultValue={market.ticker}
              className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] w-[80px] outline-none rounded-sm font-bold" />
          </label>
          <label className="flex items-center gap-1.5 text-[10px] text-[#888888]">
            Maturité
            <input ref={maturityRef} defaultValue={getDefaultMaturity()} type="date"
              className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] outline-none rounded-sm" />
          </label>
          <label className="flex items-center gap-1.5 text-[10px] text-[#888888]">
            Famille
            <select value={family}
              onChange={e => { setFamily(e.target.value); setStrategy(STRATEGY_FAMILIES[e.target.value][0]); }}
              className="bg-[#121212] border border-[#333333] text-white py-0.5 px-1.5 text-[11px] outline-none appearance-none rounded-sm">
              {Object.keys(STRATEGY_FAMILIES).map(f => <option key={f}>{f}</option>)}
            </select>
          </label>
          <label className="flex items-center gap-1.5 text-[10px] text-[#888888]">
            Stratégie
            <select value={strategy} onChange={e => setStrategy(e.target.value)}
              className="bg-[#121212] border border-[#333333] text-white py-0.5 px-1.5 text-[11px] outline-none appearance-none rounded-sm">
              {STRATEGY_FAMILIES[family].map(s => <option key={s}>{s}</option>)}
            </select>
          </label>
          <button id="strategy-analyze-btn" onClick={handleAnalyze} disabled={state.loading}
            className="bg-[#4A90E2] text-white px-4 py-1 hover:bg-[#357ABD] text-[10px] font-bold rounded-sm disabled:opacity-50 transition-colors ml-auto">
            {state.loading ? 'Analyse...' : 'Analyser'}
          </button>
        </div>
      </div>

      {/* ── Résultats ── */}
      {state.legs.length > 0 && state.metrics && (
        <div className="flex gap-1 shrink-0">

          {/* Tableau des legs */}
          <div className="flex-1 border border-[#222222]">
            <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
              <span className="font-bold text-white"> COMPOSITION : {state.strategy_name}</span>
            </div>
            <table className="w-full text-right border-collapse text-[11px]">
              <thead>
                <tr className="bg-[#0D0D0D] text-[9px] uppercase text-[#666666] divide-x divide-[#1E1E1E] border-b border-[#222222]">
                  <th className="py-1 px-2 font-normal text-left">Position</th>
                  <th className="py-1 px-2 font-normal">Instrument</th>
                  <th className="py-1 px-2 font-normal">Strike</th>
                  <th className="py-1 px-2 font-normal">Prime</th>
                  <th className="py-1 px-2 font-normal">Flux</th>
                </tr>
              </thead>
              <tbody>
                {state.legs.map((leg, i) => {
                  const isLong = leg.position === 'long';
                  const flux = isLong ? -leg.premium : +leg.premium;
                  return (
                    <tr key={i} className="divide-x divide-[#1A1A1A] border-b border-[#111111] hover:bg-[#0A0A0A]">
                      <td className="py-1 px-2 text-left">
                        <span className={`font-bold text-[10px] px-1.5 py-0.5 rounded-sm ${isLong ? 'bg-[#0D2214] text-[#6BCB77]' : 'bg-[#2A0D0D] text-[#FF6B6B]'}`}>
                          {leg.position.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-1 px-2 text-[#D4D4D4]">{leg.option_type.toUpperCase()}</td>
                      <td className="py-1 px-2 text-[#FFFFFF] font-bold">{leg.strike.toFixed(2)}</td>
                      <td className="py-1 px-2 text-[#888888]">{leg.premium.toFixed(4)} $</td>
                      <td className={`py-1 px-2 font-bold ${flux < 0 ? 'text-[#FF6B6B]' : 'text-[#6BCB77]'}`}>
                        {flux >= 0 ? '+' : ''}{flux.toFixed(4)} $
                      </td>
                    </tr>
                  );
                })}
                {/* Ligne coût net */}
                <tr className="bg-[#0D0D0D] divide-x divide-[#1E1E1E]">
                  <td colSpan={4} className="py-1 px-2 text-left text-[9px] text-[#666666] uppercase tracking-wider">Coût net total</td>
                  <td className={`py-1 px-2 font-bold text-[12px] ${costColor(state.metrics.cost)}`}>
                    {state.metrics.cost !== null
                      ? `${state.metrics.cost >= 0 ? '+' : ''}${state.metrics.cost.toFixed(4)} $`
                      : 'N/C'}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Métriques & Grecs */}
          <div className="w-[260px] border border-[#222222] flex flex-col">
            <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
              <span className="font-bold text-white"> MÉTRIQUES</span>
            </div>
            <div className="bg-[#000000] p-2 space-y-0.5 flex-1">

              {/* Volatilité */}
              {state.sigma !== undefined && (
                <MetricRow label={`Volatilité (σ) [${state.sigma_source || 'N/A'}]`}
                  value={`${(state.sigma * 100).toFixed(2)}%`} />
              )}

              <div className="h-[1px] bg-[#1E1E1E] my-1.5" />

              {/* Breakevens */}
              {state.metrics.breakevens.map((be, i) => (
                <MetricRow key={i}
                  label={`Breakeven${state.metrics!.breakevens.length > 1 ? ` ${i + 1}` : ''}`}
                  value={`${be.toFixed(2)} $`} />
              ))}

              {/* Gain / Perte max */}
              <MetricRow label="Gain maximum"
                value={state.metrics.max_gain !== null ? `+${fmtNum(state.metrics.max_gain)} $` : 'Illimité'}
                valueClass="text-[#6BCB77]" />
              <MetricRow label="Perte maximum"
                value={state.metrics.max_loss !== null ? `${fmtNum(state.metrics.max_loss)} $` : 'Illimitée'}
                valueClass="text-[#FF6B6B]" />

              <div className="h-[1px] bg-[#1E1E1E] my-1.5" />

              {/* Grecs agrégés */}
              <p className="text-[9px] text-[#666666] uppercase tracking-widest mb-1">Grecs agrégés</p>
              {state.greeks && (
                <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                  <MetricRow label="Delta"  value={fmtNum(state.greeks.delta)} />
                  <MetricRow label="Gamma"  value={fmtNum(state.greeks.gamma)} />
                  <MetricRow label="Theta"  value={fmtNum(state.greeks.theta)} />
                  <MetricRow label="Vega"   value={fmtVega(state.greeks.vega)} />
                  <MetricRow label="Rho"    value={fmtNum(state.greeks.rho)} />
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Graphique P&L ── */}
      {state.payoff_data.length > 0 && (
        <div className="flex-1 border border-[#222222] flex flex-col min-h-[250px]">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white"> PROFIL P&amp;L : {state.strategy_name}</span>
            <div className="ml-4 flex gap-4 text-[9px] text-[#666666]">
              <span className="flex items-center gap-1">
                <span style={{ display: 'inline-block', width: 14, height: 2, background: '#4A90E2', borderRadius: 1 }} />
                Payoff à maturité
              </span>
              <span className="flex items-center gap-1">
                <span style={{ display: 'inline-block', width: 14, height: 2, background: '#FFCC00', borderRadius: 1 }} />
                Valeur aujourd'hui
              </span>
              {market.S && (
                <span className="flex items-center gap-1">
                  <span style={{ display: 'inline-block', width: 14, height: 2, background: '#888888', borderRadius: 1, borderTop: '1px dashed #888' }} />
                  Spot actuel
                </span>
              )}
            </div>
          </div>
          <div className="flex-1 bg-[#050505] p-2">
            <Plot
              data={[
                {
                  x: state.payoff_data.map(p => p.spot),
                  y: state.payoff_data.map(p => p.payoff),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  line: { color: '#4A90E2', width: 2 },
                  fill: 'tozeroy',
                  fillcolor: 'rgba(74, 144, 226, 0.07)',
                  name: 'Payoff maturité',
                  hovertemplate: 'S: %{x:.2f} $<br>P&L: <b>%{y:.4f} $</b><extra></extra>',
                },
                {
                  x: state.value_today_data.map(p => p.spot),
                  y: state.value_today_data.map(p => p.value),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  line: { color: '#FFCC00', width: 1.5, dash: 'dot' as const },
                  name: "Valeur aujourd'hui",
                  hovertemplate: 'S: %{x:.2f} $<br>Valeur: <b>%{y:.4f} $</b><extra></extra>',
                }
              ]}
              layout={{
                autosize: true,
                margin: { l: 50, r: 20, t: 10, b: 40 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                xaxis: {
                  title: { text: 'Prix sous-jacent (S)', font: { size: 9, color: '#666' } },
                  gridcolor: '#111111',
                  zerolinecolor: '#333333',
                  tickfont: { color: '#666', size: 9 },
                },
                yaxis: {
                  title: { text: 'P&L ($)', font: { size: 9, color: '#666' } },
                  gridcolor: '#111111',
                  zerolinecolor: '#444444',
                  zeroline: true,
                  tickfont: { color: '#666', size: 9 },
                },
                hovermode: 'x unified',
                hoverlabel: {
                  bgcolor: '#1A1A1A',
                  bordercolor: '#333333',
                  font: { color: '#D4D4D4', size: 10 },
                },
                legend: {
                  orientation: 'h', y: -0.15,
                  font: { color: '#888', size: 9 },
                  bgcolor: 'transparent',
                },
                shapes: [
                  // Ligne zéro
                  {
                    type: 'line', xref: 'paper', x0: 0, x1: 1,
                    y0: 0, y1: 0,
                    line: { color: '#333333', width: 1 }
                  },
                  // Breakevens
                  ...(state.metrics?.breakevens || []).map(be => ({
                    type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: be, x1: be,
                    y0: 0, y1: 1,
                    line: { color: '#555555', dash: 'dash' as const, width: 1 }
                  })),
                  // Spot actuel
                  ...(market.S ? [{
                    type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: market.S, x1: market.S,
                    y0: 0, y1: 1,
                    line: { color: '#888888', dash: 'dot' as const, width: 1 }
                  }] : [])
                ],
                annotations: [
                  ...(state.metrics?.breakevens || []).map(be => ({
                    x: be, y: 0.98, xref: 'x' as const, yref: 'paper' as const,
                    text: `BE ${be.toFixed(0)}`, showarrow: false, yanchor: 'top',
                    font: { color: '#666666', size: 9 },
                    bgcolor: 'transparent',
                  })),
                  ...(market.S ? [{
                    x: market.S, y: 0.98, xref: 'x' as const, yref: 'paper' as const,
                    text: `S ${market.S.toFixed(0)}`, showarrow: false, yanchor: 'top',
                    font: { color: '#888888', size: 9 },
                    bgcolor: 'transparent',
                  }] : [])
                ],
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler={true}
            />
          </div>
        </div>
      )}

      {/* État vide */}
      {state.payoff_data.length === 0 && !state.loading && !state.error && (
        <div className="flex-1 flex items-center justify-center text-[#444444] text-[12px]">
          Sélectionnez une stratégie et cliquez sur « Analyser »
        </div>
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────

function MetricRow({ label, value, valueClass }: { label: string; value: string; valueClass?: string }) {
  return (
    <div className="flex items-center justify-between gap-2 py-0.5 border-b border-[#111111]">
      <span className="text-[9px] text-[#666666]">{label}</span>
      <span className={`text-[11px] font-bold ${valueClass || 'text-[#D4D4D4]'}`}>{value}</span>
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
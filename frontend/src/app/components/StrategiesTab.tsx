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

// Familles de stratégies (correspondant aux noms Python)
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

  // Charge les noms de stratégies depuis Python au montage
  useEffect(() => {
    if (window.eel) {
      window.eel.get_strategy_names()().then((names: string[]) => {
        // Si la stratégie par défaut n'est pas dans la liste, on prend la première
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

  const currentSigma = state.sigma ? (state.sigma * 100).toFixed(2) : 'N/A';

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">

      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] rounded shrink-0">
          Warning {state.error}
        </div>
      )}

      {/* Sélecteur de stratégie */}
      <div className="border border-[#222222] shrink-0">
        <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] justify-between flex-wrap gap-2">
          <span className="font-bold text-white"> STRATÉGIES  Construction et Analyse</span>
          <div className="flex items-center gap-2 flex-wrap">
            <label className="text-[#888888] flex items-center gap-1">
              Ticker :
              <input ref={tickerRef} defaultValue={market.ticker}
                className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] w-[80px] outline-none ml-1" />
            </label>

            <label className="text-[#888888] flex items-center gap-1">
              Maturité :
              <input ref={maturityRef} defaultValue={getDefaultMaturity()} type="date"
                className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] outline-none ml-1" />
            </label>
            <label className="text-[#888888] flex items-center gap-1">
              Famille :
              <select value={family}
                onChange={e => { setFamily(e.target.value); setStrategy(STRATEGY_FAMILIES[e.target.value][0]); }}
                className="bg-[#121212] border border-[#333333] text-white py-0.5 px-1.5 text-[11px] outline-none appearance-none ml-1">
                {Object.keys(STRATEGY_FAMILIES).map(f => <option key={f}>{f}</option>)}
              </select>
            </label>
            <label className="text-[#888888] flex items-center gap-1">
              Stratégie :
              <select value={strategy} onChange={e => setStrategy(e.target.value)}
                className="bg-[#121212] border border-[#333333] text-white py-0.5 px-1.5 text-[11px] outline-none appearance-none ml-1">
                {STRATEGY_FAMILIES[family].map(s => <option key={s}>{s}</option>)}
              </select>
            </label>
            <button id="strategy-analyze-btn" onClick={handleAnalyze} disabled={state.loading}
              className="bg-[#4A90E2] text-white px-3 py-0.5 hover:bg-[#357ABD] text-[10px] font-bold rounded-sm disabled:opacity-50">
              {state.loading ? 'Analyse...' : 'Analyser'}
            </button>
          </div>
        </div>
      </div>

      {/* Legs + Métriques */}
      {(state.legs.length > 0 || state.metrics !== null) && (
        <div className="flex gap-1 shrink-0">
          {/* Legs table */}
          <div className="flex-1 border border-[#222222]">
            <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
              <span className="font-bold text-white"> LEGS  {state.strategy_name}</span>
            </div>
            <div className="overflow-auto">
              <table className="w-full text-right border-collapse text-[11px]">
                <thead>
                  <tr className="bg-[#111111] text-[9px] uppercase text-[#888888] divide-x divide-[#222222] border-b border-[#222222]">
                    <th className="py-1 px-2 font-normal text-left">Position</th>
                    <th className="py-1 px-2 font-normal">Type</th>
                    <th className="py-1 px-2 font-normal">Strike</th>
                    <th className="py-1 px-2 font-normal">Prime</th>
                  </tr>
                </thead>
                <tbody>
                  {state.legs.map((leg, i) => (
                    <tr key={i} className={`divide-x divide-[#222222] border-b border-[#1A1A1A] ${leg.position === 'long' ? 'text-[#00FF00]' : 'text-[#FF4444]'}`}>
                      <td className="py-1 px-2 text-left font-bold">{leg.position.toUpperCase()}</td>
                      <td className="py-1 px-2 text-[#D4D4D4]">{leg.option_type.toUpperCase()}</td>
                      <td className="py-1 px-2 text-[#D4D4D4] font-bold">{leg.strike.toFixed(2)}</td>
                      <td className={`py-1 px-2 ${leg.position === 'long' ? 'text-[#FF4444]' : 'text-[#00FF00]'}`}>
                        {leg.position === 'long' ? '-' : '+'}{leg.premium.toFixed(4)} $
                      </td>
                    </tr>
                  ))}
                  {state.metrics !== null && (
                    <tr className="bg-[#111111] divide-x divide-[#222222]">
                      <td colSpan={3} className="py-1 px-2 text-left text-[#888888]">Coût net</td>
                      <td className={`py-1 px-2 font-bold ${(state.metrics.cost ?? 0) < 0 ? 'text-[#FF4444]' : 'text-[#00FF00]'}`}>
                        {state.metrics.cost !== null ? `${state.metrics.cost.toFixed(4)} $` : 'N/C'}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Métriques */}
          {state.metrics && state.greeks && (
            <div className="w-[260px] border border-[#222222] flex flex-col">
              <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
                <span className="font-bold text-white"> MÉTRIQUES</span>
              </div>
              <div className="bg-[#000000] p-1.5 space-y-1">
                <MR label="Coût total" value={state.metrics.cost !== null ? `${state.metrics.cost.toFixed(4)} $` : 'N/C'}
                  color={(state.metrics.cost ?? 0) < 0 ? 'text-[#FF4444]' : 'text-[#00FF00]'} />
                {state.metrics.breakevens.map((be, i) => (
                  <MR key={i} label={`Breakeven${state.metrics!.breakevens.length > 1 ? ` ${i + 1}` : ''}`} value={`${be.toFixed(2)} $`} />
                ))}
                <MR label="Gain maximum"
                  value={state.metrics.max_gain !== null ? `+${state.metrics.max_gain.toFixed(4)} $` : 'Illimité'}
                  color="text-[#00FF00]" />
                <MR label="Perte maximum"
                  value={state.metrics.max_loss !== null ? `${state.metrics.max_loss.toFixed(4)} $` : 'Illimitée'}
                  color="text-[#FF4444]" />
                <div className="flex justify-between py-0.5 border-b border-[#222]">
                  <span className="text-[#888] text-[10px]">Volatilité (σ) [{state.sigma_source || 'N/A'}] :</span>
                  <span className="text-[#0F0] font-bold text-[11px]">{currentSigma !== 'N/A' ? `${currentSigma}%` : 'N/A'}</span>
                </div>

                <div className="border-t border-[#222222] pt-1 mt-1">
                  <p className="text-[9px] text-[#888888] uppercase tracking-widest mb-1">Grecs Agrégés</p>
                  <div className="grid grid-cols-2 gap-y-1 gap-x-2 text-[10px]">
                    {Object.entries(state.greeks).map(([k, v]) => (
                      <div key={k} className="flex justify-between">
                        <span className="text-[#888888]">{k.charAt(0).toUpperCase() + k.slice(1)}</span>
                        <span className={typeof v === 'number' && v >= 0 ? 'text-[#00FF00]' : 'text-[#FF4444]'}>
                          {typeof v === 'number' ? v.toFixed(4) : v}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Graphique P&L */}
      {state.payoff_data.length > 0 && (
        <div className="flex-1 border border-[#222222] flex flex-col min-h-[250px]">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white"> PROFIL P&L  {state.strategy_name}</span>
            <div className="ml-4 flex gap-4 text-[9px] text-[#888888]">
              <span className="flex items-center gap-1"><span className="text-[#00FF00]"></span> Payoff à maturité</span>
              <span className="flex items-center gap-1"><span className="text-[#FFCC00]"></span> Valeur aujourd'hui</span>
            </div>
          </div>
          <div className="flex-1 bg-[#0A0A0A] p-2">
            <Plot
              data={[
                {
                  x: state.payoff_data.map(p => p.spot),
                  y: state.payoff_data.map(p => p.payoff),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  line: { color: '#00FF00', width: 1.5 },
                  fill: 'tozeroy',
                  fillcolor: 'rgba(0, 255, 0, 0.1)',
                  name: 'Payoff maturité'
                },
                {
                  x: state.value_today_data.map(p => p.spot),
                  y: state.value_today_data.map(p => p.value),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  line: { color: '#FFCC00', width: 1.5 },
                  name: "Valeur aujourd'hui"
                }
              ]}
              layout={{
                autosize: true,
                margin: { l: 40, r: 20, t: 10, b: 30 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                xaxis: { title: { text: 'Prix sous-jacent (S)', font: { size: 9, color: '#888' } }, gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 } },
                yaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 } },
                hovermode: 'x unified',
                shapes: [
                  {
                    type: 'line', xref: 'paper', x0: 0, x1: 1,
                    y0: 0, y1: 0,
                    line: { color: '#444444', width: 1 }
                  },
                  ...(state.metrics?.breakevens || []).map(be => ({
                    type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: be, x1: be,
                    y0: 0, y1: 1,
                    line: { color: '#D0D0D0', dash: 'dash' as const, width: 1 }
                  })),
                  ...(market.S ? [{
                    type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: market.S, x1: market.S,
                    y0: 0, y1: 1,
                    line: { color: '#FF4444', dash: 'dashdot' as const, width: 1 }
                  }] : [])
                ],
                annotations: [
                  ...(state.metrics?.breakevens || []).map(be => ({
                    x: be, y: 1, xref: 'x' as const, yref: 'paper' as const,
                    text: `BE=${be.toFixed(0)}`, showarrow: false, yanchor: 'bottom', font: { color: '#D0D0D0', size: 9 }
                  })),
                  ...(market.S ? [{
                    x: market.S, y: 1, xref: 'x' as const, yref: 'paper' as const,
                    text: `S=${market.S.toFixed(0)}`, showarrow: false, yanchor: 'bottom', font: { color: '#FF4444', size: 9 }
                  }] : [])
                ],
                legend: { orientation: 'h', y: -0.2 }
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler={true}
            />
          </div>
        </div>
      )}

      {/* État vide */}
      {state.payoff_data.length === 0 && !state.loading && !state.error && (
        <div className="flex-1 flex items-center justify-center text-[#888888] text-[12px]">
          Sélectionnez une stratégie et cliquez sur "Analyser"
        </div>
      )}
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────

function MR({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-[#1A1A1A] pb-0.5">
      <span className="text-[10px] text-[#888888]">{label}</span>
      <span className={`text-[11px] font-bold ${color || 'text-[#FFFFFF]'}`}>{value}</span>
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
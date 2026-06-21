import React, { useState, useRef } from 'react';
import Plot from 'react-plotly.js';
import { useMarket } from '../App';

// ── Types ─────────────────────────────────────────────────────────────────

interface ForecastPoint {
  day: number;
  iv_hist: number | null;
  iv_forecast: number | null;
  price_hist: number | null;
  price_forecast: number | null;
  delta_hist: number | null;
  delta_forecast: number | null;
}

interface ForecastState {
  data: ForecastPoint[];
  occ_symbol: string | null;
  loading: boolean;
  error: string | null;
}

// ── Composant principal ────────────────────────────────────────────────────

export function ForecastTab() {
  const market = useMarket();
  const tickerRef    = useRef<HTMLInputElement>(null);
  const strikeRef    = useRef<HTMLInputElement>(null);
  const optTypeRef   = useRef<HTMLSelectElement>(null);
  const maturityRef  = useRef<HTMLInputElement>(null);
  const histDaysRef  = useRef<HTMLInputElement>(null);
  const forecastDaysRef = useRef<HTMLInputElement>(null);

  const [state, setState] = useState<ForecastState>({
    data: [], occ_symbol: null, loading: false, error: null,
  });

  const handleForecast = async () => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const ticker   = tickerRef.current?.value.trim().toUpperCase() || market.ticker;
      const strike   = parseFloat(strikeRef.current?.value || String(Math.round(market.S ?? 100)));
      const optType  = optTypeRef.current?.value || 'call';
      const matStr   = maturityRef.current?.value || getDefaultMaturity();
      const T_days   = computeDaysFromDate(matStr);
      const histDays = parseInt(histDaysRef.current?.value || '60');
      const forecastDays = parseInt(forecastDaysRef.current?.value || '10');

      if (!(window as any).eel) {
        setState(s => ({ ...s, loading: false, error: 'Eel non disponible' }));
        return;
      }

      const res = await (window as any).eel.run_forecast(ticker, strike, T_days, optType, matStr, histDays, forecastDays)();

      if (res.error) {
        setState(s => ({ ...s, loading: false, error: res.error }));
        return;
      }

      // Construire le tableau combiné historique + forecast
      const N = res.x_history.length;
      const horizon = res.iv_forecast.length;
      const combined: ForecastPoint[] = [];

      // Historique (jours négatifs)
      for (let i = 0; i < N; i++) {
        combined.push({
          day: res.x_history[i],
          iv_hist:    (res.iv_history[i] ?? null) !== null ? (res.iv_history[i]! * 100) : null,
          iv_forecast: null,
          price_hist:   res.option_prices_history[i] ?? null,
          price_forecast: null,
          delta_hist:   res.deltas_history[i] ?? null,
          delta_forecast: null,
        });
      }

      // Jonction : dernier point historique = premier point forecast
      if (N > 0 && horizon > 0) {
        const lastDay = res.x_history[N - 1];
        for (let i = 0; i < horizon; i++) {
          const day = lastDay + i + 1;
          combined.push({
            day,
            iv_hist: i === 0 ? (res.iv_history[N - 1] !== null ? (res.iv_history[N - 1]! * 100) : null) : null,
            iv_forecast: (res.iv_forecast[i] ?? null) !== null ? (res.iv_forecast[i]! * 100) : null,
            price_hist: i === 0 ? (res.option_prices_history[N - 1] ?? null) : null,
            price_forecast: res.option_prices_forecast[i] ?? null,
            delta_hist: i === 0 ? (res.deltas_history[N - 1] ?? null) : null,
            delta_forecast: res.deltas_forecast[i] ?? null,
          });
        }
      }

      setState(s => ({
        ...s, loading: false, error: null,
        data: combined,
        occ_symbol: res.occ_symbol,
      }));
    } catch (e: any) {
      setState(s => ({ ...s, loading: false, error: String(e) }));
    }
  };

  const defaultStrike = market.S ? Math.round(market.S) : 100;

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">

      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] rounded shrink-0">
          Warning {state.error}
        </div>
      )}

      {/* Toolbar */}
      <div className="border border-[#222222] shrink-0">
        <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] justify-between flex-wrap gap-2">
          <span className="font-bold text-white"> FORECAST IV</span>
          <div className="flex items-center gap-2 flex-wrap">
            <label className="text-[#888888] flex items-center gap-1">
              Ticker :
              <input ref={tickerRef} defaultValue={market.ticker}
                className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] w-[80px] outline-none ml-1" />
            </label>
            <label className="text-[#888888] flex items-center gap-1">
              Strike :
              <input ref={strikeRef} key={defaultStrike} defaultValue={defaultStrike}
                type="number" step="0.5"
                className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] w-[80px] outline-none ml-1" />
            </label>
            <label className="text-[#888888] flex items-center gap-1">
              Type :
              <select ref={optTypeRef} defaultValue="call"
                className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] outline-none appearance-none ml-1">
                <option value="call">call</option>
                <option value="put">put</option>
              </select>
            </label>
            <label className="text-[#888888] flex items-center gap-1">
              Maturité :
              <input ref={maturityRef} defaultValue={getDefaultMaturity()} type="date"
                className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] outline-none ml-1" />
            </label>
            <label className="text-[#888888] flex items-center gap-1">
              Historique (j) :
              <input ref={histDaysRef} defaultValue="60" type="number" step="10" min="10" max="180"
                className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] w-[50px] outline-none ml-1" />
            </label>
            <label className="text-[#888888] flex items-center gap-1">
              Horizon (j) :
              <input ref={forecastDaysRef} defaultValue="10" type="number" step="1" min="1" max="90"
                className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] w-[50px] outline-none ml-1" />
            </label>
            <button id="forecast-run-btn" onClick={handleForecast} disabled={state.loading}
              className="bg-[#4A90E2] text-white px-3 py-0.5 hover:bg-[#357ABD] text-[10px] font-bold rounded-sm disabled:opacity-50">
              {state.loading ? 'Calcul...' : 'Lancer le Forecast'}
            </button>
          </div>
        </div>

        {/* Model info bar */}
        <div className="flex gap-6 px-2 py-0.5 bg-[#0A0A0A] text-[9px] text-[#888888]">
          {state.occ_symbol && <span>Contrat OCC : <span className="text-[#FFCC00]">{state.occ_symbol}</span></span>}
        </div>
      </div>

      <div className="flex gap-1 flex-1 min-h-[220px]">
        {/* Chart 1: IV */}
        <div className="flex-1 border border-[#222222] flex flex-col">
        <div className="flex items-center justify-between px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
          <span className="font-bold text-white"> VOLATILITÉ IMPLICITE  Historique + Forecast</span>
          <div className="flex gap-3 text-[9px] text-[#888888]">
            <span className="flex items-center gap-1"><span className="text-[#D0D0D0]"></span> IV historique</span>
            <span className="flex items-center gap-1"><span className="text-[#4A90E2]"></span> IV forecast</span>
          </div>
        </div>
        <div className="flex-1 bg-[#0A0A0A] p-2">
          {state.data.length === 0 ? (
            <div className="flex items-center justify-center h-full text-[#888888] text-[12px]">
              Configurez les paramètres et lancez le forecast
            </div>
          ) : (
            <Plot
              data={[
                {
                  x: state.data.map(d => d.day),
                  y: state.data.map(d => d.iv_hist),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  line: { color: '#D0D0D0', width: 1.5 },
                  name: 'IV historique',
                  connectgaps: false
                },
                {
                  x: state.data.map(d => d.day),
                  y: state.data.map(d => d.iv_forecast),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  line: { color: '#4A90E2', width: 2 },
                  name: 'IV forecast',
                  connectgaps: false
                }
              ]}
              layout={{
                autosize: true,
                margin: { l: 40, r: 20, t: 10, b: 30 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                xaxis: { title: { text: 'Jours (0 = début forecast)', font: { size: 9, color: '#888' } }, gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 } },
                yaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 }, ticksuffix: '%' },
                hovermode: 'x unified',
                shapes: [
                  {
                    type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: 0, x1: 0,
                    y0: 0, y1: 1,
                    line: { color: '#FFCC00', dash: 'dash' as const, width: 1 }
                  }
                ],
                annotations: [
                  {
                    x: 0, y: 1, xref: 'x' as const, yref: 'paper' as const,
                    text: 'Forecast →', showarrow: false, yanchor: 'bottom', font: { color: '#FFCC00', size: 9 }
                  }
                ],
                showlegend: false
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler={true}
            />
          )}
        </div>
      </div>

      {/* Chart 2: Prix option */}
      {state.data.length > 0 && (
        <div className="flex-1 border border-[#222222] flex flex-col">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white"> PRIX DE L'OPTION (BSM à IV prédite)</span>
          </div>
          <div className="flex-1 bg-[#0A0A0A] p-2">
            <Plot
              data={[
                {
                  x: state.data.map(d => d.day),
                  y: state.data.map(d => d.price_hist),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  line: { color: '#D0D0D0', width: 1.5 },
                  name: 'Prix historique',
                  connectgaps: false
                },
                {
                  x: state.data.map(d => d.day),
                  y: state.data.map(d => d.price_forecast),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  line: { color: '#00FF00', width: 2 },
                  name: 'Prix forecast',
                  connectgaps: false
                }
              ]}
              layout={{
                autosize: true,
                margin: { l: 40, r: 20, t: 10, b: 30 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                xaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 } },
                yaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 } },
                hovermode: 'x unified',
                shapes: [
                  {
                    type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: 0, x1: 0,
                    y0: 0, y1: 1,
                    line: { color: '#FFCC00', dash: 'dash' as const, width: 1 }
                  }
                ],
                showlegend: false
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler={true}
            />
          </div>
        </div>
      )}
      </div>

      {/* Chart 3: Delta */}
      {state.data.length > 0 && (
        <div className="flex-1 border border-[#222222] flex flex-col min-h-[160px]">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white"> DELTA (recalculé jour par jour)</span>
          </div>
          <div className="flex-1 bg-[#0A0A0A] p-2">
            <Plot
              data={[
                {
                  x: state.data.map(d => d.day),
                  y: state.data.map(d => d.delta_hist),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  line: { color: '#D0D0D0', width: 1.5 },
                  name: 'Delta historique',
                  connectgaps: false
                },
                {
                  x: state.data.map(d => d.day),
                  y: state.data.map(d => d.delta_forecast),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  line: { color: '#FFCC00', width: 2 },
                  name: 'Delta forecast',
                  connectgaps: false
                }
              ]}
              layout={{
                autosize: true,
                margin: { l: 40, r: 20, t: 10, b: 30 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                xaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 } },
                yaxis: { gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 }, range: [-1, 1] },
                hovermode: 'x unified',
                shapes: [
                  {
                    type: 'line', xref: 'paper', x0: 0, x1: 1,
                    y0: 0, y1: 0,
                    line: { color: '#444444', width: 1 }
                  },
                  {
                    type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: 0, x1: 0,
                    y0: 0, y1: 1,
                    line: { color: '#FFCC00', dash: 'dash' as const, width: 1 }
                  }
                ],
                showlegend: false
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler={true}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────

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

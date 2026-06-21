import React, { useState, useRef } from 'react';
import { useMarket } from '../App';
import Plot from 'react-plotly.js';

interface SurfaceState {
  strikes: number[];
  maturities: number[];
  iv_surface: (number | null)[][];
  loading: boolean;
  error: string | null;
}

export function SurfaceTab() {
  const market = useMarket();
  const tickerRef = useRef<HTMLInputElement>(null);

  const [state, setState] = useState<SurfaceState>({
    strikes: [], maturities: [], iv_surface: [],
    loading: false, error: null,
  });

  const [axisMode, setAxisMode] = useState<'strike' | 'moneyness'>('strike');

  const handleCalculate = async () => {
    const ticker = tickerRef.current?.value.trim().toUpperCase() || market.ticker;
    setState(s => ({ ...s, loading: true, error: null }));

    try {
      if (!window.eel) {
        setState(s => ({ ...s, loading: false, error: 'Eel non disponible' }));
        return;
      }
      const res = await window.eel.calculate_surface(ticker)();
      if (res.error) {
        setState(s => ({ ...s, loading: false, error: res.error }));
        return;
      }
      setState(s => ({
        ...s, loading: false, error: null,
        strikes: res.strikes,
        maturities: res.maturities,
        iv_surface: res.iv_surface,
      }));
    } catch (e: any) {
      setState(s => ({ ...s, loading: false, error: String(e) }));
    }
  };

  const S = market.S ?? 1;
  const xValues = axisMode === 'moneyness' ? state.strikes.map(k => k / S) : state.strikes;

  // Flatten and filter iv_surface to get min/max for stats
  const allIv = state.iv_surface.flat().filter((v): v is number => v !== null);
  const minIv = allIv.length > 0 ? Math.min(...allIv).toFixed(1) : '0.0';
  const maxIv = allIv.length > 0 ? Math.max(...allIv).toFixed(1) : '0.0';

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">

      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] rounded shrink-0">
          Warning {state.error}
        </div>
      )}

      {/* Toolbar */}
      <div className="border border-[#222222] shrink-0">
        <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] justify-between">
          <span className="font-bold text-white"> SURFACE DE VOLATILITÉ IMPLICITE 3D</span>
          <div className="flex items-center gap-2">
            <label className="text-[#888888] flex items-center gap-1">
              Ticker :
              <input ref={tickerRef} defaultValue={market.ticker}
                className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] w-[80px] outline-none ml-1" />
            </label>
            <label className="text-[#888888] flex items-center gap-1">
              Axe X :
              <select value={axisMode} onChange={e => setAxisMode(e.target.value as 'strike' | 'moneyness')}
                className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[11px] outline-none appearance-none ml-1">
                <option value="strike">Strike</option>
                <option value="moneyness">Moneyness (S/K)</option>
              </select>
            </label>
            <button id="surface-calc-btn" onClick={handleCalculate} disabled={state.loading}
              className="bg-[#4A90E2] text-white px-3 py-0.5 hover:bg-[#357ABD] text-[10px] font-bold rounded-sm disabled:opacity-50">
              {state.loading ? 'Calcul...' : 'Calculer Surface IV'}
            </button>
          </div>
        </div>

        {/* Info */}
        {state.strikes.length > 0 && (
          <div className="px-2 py-0.5 bg-[#0A0A0A] text-[9px] text-[#888888] flex gap-4">
            <span>Strikes : {state.strikes.length}</span>
            <span>Maturités : {state.maturities.length}</span>
            <span>IV range : {minIv}% – {maxIv}%</span>
            <span>Z = IV (%)  ·  X = {axisMode === 'moneyness' ? 'Moneyness (S/K)' : 'Strike'}  ·  Y = Maturité (jours)</span>
          </div>
        )}
      </div>

      {/* Plotly 3D */}
      <div className="flex-1 border border-[#222222] flex flex-col min-h-[400px] relative">
        {state.strikes.length === 0 && !state.loading && !state.error && (
          <div className="absolute inset-0 flex items-center justify-center text-[#888888] text-[12px]">
            Entrez un ticker et cliquez sur "Calculer Surface IV"
          </div>
        )}
        {state.loading && (
          <div className="absolute inset-0 flex items-center justify-center text-[#FFCC00] text-[12px] z-10">
            Calcul en cours... (peut prendre quelques secondes)
          </div>
        )}
        {state.strikes.length > 0 && !state.loading && (
          <Plot
            data={[
              {
                z: state.iv_surface,
                x: xValues,
                y: state.maturities,
                type: 'surface',
                colorscale: 'Viridis',
                colorbar: { title: 'IV (%)', thickness: 15, len: 0.5 },
              }
            ]}
            layout={{
              autosize: true,
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#888' },
              scene: {
                xaxis: { title: axisMode === 'moneyness' ? 'Moneyness' : 'Strike', gridcolor: '#333' },
                yaxis: { title: 'Maturity', gridcolor: '#333' },
                zaxis: { title: 'Implied Vol', gridcolor: '#333' },
                camera: {
                  eye: { x: 1.5, y: 1.5, z: 1.2 }
                }
              },
              margin: { l: 0, r: 0, b: 0, t: 0 }
            }}
            useResizeHandler={true}
            style={{ width: '100%', height: '100%' }}
            config={{ responsive: true, displayModeBar: false }}
          />
        )}
      </div>
    </div>
  );
}
import React, { useState, useRef, useCallback } from 'react';
import { useMarket } from '../App';
import Plot from 'react-plotly.js';

interface SimState {
  vols: number[];
  prices: number[];
  matrix: number[][];
  loading: boolean;
  error: string | null;
}

export function SimulationTab() {
  const market = useMarket();
  const strikeRef = useRef<HTMLInputElement>(null);
  const maturityRef = useRef<HTMLInputElement>(null);
  const optTypeRef = useRef<HTMLSelectElement>(null);
  const volMinRef = useRef<HTMLInputElement>(null);
  const volMaxRef = useRef<HTMLInputElement>(null);
  const volStepRef = useRef<HTMLInputElement>(null);
  const sMinRef = useRef<HTMLInputElement>(null);
  const sMaxRef = useRef<HTMLInputElement>(null);
  const sStepRef = useRef<HTMLInputElement>(null);

  const [state, setState] = useState<SimState>({
    vols: [], prices: [], matrix: [], loading: false, error: null,
  });

  const handleRun = useCallback(async () => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const S = market.S ?? 100;
      const K = parseFloat(strikeRef.current?.value || String(Math.round(S)));
      const matStr = maturityRef.current?.value || '';
      const T_days = matStr ? computeDaysFromDate(matStr) : 90;
      const optType = optTypeRef.current?.value || 'call';
      const volMin = parseInt(volMinRef.current?.value || '5');
      const volMax = parseInt(volMaxRef.current?.value || '80');
      const volStep = parseInt(volStepRef.current?.value || '5');
      const sMin = parseInt(sMinRef.current?.value || String(Math.round(S * 0.7)));
      const sMax = parseInt(sMaxRef.current?.value || String(Math.round(S * 1.3)));
      const sStep = parseInt(sStepRef.current?.value || String(Math.round(S * 0.05)));

      if (!(window as any).eel) {
        setState(s => ({ ...s, loading: false, error: 'Eel non disponible' }));
        return;
      }

      const res = await (window as any).eel.run_simulation(
        K, T_days, market.r, market.q,
        volMin, volMax, volStep,
        sMin, sMax, sStep, optType
      )();

      if (res.error) {
        setState(s => ({ ...s, loading: false, error: res.error }));
        return;
      }

      setState(s => ({ ...s, loading: false, error: null, ...res }));
    } catch (e: any) {
      setState(s => ({ ...s, loading: false, error: String(e) }));
    }
  }, [market]);

  const S = market.S ?? 100;
  const defaultStrike = Math.round(S);
  const defaultMat = getDefaultMaturity();

  const textMatrix = state.matrix.map(row => row.map(val => val.toFixed(2)));

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">
      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] rounded shrink-0">
          Warning {state.error}
        </div>
      )}

      {/* Top row: Paramètres */}
      <div className="border border-[#222222] shrink-0">
        <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
          <span className="font-bold text-white"> PARAMÈTRES SIMULATION</span>
        </div>
        <div className="bg-[#000000] p-1.5 grid grid-cols-4 gap-4">
          <div className="space-y-1">
            <FR label="Strike (K)">
              <input ref={strikeRef} key={`k-${defaultStrike}`} defaultValue={defaultStrike} type="number" step="0.5" className={INP} />
            </FR>
            <FR label="Date d'échéance">
              <input ref={maturityRef} defaultValue={defaultMat} type="date" className={INP} />
            </FR>
            <FR label="Type">
              <select ref={optTypeRef} defaultValue="call" className={SEL}>
                <option value="call">call</option>
                <option value="put">put</option>
              </select>
            </FR>
          </div>
          <div className="space-y-1">
            <FR label="Vol Min (%)">
              <input ref={volMinRef} defaultValue="5" type="number" className={INP} />
            </FR>
            <FR label="Vol Max (%)">
              <input ref={volMaxRef} defaultValue="80" type="number" className={INP} />
            </FR>
            <FR label="Pas Vol (%)">
              <input ref={volStepRef} defaultValue="5" type="number" className={INP} />
            </FR>
          </div>
          <div className="space-y-1">
            <FR label="Spot Min ($)">
              <input ref={sMinRef} key={`smin-${S}`} defaultValue={Math.round(S * 0.7)} type="number" className={INP} />
            </FR>
            <FR label="Spot Max ($)">
              <input ref={sMaxRef} key={`smax-${S}`} defaultValue={Math.round(S * 1.3)} type="number" className={INP} />
            </FR>
            <FR label="Pas Spot ($)">
              <input ref={sStepRef} key={`sstep-${S}`} defaultValue={Math.max(1, Math.round(S * 0.05))} type="number" className={INP} />
            </FR>
          </div>
          <div className="flex flex-col justify-end pb-1">
            <button id="sim-run-btn" onClick={handleRun} disabled={state.loading}
              className="w-auto bg-[#4A90E2] text-white py-1.5 hover:bg-[#357ABD] text-[11px] font-bold rounded-sm disabled:opacity-50">
              {state.loading ? 'Calcul...' : 'Simulation'}
            </button>
          </div>
        </div>
      </div>

      {/* Heatmap Area */}
      <div className="flex-1 border border-[#222222] flex flex-col min-h-[300px] relative">
        <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] justify-between">
          <span className="font-bold text-white"> MATRICE DE PRIX BSM</span>
          <div className="text-[#888888] text-[9px]">Axe X = Prix Sous-Jacent ($)  ·  Axe Y = Volatilité Implicite (%)</div>
        </div>

        <div className="flex-1 bg-[#0A0A0A] relative p-1">
          {state.matrix.length === 0 && !state.loading && !state.error && (
            <div className="absolute inset-0 flex items-center justify-center text-[#888888] text-[12px]">
              Configurez les paramètres et lancez la simulation
            </div>
          )}

          {state.matrix.length > 0 && (
            <Plot
              data={[
                {
                  z: state.matrix,
                  x: state.prices,
                  y: state.vols,
                  type: 'heatmap',
                  colorscale: 'Rainbow',
                  text: textMatrix,
                  texttemplate: "%{text}",
                  hoverinfo: 'x+y+z',
                  showscale: false,
                }
              ]}
              layout={{
                autosize: true,
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#888' },
                xaxis: { title: 'Prix Sous-Jacent ($)', gridcolor: '#333' },
                yaxis: { title: 'Volatilité Implicite (%)', gridcolor: '#333', autorange: 'reversed' },
                margin: { l: 50, r: 20, b: 40, t: 20 }
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{ responsive: true, displayModeBar: false }}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Helpers ────────────────────────────────────────────────────────────────

const INP = "w-[60px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none";
const SEL = "w-[60px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] outline-none appearance-none cursor-pointer";

function FR({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-[#222222] pb-1">
      <span className="text-[10px] text-[#888888]">{label}</span>
      {children}
    </div>
  );
}

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

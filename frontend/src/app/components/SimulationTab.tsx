import React, { useState, useRef, useCallback } from 'react';
import { useMarket } from '../App';

interface SimState {
  vols: number[];
  prices: number[];
  matrix: number[][];
  loading: boolean;
  error: string | null;
}

export function SimulationTab() {
  const market = useMarket();
  const strikeRef   = useRef<HTMLInputElement>(null);
  const maturityRef = useRef<HTMLInputElement>(null);
  const optTypeRef  = useRef<HTMLSelectElement>(null);
  const volMinRef   = useRef<HTMLInputElement>(null);
  const volMaxRef   = useRef<HTMLInputElement>(null);
  const volStepRef  = useRef<HTMLInputElement>(null);
  const sMinRef     = useRef<HTMLInputElement>(null);
  const sMaxRef     = useRef<HTMLInputElement>(null);
  const sStepRef    = useRef<HTMLInputElement>(null);

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
      const volMin  = parseInt(volMinRef.current?.value  || '5');
      const volMax  = parseInt(volMaxRef.current?.value  || '80');
      const volStep = parseInt(volStepRef.current?.value || '5');
      const sMin    = parseInt(sMinRef.current?.value    || String(Math.round(S * 0.7)));
      const sMax    = parseInt(sMaxRef.current?.value    || String(Math.round(S * 1.3)));
      const sStep   = parseInt(sStepRef.current?.value   || String(Math.round(S * 0.05)));

      if (!window.eel) {
        setState(s => ({ ...s, loading: false, error: 'Eel non disponible' }));
        return;
      }

      const res = await window.eel.run_simulation(
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

  // Heatmap canvas renderer
  const canvasRef = useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || state.matrix.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    const { vols, prices, matrix } = state;
    const rows = matrix.length;    // vols axis (Y)
    const cols = matrix[0]?.length || 0; // prices axis (X)
    if (rows === 0 || cols === 0) return;

    const W = canvas.width;
    const H = canvas.height;
    const padL = 50, padB = 30, padT = 10, padR = 60;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const cellW = plotW / cols;
    const cellH = plotH / rows;

    // Plages de valeurs
    let minV = Infinity, maxV = -Infinity;
    matrix.forEach(row => row.forEach(v => { if (v < minV) minV = v; if (v > maxV) maxV = v; }));

    // Dessin heatmap (vol en Y : bas = vol min, haut = vol max)
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const norm = maxV > minV ? (matrix[i][j] - minV) / (maxV - minV) : 0;
        ctx.fillStyle = heatColor(norm);
        const x = padL + j * cellW;
        const y = padT + (rows - 1 - i) * cellH;  // inverser axe Y
        ctx.fillRect(x, y, cellW + 0.5, cellH + 0.5);
      }
    }

    // Axes
    ctx.fillStyle = '#666666';
    ctx.font = '9px monospace';
    ctx.textAlign = 'center';

    // X axis (prix sous-jacent)
    const xStep = Math.max(1, Math.floor(cols / 6));
    for (let j = 0; j < cols; j += xStep) {
      const x = padL + j * cellW + cellW / 2;
      ctx.fillText(String(prices[j] ?? ''), x, H - padB + 12);
    }

    // Y axis (volatilité %)
    ctx.textAlign = 'right';
    const yStep = Math.max(1, Math.floor(rows / 6));
    for (let i = 0; i < rows; i += yStep) {
      const y = padT + (rows - 1 - i) * cellH + cellH / 2 + 3;
      ctx.fillText(`${vols[i]}%`, padL - 4, y);
    }

    // Titre axes
    ctx.fillStyle = '#888888';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Prix sous-jacent (S)', padL + plotW / 2, H - 4);

    // Légende colour bar
    const barH = plotH * 0.7;
    const barX = W - padR + 8;
    const barY = padT + (plotH - barH) / 2;
    for (let p = 0; p < barH; p++) {
      ctx.fillStyle = heatColor(1 - p / barH);
      ctx.fillRect(barX, barY + p, 14, 1);
    }
    ctx.strokeStyle = '#333333';
    ctx.strokeRect(barX, barY, 14, barH);
    ctx.fillStyle = '#888888';
    ctx.textAlign = 'left';
    ctx.font = '9px monospace';
    ctx.fillText(`$${maxV.toFixed(0)}`, barX + 16, barY + 10);
    ctx.fillText(`$${((minV + maxV) / 2).toFixed(0)}`, barX + 16, barY + barH / 2 + 4);
    ctx.fillText(`$${minV.toFixed(0)}`, barX + 16, barY + barH);
  }, [state.matrix]);

  const defaultStrike = market.S ? Math.round(market.S) : 100;
  const defaultSMin   = market.S ? Math.round(market.S * 0.7) : 70;
  const defaultSMax   = market.S ? Math.round(market.S * 1.3) : 130;
  const defaultSStep  = market.S ? Math.round(market.S * 0.05) : 5;

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">

      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] rounded">⚠ {state.error}</div>
      )}

      {/* Paramètres */}
      <div className="border border-[#222222]">
        <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] justify-between">
          <span className="font-bold text-white">▼ SIMULATION — PARAMÈTRES</span>
          <button id="sim-run-btn" onClick={handleRun} disabled={state.loading}
            className="bg-[#4A90E2] text-white px-3 py-0.5 hover:bg-[#357ABD] text-[10px] font-bold rounded-sm disabled:opacity-50">
            {state.loading ? '⏳ Calcul...' : '▶ Lancer la Simulation'}
          </button>
        </div>
        <div className="bg-[#000000] p-2 grid grid-cols-3 gap-x-6 gap-y-1.5">
          {/* Colonne 1 : Option */}
          <div className="space-y-1.5">
            <p className="text-[9px] text-[#4A90E2] font-bold uppercase tracking-widest mb-2">Option</p>
            <FR label="Strike (K)">
              <input ref={strikeRef} key={defaultStrike} defaultValue={defaultStrike} type="number" step="1" className={INP} />
            </FR>
            <FR label="Maturité">
              <input ref={maturityRef} defaultValue={getDefaultMaturity()} type="date" className={INP} />
            </FR>
            <FR label="Type">
              <select ref={optTypeRef} defaultValue="call" className={SEL}>
                <option value="call">call</option>
                <option value="put">put</option>
              </select>
            </FR>
          </div>
          {/* Colonne 2 : Volatilité */}
          <div className="space-y-1.5">
            <p className="text-[9px] text-[#4A90E2] font-bold uppercase tracking-widest mb-2">Axe Volatilité (σ)</p>
            <FR label="σ Min (%)"><input ref={volMinRef} defaultValue="5" type="number" step="1" className={INP} /></FR>
            <FR label="σ Max (%)"><input ref={volMaxRef} defaultValue="80" type="number" step="1" className={INP} /></FR>
            <FR label="Pas (%)">  <input ref={volStepRef} defaultValue="5" type="number" step="1" min="1" className={INP} /></FR>
          </div>
          {/* Colonne 3 : Sous-jacent */}
          <div className="space-y-1.5">
            <p className="text-[9px] text-[#4A90E2] font-bold uppercase tracking-widest mb-2">Axe Sous-jacent (S)</p>
            <FR label="S Min ($)"><input ref={sMinRef} key={defaultSMin} defaultValue={defaultSMin} type="number" step="1" className={INP} /></FR>
            <FR label="S Max ($)"><input ref={sMaxRef} key={defaultSMax} defaultValue={defaultSMax} type="number" step="1" className={INP} /></FR>
            <FR label="Pas ($)"> <input ref={sStepRef} key={defaultSStep} defaultValue={defaultSStep} type="number" step="1" min="1" className={INP} /></FR>
          </div>
        </div>
      </div>

      {/* Heatmap */}
      <div className="flex-1 border border-[#222222] flex flex-col min-h-[350px]">
        <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] justify-between">
          <span className="font-bold text-white">▼ HEATMAP PRIX — σ (Axe Y) × S (Axe X)</span>
          <div className="flex gap-4 text-[10px] text-[#888888]">
            {state.matrix.length > 0 && (
              <span>{state.vols.length} σ × {state.prices.length} S = {state.vols.length * state.prices.length} scénarios</span>
            )}
            <span>Couleur : Prix de l'option ($)</span>
          </div>
        </div>
        <div className="flex-1 bg-[#0A0A0A] relative p-1">
          {state.matrix.length === 0 ? (
            <div className="flex items-center justify-center h-full text-[#888888] text-[12px]">
              Configurez les paramètres et cliquez sur "Lancer la Simulation"
            </div>
          ) : (
            <canvas ref={canvasRef} className="w-full h-full" />
          )}
        </div>
      </div>

      {/* Tableau de résultats (sous-échantillon) */}
      {state.matrix.length > 0 && (
        <div className="border border-[#222222]">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white">▼ TABLEAU — PRIX PAR SCÉNARIO ($)</span>
          </div>
          <div className="bg-[#000000] overflow-auto max-h-[200px]">
            <table className="w-full text-right border-collapse text-[10px]">
              <thead>
                <tr className="bg-[#111111] text-[9px] text-[#888888] divide-x divide-[#222222] border-b border-[#222222] sticky top-0">
                  <th className="py-1 px-2 font-normal text-left">σ \ S</th>
                  {state.prices.map(p => (
                    <th key={p} className="py-1 px-2 font-normal">{p}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {state.vols.map((v, i) => (
                  <tr key={v} className="divide-x divide-[#111111] border-b border-[#111111] hover:bg-[#111111]">
                    <td className="py-0.5 px-2 text-[#4A90E2] text-left">{v}%</td>
                    {state.matrix[i]?.map((price, j) => (
                      <td key={j} className="py-0.5 px-2 text-[#D4D4D4]">{price.toFixed(2)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

const INP = "w-[110px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none";
const SEL = "w-[110px] bg-[#121212] border border-[#333333] text-[#FFFFFF] py-0.5 px-1 text-[11px] text-right outline-none appearance-none";

function FR({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-[#1A1A1A] pb-1">
      <span className="text-[10px] text-[#888888]">{label}</span>
      {children}
    </div>
  );
}

function heatColor(t: number): string {
  let r: number, g: number, b: number;
  if (t < 0.25) {
    const p = t / 0.25;
    r = 0; g = Math.floor(p * 100); b = Math.floor(80 + p * 175);
  } else if (t < 0.5) {
    const p = (t - 0.25) / 0.25;
    r = 0; g = Math.floor(100 + p * 155); b = Math.floor(255 * (1 - p));
  } else if (t < 0.75) {
    const p = (t - 0.5) / 0.25;
    r = Math.floor(p * 255); g = 255; b = 0;
  } else {
    const p = (t - 0.75) / 0.25;
    r = 255; g = Math.floor(255 * (1 - p)); b = 0;
  }
  return `rgb(${r},${g},${b})`;
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

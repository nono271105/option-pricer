import React, { useState, useRef, useEffect } from 'react';
import { useMarket } from '../App';

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
  const canvasRef = useRef<HTMLCanvasElement>(null);

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

  // Render 3D surface on canvas using painter's algorithm
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { strikes, maturities, iv_surface } = state;
    if (strikes.length === 0 || maturities.length === 0 || iv_surface.length === 0) {
      // Draw placeholder
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const W = canvas.width = canvas.clientWidth;
    const H = canvas.height = canvas.clientHeight;

    const rows = maturities.length;
    const cols = strikes.length;

    // Compute X axis values
    const S = market.S ?? 1;
    const xValues = axisMode === 'moneyness'
      ? strikes.map(k => k / S)
      : strikes;

    // Normalize
    const xMin = Math.min(...xValues), xMax = Math.max(...xValues);
    const yMin = Math.min(...maturities), yMax = Math.max(...maturities);
    let zMin = Infinity, zMax = -Infinity;
    iv_surface.forEach(row => row.forEach(v => {
      if (v !== null) { if (v < zMin) zMin = v; if (v > zMax) zMax = v; }
    }));

    // Project (isometric-like)
    const angle = 0.55;
    const pitch = 0.45;
    const cosA = Math.cos(angle), sinA = Math.sin(angle);
    const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    const cx = W * 0.5, cy = H * 0.45;
    const scaleX = W * 0.32, scaleY = H * 0.32, scaleZ = H * 0.25;

    function project(xi: number, yi: number, zi: number) {
      const nx = (xi - xMin) / (xMax - xMin) * 2 - 1;
      const ny = (yi - yMin) / (yMax - yMin) * 2 - 1;
      const nz = zMax > zMin ? (zi - zMin) / (zMax - zMin) : 0;
      const rx = nx * cosA - ny * sinA;
      const ry = nx * sinA + ny * cosA;
      const rz = nz;
      const py = ry * cosP - rz * sinP;
      return {
        px: cx + rx * scaleX,
        py: cy + py * scaleY - rz * scaleZ,
        nz,
      };
    }

    // Plasma colorscale
    function plasmaColor(t: number): string {
      t = Math.max(0, Math.min(1, t));
      let r, g, b;
      if (t < 0.25) { const p = t / 0.25; r = Math.floor(13 + p * 120); g = Math.floor(8 + p * 12); b = Math.floor(135 + p * 80); }
      else if (t < 0.5) { const p = (t - 0.25) / 0.25; r = Math.floor(133 + p * 80); g = Math.floor(20 + p * 15); b = Math.floor(215 - p * 40); }
      else if (t < 0.75) { const p = (t - 0.5) / 0.25; r = Math.floor(213 + p * 30); g = Math.floor(35 + p * 100); b = Math.floor(175 - p * 100); }
      else { const p = (t - 0.75) / 0.25; r = 243; g = Math.floor(135 + p * 100); b = Math.floor(75 - p * 40); }
      return `rgb(${Math.min(255, r)},${Math.min(255, g)},${Math.min(255, b)})`;
    }

    // Build quads
    const quads: {
      p1: { px: number; py: number };
      p2: { px: number; py: number };
      p3: { px: number; py: number };
      p4: { px: number; py: number };
      depth: number; avgNz: number;
    }[] = [];

    for (let i = 0; i < rows - 1; i++) {
      for (let j = 0; j < cols - 1; j++) {
        const z00 = iv_surface[i]?.[j] ?? zMin;
        const z01 = iv_surface[i]?.[j + 1] ?? zMin;
        const z10 = iv_surface[i + 1]?.[j] ?? zMin;
        const z11 = iv_surface[i + 1]?.[j + 1] ?? zMin;
        const avgZ = (z00 + z01 + z10 + z11) / 4;
        const avgNz = zMax > zMin ? (avgZ - zMin) / (zMax - zMin) : 0;

        const p1 = project(xValues[j],     maturities[i],     z00);
        const p2 = project(xValues[j + 1], maturities[i],     z01);
        const p3 = project(xValues[j + 1], maturities[i + 1], z11);
        const p4 = project(xValues[j],     maturities[i + 1], z10);

        const depth = p1.py + p2.py + p3.py + p4.py;
        quads.push({ p1, p2, p3, p4, depth, avgNz });
      }
    }

    quads.sort((a, b) => b.depth - a.depth);

    ctx.clearRect(0, 0, W, H);
    ctx.lineWidth = 0.4;
    quads.forEach(q => {
      ctx.beginPath();
      ctx.moveTo(q.p1.px, q.p1.py);
      ctx.lineTo(q.p2.px, q.p2.py);
      ctx.lineTo(q.p3.px, q.p3.py);
      ctx.lineTo(q.p4.px, q.p4.py);
      ctx.closePath();
      ctx.fillStyle = plasmaColor(q.avgNz);
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.3)';
      ctx.stroke();
    });

    // Axis labels
    ctx.font = '9px monospace';
    ctx.fillStyle = '#888888';
    ctx.textAlign = 'center';
    // Strike axis labels (every ~4 cols)
    for (let j = 0; j < cols; j += Math.max(1, Math.floor(cols / 5))) {
      const pt = project(xValues[j], maturities[0], zMin);
      ctx.fillText(
        axisMode === 'moneyness' ? xValues[j].toFixed(2) : xValues[j].toFixed(0),
        pt.px, pt.py + 14
      );
    }
    // Maturity axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i < rows; i += Math.max(1, Math.floor(rows / 4))) {
      const pt = project(xValues[0], maturities[i], zMin);
      ctx.fillText(`${maturities[i].toFixed(0)}d`, pt.px - 6, pt.py + 4);
    }
    // IV axis labels
    ctx.textAlign = 'right';
    for (let v = zMin; v <= zMax; v += (zMax - zMin) / 4) {
      const pt = project(xValues[0], maturities[0], v);
      ctx.fillText(`${v.toFixed(0)}%`, pt.px - 4, pt.py);
    }

    // Colour bar
    const barH = H * 0.5;
    const barX = W - 40;
    const barY = H * 0.1;
    for (let p = 0; p < barH; p++) {
      ctx.fillStyle = plasmaColor(1 - p / barH);
      ctx.fillRect(barX, barY + p, 10, 1);
    }
    ctx.strokeStyle = '#333333';
    ctx.strokeRect(barX, barY, 10, barH);
    ctx.fillStyle = '#888888';
    ctx.textAlign = 'left';
    ctx.fillText(`${zMax.toFixed(1)}%`, barX + 13, barY + 10);
    ctx.fillText(`${((zMin + zMax) / 2).toFixed(1)}%`, barX + 13, barY + barH / 2 + 4);
    ctx.fillText(`${zMin.toFixed(1)}%`, barX + 13, barY + barH);

  }, [state.iv_surface, state.strikes, state.maturities, axisMode, market.S]);

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">

      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] rounded shrink-0">
          ⚠ {state.error}
        </div>
      )}

      {/* Toolbar */}
      <div className="border border-[#222222] shrink-0">
        <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] justify-between">
          <span className="font-bold text-white">▼ SURFACE DE VOLATILITÉ IMPLICITE 3D</span>
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
              {state.loading ? '⏳ Calcul...' : 'Calculer Surface IV'}
            </button>
          </div>
        </div>

        {/* Info */}
        {state.strikes.length > 0 && (
          <div className="px-2 py-0.5 bg-[#0A0A0A] text-[9px] text-[#888888] flex gap-4">
            <span>Strikes : {state.strikes.length}</span>
            <span>Maturités : {state.maturities.length}</span>
            <span>IV range : {Math.min(...state.iv_surface.flat().filter(v => v !== null) as number[]).toFixed(1)}% – {Math.max(...state.iv_surface.flat().filter(v => v !== null) as number[]).toFixed(1)}%</span>
            <span>Z = IV (%)  ·  X = {axisMode === 'moneyness' ? 'Moneyness (S/K)' : 'Strike'}  ·  Y = Maturité (jours)</span>
          </div>
        )}
      </div>

      {/* Canvas 3D */}
      <div className="flex-1 border border-[#222222] flex flex-col min-h-[400px] relative">
        {state.strikes.length === 0 && !state.loading && !state.error && (
          <div className="absolute inset-0 flex items-center justify-center text-[#888888] text-[12px]">
            Entrez un ticker et cliquez sur "Calculer Surface IV"
          </div>
        )}
        {state.loading && (
          <div className="absolute inset-0 flex items-center justify-center text-[#FFCC00] text-[12px]">
            ⏳ Calcul de la surface... (peut prendre quelques secondes)
          </div>
        )}
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
    </div>
  );
}
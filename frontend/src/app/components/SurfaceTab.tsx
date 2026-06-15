import React, { useEffect, useRef } from 'react';

export function SurfaceTab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = (canvas.width = canvas.clientWidth);
    let height = (canvas.height = canvas.clientHeight);

    // Simple 3D projection parameters
    const size = 30;
    const points: { x: number; y: number; z: number }[] = [];

    // Generate data (Strike vs Time to Maturity vs IV)
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const x = (i / (size - 1)) * 2 - 1;
        const y = (j / (size - 1)) * 2 - 1;
        // Volatility smile/surface formula (mock)
        const z = 0.2 + 0.5 * x * x + 0.1 * y - 0.2 * x * y * y;
        points.push({ x, y, z });
      }
    }

    // Plasma-like colorscale
    function getPlasmaColor(t: number) {
      let r: number, g: number, b: number;
      if (t < 0.2) {
        const p = t / 0.2;
        r = Math.floor(13 + p * 70); g = Math.floor(8 + p * 15); b = Math.floor(135 + p * 80);
      } else if (t < 0.4) {
        const p = (t - 0.2) / 0.2;
        r = Math.floor(83 + p * 100); g = Math.floor(23 + p * 10); b = Math.floor(215 - p * 30);
      } else if (t < 0.6) {
        const p = (t - 0.4) / 0.2;
        r = Math.floor(183 + p * 50); g = Math.floor(33 + p * 70); b = Math.floor(185 - p * 80);
      } else if (t < 0.8) {
        const p = (t - 0.6) / 0.2;
        r = Math.floor(233 + p * 22); g = Math.floor(103 + p * 80); b = Math.floor(105 - p * 60);
      } else {
        const p = (t - 0.8) / 0.2;
        r = 255; g = Math.floor(183 + p * 60); b = Math.floor(45 + p * 30);
      }
      return `rgb(${Math.min(255, r)},${Math.min(255, g)},${Math.min(255, b)})`;
    }

    const angle = 0.6;

    function render() {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const scaleX = width * 0.35;
      const scaleY = height * 0.35;
      const scaleZ = height * 0.25;

      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const pitch = 0.5;
      const cosP = Math.cos(pitch);
      const sinP = Math.sin(pitch);

      const projected = points.map((p) => {
        let rx = p.x * cosA - p.y * sinA;
        let ry = p.x * sinA + p.y * cosA;
        let rz = p.z;
        let py = ry * cosP - rz * sinP;
        return {
          px: cx + rx * scaleX,
          py: cy + py * scaleY - rz * scaleZ,
          zValue: p.z,
        };
      });

      // Sort polygons by depth (Painter's algorithm)
      const quads: { p1: any; p2: any; p3: any; p4: any; depth: number; avgZ: number }[] = [];
      for (let i = 0; i < size - 1; i++) {
        for (let j = 0; j < size - 1; j++) {
          const i1 = i * size + j;
          const i2 = i * size + j + 1;
          const i3 = (i + 1) * size + j + 1;
          const i4 = (i + 1) * size + j;

          const depth = projected[i1].py + projected[i2].py + projected[i3].py + projected[i4].py;
          const avgZ = (projected[i1].zValue + projected[i2].zValue + projected[i3].zValue + projected[i4].zValue) / 4;

          quads.push({ p1: projected[i1], p2: projected[i2], p3: projected[i3], p4: projected[i4], depth, avgZ });
        }
      }

      quads.sort((a, b) => b.depth - a.depth);

      const minZ = 0.2;
      const maxZ = 0.9;

      ctx.lineWidth = 0.5;
      quads.forEach((q) => {
        ctx.beginPath();
        ctx.moveTo(q.p1.px, q.p1.py);
        ctx.lineTo(q.p2.px, q.p2.py);
        ctx.lineTo(q.p3.px, q.p3.py);
        ctx.lineTo(q.p4.px, q.p4.py);
        ctx.closePath();

        const normZ = (q.avgZ - minZ) / (maxZ - minZ);
        ctx.fillStyle = getPlasmaColor(Math.min(1, Math.max(0, normZ)));
        ctx.fill();

        ctx.strokeStyle = '#000000';
        ctx.stroke();
      });
    }

    render();

    const handleResize = () => {
      width = canvas.width = canvas.clientWidth;
      height = canvas.height = canvas.clientHeight;
      render();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="flex flex-col h-full gap-2 p-2 overflow-auto">
      <div className="flex-1 border border-border flex flex-col min-h-[500px]">
        <div className="bg-panel-header px-3 py-1.5 text-[11px] uppercase tracking-wider text-[#FFFFFF] border-b border-border flex justify-between items-center">
          <span>Surface de Volatilité Implicite 3D</span>
          <div className="flex gap-3 items-center">
            <label className="text-[10px] text-[#888888] flex items-center gap-2 normal-case tracking-normal">
              Axe :
              <select className="bg-[#1E1E1E] border border-border text-[#FFFFFF] py-1 px-2 text-[11px] focus:border-[#4A90E2] outline-none">
                <option>Moneyness (S/K)</option>
                <option>Strike</option>
              </select>
            </label>
            <button className="bg-[#1E1E1E] border border-border text-[#D0D0D0] px-3 py-1 text-[10px] hover:border-[#4A90E2] hover:text-[#4A90E2] transition-colors uppercase tracking-wider">
              Export HTML
            </button>
          </div>
        </div>
        <div className="flex-1 bg-[#000000] relative">
          <div className="absolute top-3 left-3 z-10 text-[10px] text-[#888888] bg-[#000000]/80 p-2 border border-[#333333] space-y-0.5">
            <div>Z = Volatilité Implicite (σ)</div>
            <div>X = Moneyness (S/K)</div>
            <div>Y = Maturité (T)</div>
            <div className="pt-1 text-[9px] text-[#888888]">Interpolation : Griddata cubique</div>
          </div>
          <canvas ref={canvasRef} className="w-full h-full cursor-move" />
        </div>
      </div>
    </div>
  );
}
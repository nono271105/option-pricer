import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
  BarChart,
  Bar,
  AreaChart,
  Area,
} from 'recharts';

const NUM_PATHS = 20;
const NUM_STEPS = 50;

function generateMonteCarloPaths() {
  const paths: number[][] = [];
  for (let p = 0; p < NUM_PATHS; p++) {
    let spot = 254.23;
    const path = [spot];
    for (let i = 1; i <= NUM_STEPS; i++) {
      const change = (Math.random() - 0.48) * 20;
      spot += change;
      path.push(spot);
    }
    paths.push(path);
  }

  const data = [];
  for (let i = 0; i <= NUM_STEPS; i++) {
    const pt: any = { step: i };
    paths.forEach((p, idx) => {
      pt[`path_${idx}`] = p[i];
    });
    // Average
    pt.avg = paths.reduce((s, p) => s + p[i], 0) / paths.length;
    data.push(pt);
  }
  return data;
}

const DIST_DATA = [
  { bucket: '18', count: 12 },
  { bucket: '19', count: 22 },
  { bucket: '20', count: 38 },
  { bucket: '21', count: 55 },
  { bucket: '22', count: 42 },
  { bucket: '23', count: 28 },
  { bucket: '24', count: 18 },
  { bucket: '25', count: 10 },
  { bucket: '26', count: 5 },
  { bucket: '27', count: 3 },
  { bucket: '28', count: 1 },
];

const PAYOFF_PROFILE = Array.from({ length: 60 }, (_, i) => {
  const s = 150 + i * 4;
  const k = 280;
  const payoff = Math.max(0, k - s); // put payoff (digital example)
  return { spot: s, payoff: payoff > 0 ? 28 : 0 };
});

export function ExoticsTab() {
  const pathData = useMemo(() => generateMonteCarloPaths(), []);

  return (
    <div className="flex flex-col h-full gap-2 p-2 overflow-auto">
      {/* Top row: Params + Données + Résultats */}
      <div className="flex gap-2">
        {/* PARAMÈTRES EXOTIQUES */}
        <div className="border border-border flex-shrink-0 w-[420px] flex flex-col">
          <div className="bg-panel-header px-3 py-1.5 text-[11px] uppercase tracking-wider text-[#FFFFFF] border-b border-border">
            PARAMÈTRES EXOTIQUES
          </div>
          <div className="bg-[#000000] p-2 space-y-2">
            <FormRow label="Ticker Symbole :" value="AAPL" />
            <FormRow label="Type exotique :" value="Digitale / Binaire" isSelect options={['Barrière (Knock-In/Out)', 'Asiatique', 'Lookback', 'Digitale / Binaire']} />
            <FormRow label="Type d'option :" value="put" isSelect options={['call', 'put']} />
            <FormRow label="Prix d'exercice (K) :" value="280" />
            <FormRow label="Date d'échéance :" value="19/03/2027" />
            <FormRow label="Position :" value="long" isSelect options={['long', 'short']} />
          </div>

          <div className="border-t border-border">
            <div className="bg-panel-header px-3 py-1.5 text-[11px] uppercase tracking-wider text-[#FFFFFF] border-b border-border">
              Paramètres spécifiques
            </div>
            <div className="bg-[#000000] p-2 space-y-2">
              <FormRow label="Montant payoff ($) :" value="28" />
            </div>
          </div>

          <div className="border-t border-border">
            <div className="bg-panel-header px-3 py-1.5 text-[11px] uppercase tracking-wider text-[#FFFFFF] border-b border-border">
              Monte Carlo
            </div>
            <div className="bg-[#000000] p-2 space-y-2">
              <FormRow label="Simulations :" value="50000" />
              <FormRow label="Pas de temps :" value="252" />
            </div>
          </div>

          <div className="bg-card p-4 space-y-2 border-t border-border">
            <button className="w-full bg-[#1E1E1E] border border-border text-[#D0D0D0] py-2 hover:border-[#4A90E2] hover:text-[#4A90E2] transition-colors text-[12px]">
              Récupérer/Synchroniser les Données
            </button>
            <button className="w-full bg-[#4A90E2] text-[#000000] py-2 hover:bg-[#357ABD] transition-colors text-[12px] font-semibold">
              Calculer (Analytique + Monte Carlo)
            </button>
          </div>
        </div>

        {/* Right: Données + Résultats */}
        <div className="flex-1 flex flex-col gap-2">
          {/* DONNÉES MARCHÉ */}
          <div className="border border-border">
            <div className="bg-panel-header px-3 py-1.5 text-[11px] uppercase tracking-wider text-[#FFFFFF] border-b border-border">
              DONNÉES MARCHÉ
            </div>
            <div className="bg-[#000000] p-2">
              <DataRow label="Prix Actuel (S) :" value="254.23" />
              <DataRow label="Taux Sans Risque SOFR (r) :" value="3.70%" />
              <DataRow label="Rendement Dividende (q) :" value="0.41%" />
              <DataRow label="Volatilité Utilisée (σ) :" value="24.39% (IV Marché)" highlight />
            </div>
          </div>

          {/* RÉSULTATS PRICING */}
          <div className="border border-border">
            <div className="bg-panel-header px-3 py-1.5 text-[11px] uppercase tracking-wider text-[#FFFFFF] border-b border-border">
              RÉSULTATS PRICING
            </div>
            <div className="bg-[#000000] p-2">
              <DataRow label="Prix Analytique :" value="$17.5075" highlight />
              <DataRow label="Prix Monte Carlo :" value="$17.5425" />
              <DataRow label="Std Error MC :" value="±0.0575" />
              <DataRow label="Écart Ana. / MC :" value="$0.0350 (0.20%)" />
            </div>
          </div>

          {/* Visualisation Monte Carlo (Moved up) */}
          <div className="border border-border flex flex-col flex-1 min-h-[250px]">
            <div className="bg-[#2D2D2D] px-3 py-1 text-[10px] text-[#888888] border-b border-border flex justify-between">
              <span>Analyse Options Exotiques — Trajectoires GBM</span>
              <div className="flex gap-3">
                <div className="flex items-center gap-1"><div className="w-2.5 h-[2px] bg-[#FFCC00]" /><span>Moyenne</span></div>
                <div className="flex items-center gap-1"><div className="w-2.5 h-[2px] bg-[#FF4444] border-t border-dashed" /><span>S₀</span></div>
                <div className="flex items-center gap-1"><div className="w-2.5 h-[2px] bg-[#888888]" style={{ borderTop: '1px dotted #888888' }} /><span>K</span></div>
              </div>
            </div>
            <div className="flex-1 bg-card p-2 relative">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={pathData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                  <CartesianGrid stroke="#333333" vertical={false} />
                  <XAxis
                    dataKey="step"
                    stroke="#333333"
                    tick={{ fill: '#888888', fontSize: 9 }}
                    label={{ value: 'Jours', position: 'insideBottom', offset: -12, fill: '#888888', fontSize: 10 }}
                  />
                  <YAxis
                    domain={['auto', 'auto']}
                    stroke="#333333"
                    tick={{ fill: '#888888', fontSize: 9 }}
                  />
                  <ReferenceLine y={254.23} stroke="#FF4444" strokeDasharray="4 4" />
                  <ReferenceLine y={280} stroke="#888888" strokeDasharray="2 4" />
                  {Array.from({ length: NUM_PATHS }).map((_, i) => (
                    <Line
                      key={i}
                      type="monotone"
                      dataKey={`path_${i}`}
                      stroke="#4A90E2"
                      strokeOpacity={0.15}
                      strokeWidth={1}
                      dot={false}
                      isAnimationActive={false}
                    />
                  ))}
                  <Line type="monotone" dataKey="avg" stroke="#FFCC00" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>


        <div className="flex gap-2 flex-1 min-h-[200px]">
          {/* Distribution payoffs */}
          <div className="flex-1 border border-border flex flex-col">
            <div className="bg-[#2D2D2D] px-3 py-1 text-[10px] text-[#FFFFFF] border-b border-border">
              Distribution payoffs — 65.0% ITM
            </div>
            <div className="flex-1 bg-card p-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={DIST_DATA} margin={{ top: 10, right: 0, left: 0, bottom: 20 }}>
                  <CartesianGrid stroke="#333333" vertical={false} />
                  <XAxis
                    dataKey="bucket"
                    stroke="#333333"
                    tick={{ fill: '#888888', fontSize: 9 }}
                    label={{ value: 'Payoff ($)', position: 'insideBottom', offset: -12, fill: '#888888', fontSize: 10 }}
                  />
                  <YAxis stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }} label={{ value: 'Densité', angle: -90, position: 'insideLeft', offset: 10, fill: '#888888', fontSize: 10 }} />
                  <Bar dataKey="count" fill="#4A90E2" opacity={0.5} />
                  <ReferenceLine x="21" stroke="#FFCC00" strokeDasharray="3 3" label={{ value: 'Moyenne = 18.206', fill: '#FFCC00', fontSize: 9, position: 'insideBottomRight' }} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Profil de payoff à maturité */}
          <div className="flex-1 border border-border flex flex-col">
            <div className="bg-[#2D2D2D] px-3 py-1 text-[10px] text-[#FFFFFF] border-b border-border">
              Profil de payoff à maturité
            </div>
            <div className="flex-1 bg-card p-2">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={PAYOFF_PROFILE} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                  <defs>
                    <linearGradient id="exoticPayoff" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#4A90E2" stopOpacity={0.15} />
                      <stop offset="100%" stopColor="#4A90E2" stopOpacity={0.01} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#333333" vertical={false} />
                  <XAxis
                    dataKey="spot"
                    stroke="#333333"
                    tick={{ fill: '#888888', fontSize: 9 }}
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    label={{ value: 'Prix sous-jacent ($)', position: 'insideBottom', offset: -12, fill: '#888888', fontSize: 10 }}
                  />
                  <YAxis stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }} label={{ value: 'Payoff ($)', angle: -90, position: 'insideLeft', offset: 10, fill: '#888888', fontSize: 10 }} />
                  <ReferenceLine x={280} stroke="#FF4444" strokeDasharray="3 3" label={{ value: 'K = 280.00', fill: '#FF4444', fontSize: 9, position: 'top' }} />
                  <ReferenceLine x={254.23} stroke="#FF4444" strokeDasharray="3 3" label={{ value: 'S₀ = 254.23', fill: '#FFCC00', fontSize: 9, position: 'top' }} />
                  <Area type="stepAfter" dataKey="payoff" stroke="#4A90E2" strokeWidth={1.5} fill="url(#exoticPayoff)" isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
  );
}

/* ── Helpers ── */
function FormRow({ label, value, isSelect, options }: { label: string; value: string; isSelect?: boolean; options?: string[] }) {
  return (
    <div className="flex flex-col gap-1 border-b border-border pb-1">
      <span className="text-[9px] text-[#888888] uppercase text-center tracking-wider">{label.replace(' :', '')}</span>
      {isSelect ? (
        <select
          defaultValue={value}
          className="w-full bg-[#000000] border border-border text-[#FFFFFF] py-1 px-2 text-[12px] text-center focus:border-[#4A90E2] outline-none appearance-none cursor-pointer"
        >
          {options?.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input
          type="text"
          defaultValue={value}
          className="w-full bg-[#000000] border border-border text-[#FFFFFF] py-1 px-2 text-[12px] text-center focus:border-[#4A90E2] outline-none"
        />
      )}
    </div>
  );
}

function DataRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center gap-0.5 border-b border-border py-1.5">
      <span className="text-[9px] text-[#888888] uppercase tracking-wider text-center">{label.replace(' :', '')}</span>
      <span className={`text-[12px] font-bold text-center ${highlight ? 'text-[#4A90E2]' : 'text-[#FFFFFF]'}`}>{value}</span>
    </div>
  );
}
import React, { useState } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

/* ── Strategy definitions ── */
const STRATEGY_FAMILIES: Record<string, string[]> = {
  'Positions de base': ['Long Call', 'Short Call', 'Long Put', 'Short Put'],
  'Spreads directionnels': ['Bull Call Spread', 'Bear Call Spread', 'Bull Put Spread', 'Bear Put Spread'],
  'Volatilité': ['Long Straddle', 'Short Straddle', 'Long Strangle', 'Short Strangle'],
  'Butterflies': ['Long Call Butterfly', 'Short Call Butterfly', 'Long Put Butterfly', 'Short Put Butterfly', 'Long Iron Butterfly', 'Short Iron Butterfly'],
  'Condors': ['Long Call Condor', 'Short Call Condor', 'Long Put Condor', 'Short Put Condor', 'Long Iron Condor', 'Short Iron Condor'],
};

/* ── Static Bull Call Spread legs ── */
const STRATEGY_LEGS = [
  { id: 1, position: 'Long', qty: 1, type: 'Call', strike: 275, maturity: '17/01/2026', premium: 12.50, iv: '23.1%' },
  { id: 2, position: 'Short', qty: 1, type: 'Call', strike: 295, maturity: '17/01/2026', premium: 4.20, iv: '21.8%' },
];

const PAYOFF_DATA = Array.from({ length: 80 }, (_, i) => {
  const s = 240 + i * 2;
  const leg1 = Math.max(0, s - 275) - 12.50;
  const leg2 = -(Math.max(0, s - 295)) + 4.20;
  return { spot: s, payoff: leg1 + leg2 };
});

export function StrategiesTab() {
  const [family, setFamily] = useState('Spreads directionnels');
  const [strategy, setStrategy] = useState('Bull Call Spread');

  const netCost = STRATEGY_LEGS.reduce((s, l) => s + (l.position === 'Long' ? -l.premium : l.premium), 0);

  return (
    <div className="flex flex-col h-full gap-2 p-2 overflow-auto">
      {/* Strategy selector */}
      <div className="border border-border">
        <div className="bg-panel-header px-3 py-1.5 text-[11px] uppercase tracking-wider text-[#FFFFFF] border-b border-border flex justify-between items-center">
          <span>Stratégies — Construction et Analyse</span>
          <div className="flex gap-3 items-center">
            <label className="text-[10px] text-[#888888] flex items-center gap-2 normal-case tracking-normal">
              Famille :
              <select
                value={family}
                onChange={e => { setFamily(e.target.value); setStrategy(STRATEGY_FAMILIES[e.target.value][0]); }}
                className="bg-[#1E1E1E] border border-border text-[#FFFFFF] py-1 px-2 text-[11px] focus:border-[#4A90E2] outline-none"
              >
                {Object.keys(STRATEGY_FAMILIES).map(f => <option key={f}>{f}</option>)}
              </select>
            </label>
            <label className="text-[10px] text-[#888888] flex items-center gap-2 normal-case tracking-normal">
              Stratégie :
              <select
                value={strategy}
                onChange={e => setStrategy(e.target.value)}
                className="bg-[#1E1E1E] border border-border text-[#FFFFFF] py-1 px-2 text-[11px] focus:border-[#4A90E2] outline-none"
              >
                {STRATEGY_FAMILIES[family].map(s => <option key={s}>{s}</option>)}
              </select>
            </label>
            <button className="bg-[#4A90E2] text-[#000000] px-3 py-1 text-[10px] hover:bg-[#357ABD] transition-colors font-semibold uppercase tracking-wider">
              Analyser
            </button>
          </div>
        </div>
      </div>

      {/* Legs table + Métriques */}
      <div className="flex gap-2">
        {/* Legs table */}
        <div className="flex-1 border border-border">
          <div className="bg-[#2D2D2D] px-3 py-1 text-[10px] uppercase tracking-wider text-[#FFFFFF] border-b border-border flex justify-between items-center">
            <span>Legs — {strategy}</span>
            <button className="text-[#4A90E2] hover:text-[#357ABD] text-[10px] normal-case tracking-normal">+ Ajouter un Leg</button>
          </div>
          <div className="bg-card overflow-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-[#2D2D2D] text-[10px] uppercase tracking-wider text-[#888888]">
                  <th className="border-b border-r border-border py-1.5 px-3 text-center">Position</th>
                  <th className="border-b border-r border-border py-1.5 px-3 text-center">Qté</th>
                  <th className="border-b border-r border-border py-1.5 px-3 text-center">Type</th>
                  <th className="border-b border-r border-border py-1.5 px-3 text-center">Strike</th>
                  <th className="border-b border-r border-border py-1.5 px-3 text-center">Échéance</th>
                  <th className="border-b border-r border-border py-1.5 px-3 text-center">Prime</th>
                  <th className="border-b border-border py-1.5 px-3 text-center">IV</th>
                </tr>
              </thead>
              <tbody>
                {STRATEGY_LEGS.map(leg => (
                  <tr
                    key={leg.id}
                    className={`border-b border-border hover:bg-[#2D2D2D] ${leg.position === 'Long' ? 'text-positive' : 'text-negative'}`}
                  >
                    <td className="border-r border-border py-2 px-3">{leg.position}</td>
                    <td className="border-r border-border py-2 px-3 text-center text-foreground">{leg.qty}</td>
                    <td className="border-r border-border py-2 px-3 text-center text-foreground">{leg.type}</td>
                    <td className="border-r border-border py-2 px-3 text-center font-semibold">{leg.strike}</td>
                    <td className="border-r border-border py-2 px-3 text-center text-foreground">{leg.maturity}</td>
                    <td className="border-r border-border py-2 px-3 text-center text-foreground">{leg.premium.toFixed(2)}</td>
                    <td className="py-2 px-3 text-center text-foreground">{leg.iv}</td>
                  </tr>
                ))}
                <tr className="bg-[#2D2D2D]">
                  <td colSpan={5} className="py-2 px-3 text-[#888888] font-semibold">Coût net de la stratégie</td>
                  <td className="py-2 px-3 text-center text-negative font-semibold">{netCost.toFixed(2)} $</td>
                  <td></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Métriques */}
        <div className="w-[280px] border border-border flex flex-col">
          <div className="bg-[#2D2D2D] px-3 py-1 text-[10px] uppercase tracking-wider text-[#FFFFFF] border-b border-border">
            Métriques
          </div>
          <div className="bg-[#000000] p-2 space-y-1 flex-1">
            <MetricRow label="Coût total :" value="-8.30 $" color="text-negative" />
            <MetricRow label="Breakeven :" value="283.30" />
            <MetricRow label="Gain maximum :" value="+11.70 $" color="text-positive" />
            <MetricRow label="Perte maximum :" value="-8.30 $" color="text-negative" />
            <div className="border-t border-border pt-2 mt-2">
              <div className="text-[10px] text-[#888888] uppercase tracking-wider mb-2">Grecs Agrégés</div>
              <div className="grid grid-cols-2 gap-y-1.5 gap-x-2 text-[11px]">
                <div className="flex justify-between"><span className="text-[#888888]">Δ</span><span className="text-positive">+0.3521</span></div>
                <div className="flex justify-between"><span className="text-[#888888]">Γ</span><span className="text-foreground">0.0045</span></div>
                <div className="flex justify-between"><span className="text-[#888888]">Θ</span><span className="text-negative">-3.21</span></div>
                <div className="flex justify-between"><span className="text-[#888888]">ν</span><span className="text-positive">+8.45</span></div>
                <div className="flex justify-between"><span className="text-[#888888]">ρ</span><span className="text-foreground">+4.12</span></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* P&L Profile chart */}
      <div className="flex-1 border border-border flex flex-col min-h-[280px]">
        <div className="bg-[#2D2D2D] px-3 py-1.5 text-[11px] uppercase tracking-wider text-[#FFFFFF] border-b border-border">
          Profil P&L — {strategy}
        </div>
        <div className="flex-1 bg-card p-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={PAYOFF_DATA} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
              <defs>
                <linearGradient id="stratPayoff" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#4A90E2" stopOpacity={0.1} />
                  <stop offset="100%" stopColor="#4A90E2" stopOpacity={0.01} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#333333" vertical={false} />
              <XAxis
                dataKey="spot"
                stroke="#333333"
                tick={{ fill: '#888888', fontSize: 10 }}
                tickMargin={8}
                domain={['dataMin', 'dataMax']}
                type="number"
                label={{ value: "Prix sous-jacent à l'échéance (S)", position: 'insideBottom', offset: -12, fill: '#888888', fontSize: 10 }}
              />
              <YAxis stroke="#333333" tick={{ fill: '#888888', fontSize: 10 }} tickMargin={8} />
              <ReferenceLine y={0} stroke="#333333" />
              <ReferenceLine x={275} stroke="#888888" strokeDasharray="3 3" label={{ value: 'K₁=275', fill: '#888888', fontSize: 9, position: 'top' }} />
              <ReferenceLine x={295} stroke="#888888" strokeDasharray="3 3" label={{ value: 'K₂=295', fill: '#888888', fontSize: 9, position: 'top' }} />
              <Area type="linear" dataKey="payoff" stroke="#4A90E2" strokeWidth={1.5} fill="url(#stratPayoff)" isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function MetricRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-0.5 border-b border-border py-1.5">
      <span className="text-[9px] text-[#888888] uppercase tracking-wider text-center">{label.replace(' :', '')}</span>
      <span className={`text-[12px] font-bold text-center ${color || 'text-[#FFFFFF]'}`}>{value}</span>
    </div>
  );
}
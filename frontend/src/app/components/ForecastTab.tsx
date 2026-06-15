import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

/* ── Static forecast data ── */
const DAYS = 120;
const FORECAST_START = 90;

function generateIVData() {
  const data = [];
  let iv = 0.22;
  for (let i = 0; i < DAYS; i++) {
    const isForecast = i >= FORECAST_START;
    if (i < FORECAST_START) {
      iv += (Math.random() - 0.5) * 0.008;
      iv = Math.max(0.12, Math.min(0.35, iv));
      data.push({ day: i - FORECAST_START, iv_hist: iv * 100, iv_forecast: null, iv_upper: null, iv_lower: null });
    } else {
      iv += (Math.random() - 0.48) * 0.006;
      iv = Math.max(0.12, Math.min(0.35, iv));
      data.push({
        day: i - FORECAST_START,
        iv_hist: i === FORECAST_START ? data[i - 1].iv_hist : null,
        iv_forecast: iv * 100,
        iv_upper: iv * 100 + 2.5 + (i - FORECAST_START) * 0.08,
        iv_lower: iv * 100 - 2.5 - (i - FORECAST_START) * 0.08,
      });
    }
  }
  return data;
}

function generatePriceData() {
  const data = [];
  let price = 4.55;
  for (let i = 0; i < DAYS; i++) {
    if (i < FORECAST_START) {
      price += (Math.random() - 0.5) * 0.3;
      price = Math.max(1, price);
      data.push({ day: i - FORECAST_START, price_hist: price, price_forecast: null });
    } else {
      price += (Math.random() - 0.47) * 0.25;
      price = Math.max(0.5, price);
      data.push({
        day: i - FORECAST_START,
        price_hist: i === FORECAST_START ? data[i - 1].price_hist : null,
        price_forecast: price,
      });
    }
  }
  return data;
}

function generateDeltaData() {
  const data = [];
  let delta = 0.35;
  for (let i = 0; i < DAYS; i++) {
    if (i < FORECAST_START) {
      delta += (Math.random() - 0.5) * 0.02;
      delta = Math.max(0, Math.min(1, delta));
      data.push({ day: i - FORECAST_START, delta_hist: delta, delta_forecast: null });
    } else {
      delta += (Math.random() - 0.48) * 0.015;
      delta = Math.max(0, Math.min(1, delta));
      data.push({
        day: i - FORECAST_START,
        delta_hist: i === FORECAST_START ? data[i - 1].delta_hist : null,
        delta_forecast: delta,
      });
    }
  }
  return data;
}

const ivData = generateIVData();
const priceData = generatePriceData();
const deltaData = generateDeltaData();

export function ForecastTab() {
  return (
    <div className="flex flex-col h-full gap-2 p-2 overflow-auto">
      {/* Info bar */}
      <div className="border border-border">
        <div className="bg-panel-header px-3 py-1.5 text-[11px] uppercase tracking-wider text-[#FFFFFF] border-b border-border flex justify-between items-center">
          <span>Forecast IV — Google TimesFM (2.5-200M)</span>
          <div className="flex gap-3 items-center">
            <label className="text-[10px] text-[#888888] flex items-center gap-2 normal-case tracking-normal">
              Ticker :
              <input type="text" defaultValue="AAPL" className="bg-[#1E1E1E] border border-border text-[#FFFFFF] py-1 px-2 w-20 text-[11px] focus:border-[#4A90E2] outline-none" />
            </label>
            <label className="text-[10px] text-[#888888] flex items-center gap-2 normal-case tracking-normal">
              Strike :
              <input type="text" defaultValue="290" className="bg-[#1E1E1E] border border-border text-[#FFFFFF] py-1 px-2 w-16 text-[11px] focus:border-[#4A90E2] outline-none" />
            </label>
            <label className="text-[10px] text-[#888888] flex items-center gap-2 normal-case tracking-normal">
              Horizon :
              <select className="bg-[#1E1E1E] border border-border text-[#FFFFFF] py-1 px-2 text-[11px] focus:border-[#4A90E2] outline-none">
                <option>30 jours</option>
                <option>63 jours</option>
              </select>
            </label>
            <button className="bg-[#4A90E2] text-[#000000] px-3 py-1 text-[10px] hover:bg-[#357ABD] transition-colors font-semibold uppercase tracking-wider">
              Lancer Forecast
            </button>
          </div>
        </div>
      </div>

      {/* Model info */}
      <div className="border border-border">
        <div className="bg-card p-2.5 flex gap-6 text-[10px]">
          <div className="flex gap-1.5">
            <span className="text-[#888888]">Modèle :</span>
            <span className="text-[#FFFFFF]">Google TimesFM 2.5-200M</span>
          </div>
          <div className="flex gap-1.5">
            <span className="text-[#888888]">Source IV :</span>
            <span className="text-[#FFFFFF]">Inversion BSM (Brent) via marketdata.app</span>
          </div>
          <div className="flex gap-1.5">
            <span className="text-[#888888]">Exécution :</span>
            <span className="text-[#00FF00]">CPU · QThread async</span>
          </div>
          <div className="flex gap-1.5">
            <span className="text-[#888888]">Repricing :</span>
            <span className="text-[#FFFFFF]">BSM jour par jour, IV prédite injectée</span>
          </div>
        </div>
      </div>

      {/* Chart 1: IV historique + forecast */}
      <div className="flex-1 border border-border flex flex-col min-h-[200px]">
        <div className="bg-[#2D2D2D] px-3 py-1 text-[10px] text-[#FFFFFF] border-b border-border flex justify-between items-center">
          <span className="uppercase tracking-wider text-[11px]">Volatilité Implicite — Historique + Forecast</span>
          <div className="flex gap-3 text-[#888888] normal-case tracking-normal">
            <div className="flex items-center gap-1"><div className="w-2.5 h-[2px] bg-[#D0D0D0]" /><span>IV historique</span></div>
            <div className="flex items-center gap-1"><div className="w-2.5 h-[2px] bg-[#4A90E2]" /><span>IV forecast</span></div>
            <div className="flex items-center gap-1"><div className="w-2.5 h-[2px] bg-[#4A90E2] opacity-30" /><span>Intervalle confiance</span></div>
          </div>
        </div>
        <div className="flex-1 bg-card p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={ivData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid stroke="#333333" vertical={false} />
              <XAxis dataKey="day" stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }} label={{ value: 'Jours (0 = début forecast)', position: 'insideBottom', offset: -2, fill: '#888888', fontSize: 9 }} />
              <YAxis stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }} domain={['auto', 'auto']} tickFormatter={(v: number) => v.toFixed(0) + '%'} />
              <ReferenceLine x={0} stroke="#FFCC00" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="iv_upper" stroke="#4A90E2" strokeOpacity={0.2} strokeWidth={1} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="iv_lower" stroke="#4A90E2" strokeOpacity={0.2} strokeWidth={1} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="iv_hist" stroke="#D0D0D0" strokeWidth={1.5} dot={false} isAnimationActive={false} connectNulls={false} />
              <Line type="monotone" dataKey="iv_forecast" stroke="#4A90E2" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: Prix option */}
      <div className="flex-1 border border-border flex flex-col min-h-[180px]">
        <div className="bg-[#2D2D2D] px-3 py-1 text-[10px] uppercase tracking-wider text-[#FFFFFF] border-b border-border">
          Prix de l'Option (BSM à IV prédite)
        </div>
        <div className="flex-1 bg-card p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={priceData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid stroke="#333333" vertical={false} />
              <XAxis dataKey="day" stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }} />
              <YAxis stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }} domain={['auto', 'auto']} />
              <ReferenceLine x={0} stroke="#FFCC00" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="price_hist" stroke="#D0D0D0" strokeWidth={1.5} dot={false} isAnimationActive={false} connectNulls={false} />
              <Line type="monotone" dataKey="price_forecast" stroke="#4A90E2" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 3: Delta */}
      <div className="flex-1 border border-border flex flex-col min-h-[180px]">
        <div className="bg-[#2D2D2D] px-3 py-1 text-[10px] uppercase tracking-wider text-[#FFFFFF] border-b border-border">
          Delta (recalculé jour par jour)
        </div>
        <div className="flex-1 bg-card p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={deltaData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid stroke="#333333" vertical={false} />
              <XAxis dataKey="day" stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }} />
              <YAxis stroke="#333333" tick={{ fill: '#888888', fontSize: 9 }} domain={[0, 1]} />
              <ReferenceLine x={0} stroke="#FFCC00" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="delta_hist" stroke="#D0D0D0" strokeWidth={1.5} dot={false} isAnimationActive={false} connectNulls={false} />
              <Line type="monotone" dataKey="delta_forecast" stroke="#4A90E2" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

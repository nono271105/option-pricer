import React, { useEffect, useState } from 'react';
import Plot from './Plot';
import { useMarket } from '../App';
import { calculateSmile, getAvailableExpiries } from '../../lib/api';
import { formatDateFR } from '../../lib/dates';

interface SmilePoint { strike: number; iv: number; type: string; }
interface SmileState {
  expiry_used: string | null;
  strikes_interp: number[];
  ivs_interp: number[];
  raw_data: SmilePoint[];
  current_price: number | null;
  loading: boolean;
  error: string | null;
}

interface ChartPoint { strike: number; iv: number | null; raw?: number; }

export function SmileTab() {
  const market = useMarket();
  const [ticker, setTicker] = useState(market.ticker);
  const [expiries, setExpiries] = useState<string[]>([]);
  const [selectedExpiry, setSelectedExpiry] = useState('');
  const [expiriesLoading, setExpiriesLoading] = useState(false);
  const [expiryError, setExpiryError] = useState<string | null>(null);

  const [state, setState] = useState<SmileState>({
    expiry_used: null, strikes_interp: [], ivs_interp: [], raw_data: [],
    current_price: null, loading: false, error: null,
  });

  useEffect(() => {
    setTicker(market.ticker);
  }, [market.ticker]);

  useEffect(() => {
    const normalizedTicker = ticker.trim().toUpperCase();
    setExpiries([]);
    setSelectedExpiry('');
    setExpiryError(null);

    if (!normalizedTicker) {
      setExpiriesLoading(false);
      return;
    }

    let cancelled = false;
    const timeout = setTimeout(async () => {
      setExpiriesLoading(true);
      try {
        const res = await getAvailableExpiries(normalizedTicker);
        if (cancelled) return;
        if (res.error) {
          setExpiryError(res.error);
        } else if (res.expiries.length === 0) {
          setExpiryError(`Aucune échéance disponible pour ${normalizedTicker}.`);
        } else {
          setExpiries(res.expiries);
          setSelectedExpiry(res.expiries[0]);
        }
      } catch (e: unknown) {
        if (!cancelled) setExpiryError(String(e));
      } finally {
        if (!cancelled) setExpiriesLoading(false);
      }
    }, 300);

    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [ticker]);

  const handleCalculate = async () => {
    if (!selectedExpiry) return;
    const selectedTicker = ticker.trim().toUpperCase();
    setState(s => ({
      ...s, loading: true, error: null, expiry_used: null,
      strikes_interp: [], ivs_interp: [], raw_data: [], current_price: null,
    }));

    try {
      const quote = await market.ensureMarket(selectedTicker);
      if (!quote || quote.S === null) {
        setState(s => ({ ...s, loading: false, error: `Données de marché indisponibles pour ${selectedTicker}.` }));
        return;
      }
      const res = await calculateSmile(selectedTicker, selectedExpiry, quote.S, quote.r, quote.q);
      if (res.error) {
        setState(s => ({ ...s, loading: false, error: res.error }));
        return;
      }
      setState(s => ({
        ...s, loading: false, error: null,
        expiry_used: res.expiry_used,
        strikes_interp: res.strikes_interp,
        ivs_interp: res.ivs_interp,
        raw_data: res.raw_data,
        current_price: res.current_price,
      }));
    } catch (e: any) {
      setState(s => ({ ...s, loading: false, error: String(e) }));
    }
  };

  // Combinaison données interpolées + raw pour le chart
  const chartData: ChartPoint[] = state.strikes_interp.map((strike, i) => ({
    strike,
    iv: state.ivs_interp[i] ?? null,
  }));

  // Points raw (bruts)
  const rawCallData = state.raw_data.filter(d => d.type !== 'put')
    .map(d => ({ strike: d.strike, raw: d.iv }));
  const rawPutData = state.raw_data.filter(d => d.type === 'put')
    .map(d => ({ strike: d.strike, raw: d.iv }));

  // Tableau des données brutes
  const allRaw = [...state.raw_data].sort((a, b) => a.strike - b.strike);

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">
      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] rounded">Warning {state.error}</div>
      )}

      {/* Contrôles */}
      <div className="border border-[#222222]">
        <div className="flex flex-wrap items-center px-2 py-1 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] gap-3">
          <span className="font-bold text-white"> SMILE DE VOLATILITÉ IMPLICITE</span>
          <div className="flex flex-wrap gap-3 items-center">
            <label className="text-[#888888] flex items-center gap-1">
              Ticker :
              <input value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())} disabled={state.loading}
              className="bg-[#121212] border border-[#333333] text-white py-0.5 px-1 text-[11px] w-[80px] outline-none" />
            </label>
            <label className="text-[#888888] flex items-center gap-1">
              Échéance :
              <select value={selectedExpiry} onChange={e => setSelectedExpiry(e.target.value)}
                disabled={state.loading || expiriesLoading || expiries.length === 0}
                className="bg-[#121212] border border-[#333333] text-white py-0.5 px-1 text-[11px] outline-none disabled:opacity-50">
                {expiries.length === 0 && (
                  <option value="">{expiriesLoading ? 'Chargement...' : 'Aucune échéance'}</option>
                )}
                {expiries.map(expiry => (
                  <option key={expiry} value={expiry}>{formatDateFR(expiry)}</option>
                ))}
              </select>
            </label>
            <button id="smile-calc-btn" onClick={handleCalculate}
              disabled={state.loading || expiriesLoading || !selectedExpiry}
              className="bg-[#4A90E2] text-white px-3 py-0.5 hover:bg-[#357ABD] text-[10px] font-bold rounded-sm disabled:opacity-50">
              {state.loading ? 'Calcul...' : 'Calculer le smile'}
            </button>
          </div>
          {state.expiry_used && (
            <span className="text-[#FFCC00] text-[10px]">Échéance utilisée : {formatDateFR(state.expiry_used)}</span>
          )}
        </div>
        {expiryError && (
          <div className="bg-[#3D0000] border-t border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px]">
            {expiryError}
          </div>
        )}
      </div>

      {/* Graphique smile */}
      <div className="flex-1 border border-[#222222] flex flex-col min-h-[300px]">
        <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] justify-between">
          <span className="font-bold text-white"> COURBE SMILE IV (Volatilité Implicite vs Strike)</span>
          <div className="flex gap-4 text-[9px] text-[#888888]">
            <span className="flex items-center gap-1"><span className="inline-block w-6 border-t border-[#4A90E2]"></span>IV Interpolée</span>
            <span className="flex items-center gap-1"><span className="text-[#00FF00]">*</span>Calls (brut)</span>
            <span className="flex items-center gap-1"><span className="text-[#FF4444]">*</span>Puts (brut)</span>
          </div>
        </div>
        <div className="bg-[#0A0A0A] flex-1 p-3">
          {chartData.length === 0 ? (
            <div className="flex items-center justify-center h-full text-[#888888] text-[12px]">
              Choisissez un ticker et une échéance puis calculez le smile
            </div>
          ) : (
            <Plot
              data={[
                {
                  x: chartData.map(d => d.strike),
                  y: chartData.map(d => d.iv),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  line: { color: '#4A90E2', width: 2, shape: 'linear' },
                  name: 'IV Interpolée'
                },
                {
                  x: rawCallData.map(d => d.strike),
                  y: rawCallData.map(d => d.raw),
                  type: 'scatter' as const,
                  mode: 'markers' as const,
                  marker: { color: '#00FF00', size: 6 },
                  name: 'Calls (brut)'
                },
                {
                  x: rawPutData.map(d => d.strike),
                  y: rawPutData.map(d => d.raw),
                  type: 'scatter' as const,
                  mode: 'markers' as const,
                  marker: { color: '#FF4444', size: 6 },
                  name: 'Puts (brut)'
                }
              ]}
              layout={{
                autosize: true,
                margin: { l: 40, r: 20, t: 10, b: 30 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                xaxis: { title: { text: 'Strike (K)', font: { size: 10, color: '#888' } }, gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 } },
                yaxis: { title: { text: 'IV (%)', font: { size: 10, color: '#888' } }, gridcolor: '#1A1A1A', tickfont: { color: '#888', size: 9 }, ticksuffix: '%' },
                hovermode: 'x unified',
                shapes: [
                  ...(state.current_price ? [{
                    type: 'line' as const, xref: 'x' as const, yref: 'paper' as const, x0: state.current_price, x1: state.current_price,
                    y0: 0, y1: 1,
                    line: { color: '#FF4444', dash: 'dash' as const, width: 1 }
                  }] : [])
                ],
                annotations: [
                  ...(state.current_price ? [{
                    x: state.current_price, y: 1, xref: 'x' as const, yref: 'paper' as const,
                    text: `S=${state.current_price.toFixed(2)}`, showarrow: false, yanchor: 'bottom', font: { color: '#FF4444', size: 9 }
                  }] : [])
                ],
                showlegend: false
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler={true}
            />
          )}
        </div>
      </div>

      {/* Tableau des données brutes */}
      {allRaw.length > 0 && (
        <div className="border border-[#222222]">
          <div className="flex items-center px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
            <span className="font-bold text-white"> DONNÉES BRUTES  IV PAR STRIKE</span>
            <span className="ml-2 text-[#888888]">({allRaw.length} points)</span>
          </div>
          <div className="bg-[#000000] overflow-auto max-h-[180px]">
            <table className="w-full text-right border-collapse text-[10px]">
              <thead>
                <tr className="bg-[#111111] text-[9px] text-[#888888] divide-x divide-[#222222] border-b border-[#222222] sticky top-0">
                  <th className="py-1 px-2 font-normal text-left">Type</th>
                  <th className="py-1 px-2 font-normal">Strike</th>
                  <th className="py-1 px-2 font-normal">IV (%)</th>
                </tr>
              </thead>
              <tbody>
                {allRaw.map((d, i) => (
                  <tr key={i} className="divide-x divide-[#111111] border-b border-[#111111] hover:bg-[#111111]">
                    <td className={`py-0.5 px-2 text-left font-bold ${d.type === 'put' ? 'text-[#FF4444]' : 'text-[#00FF00]'}`}>
                      {d.type}
                    </td>
                    <td className="py-0.5 px-2 text-[#D4D4D4]">{d.strike.toFixed(2)}</td>
                    <td className="py-0.5 px-2 text-[#FFCC00]">{d.iv.toFixed(2)}%</td>
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

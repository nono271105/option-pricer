import React, { useState, useRef, useEffect } from 'react';
import { useMarket } from '../App';

interface OptionRow {
  strike: number;
  bid: number;
  ask: number;
  iv: number;
  volume: number;
  oi: number;
  delta: number | null;
}

interface ChainState {
  expiry_used: string | null;
  calls: OptionRow[];
  puts: OptionRow[];
  expiries: string[];
  loading: boolean;
  error: string | null;
}

export function OptionChainTab() {
  const market = useMarket();
  const tickerRef = useRef<HTMLInputElement>(null);
  const [selectedExpiry, setSelectedExpiry] = useState<string>('');
  const [selectedStrike, setSelectedStrike] = useState<number | null>(null);

  const [state, setState] = useState<ChainState>({
    expiry_used: null, calls: [], puts: [],
    expiries: [], loading: false, error: null,
  });

  // Charge les expiries disponibles
  const fetchExpiries = async (ticker: string) => {
    if (!window.eel) return;
    try {
      const res = await window.eel.get_available_expiries(ticker)();
      if (!res.error && res.expiries.length > 0) {
        setState(s => ({ ...s, expiries: res.expiries }));
        setSelectedExpiry(res.expiries[0]);
      }
    } catch { /* silencieux */ }
  };

  const handleFetch = async () => {
    const ticker = tickerRef.current?.value.trim().toUpperCase() || market.ticker;
    const expiry = selectedExpiry || getDefaultExpiry();
    setState(s => ({ ...s, loading: true, error: null }));

    if (!window.eel) {
      setState(s => ({ ...s, loading: false, error: 'Eel non disponible' }));
      return;
    }

    try {
      // Récupérer marché si pas déjà fait
      if (!market.S) await market.fetchMarket(ticker);

      const res = await window.eel.get_option_chain(ticker, expiry)();
      if (res.error) {
        setState(s => ({ ...s, loading: false, error: res.error }));
        return;
      }
      setState(s => ({
        ...s, loading: false, error: null,
        expiry_used: res.expiry_used,
        calls: res.calls,
        puts: res.puts,
      }));
      if (res.calls.length > 0) {
        // Sélectionner le strike ATM
        const S = market.S ?? 0;
        const atm = res.calls.reduce((prev: OptionRow, curr: OptionRow) =>
          Math.abs(curr.strike - S) < Math.abs(prev.strike - S) ? curr : prev
        );
        setSelectedStrike(atm.strike);
      }
    } catch (e: any) {
      setState(s => ({ ...s, loading: false, error: String(e) }));
    }
  };

  // Charge les expiries au montage
  useEffect(() => {
    if (window.eel && market.ticker) {
      fetchExpiries(market.ticker);
    }
  }, [market.ticker]);

  // Combine calls et puts par strike
  const allStrikes = [...new Set([
    ...state.calls.map(c => c.strike),
    ...state.puts.map(p => p.strike),
  ])].sort((a, b) => a - b);

  const callMap = new Map(state.calls.map(c => [c.strike, c]));
  const putMap = new Map(state.puts.map(p => [p.strike, p]));
  const spotPrice = market.S;

  return (
    <div className="flex flex-col h-full bg-[#000000]">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-2 py-1 bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#333333] text-[10px] shrink-0">
        <div className="flex items-center gap-3">
          <span className="font-bold text-white">▼ OPTION CHAIN</span>
          {state.expiry_used && (
            <span className="text-[#FFCC00]">Échéance : {state.expiry_used}</span>
          )}
          {spotPrice && (
            <span className="text-[#888888]">Spot : <span className="text-[#00FF00]">{spotPrice.toFixed(2)}</span></span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <input ref={tickerRef} defaultValue={market.ticker}
            className="bg-[#FFF4C2] text-black px-1.5 py-0.5 font-bold w-[80px] text-[11px] outline-none rounded-sm"
            placeholder="AAPL" />
          {state.expiries.length > 0 ? (
            <select
              value={selectedExpiry}
              onChange={e => setSelectedExpiry(e.target.value)}
              className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[10px] outline-none appearance-none"
            >
              {state.expiries.map(e => (
                <option key={e} value={e}>{e}</option>
              ))}
            </select>
          ) : (
            <input
              value={selectedExpiry}
              onChange={e => setSelectedExpiry(e.target.value)}
              placeholder={getDefaultExpiry()}
              className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[10px] outline-none w-[110px]"
            />
          )}
          <button id="chain-fetch-btn" onClick={handleFetch} disabled={state.loading}
            className="bg-[#4A90E2] text-white px-3 py-0.5 hover:bg-[#357ABD] text-[10px] font-bold rounded-sm disabled:opacity-50">
            {state.loading ? '⏳ Chargement...' : '↻ Charger Chain'}
          </button>
        </div>
      </div>

      {/* Erreur */}
      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] mx-2 mt-1 rounded shrink-0">
          ⚠ {state.error}
        </div>
      )}

      {/* État vide */}
      {state.calls.length === 0 && !state.loading && !state.error && (
        <div className="flex-1 flex items-center justify-center text-[#888888] text-[12px]">
          Entrez un ticker et cliquez sur "Charger Chain"
        </div>
      )}

      {/* Grid Header */}
      {state.calls.length > 0 && (
        <>
          <div className="grid grid-cols-[1fr_60px_1fr] bg-[#111111] border-b border-[#333333] shrink-0 text-[9px]">
            <div className="flex flex-col">
              <div className="text-center font-bold text-white py-1 border-b border-[#333333]">CALLS</div>
              <div className="flex text-[#888888] uppercase tracking-wider divide-x divide-[#333333]">
                <div className="flex-1 text-right px-1 py-0.5">BID</div>
                <div className="flex-1 text-right px-1 py-0.5">ASK</div>
                <div className="flex-1 text-right px-1 py-0.5">IV%</div>
                <div className="flex-1 text-right px-1 py-0.5">VOL</div>
                <div className="flex-1 text-right px-1 py-0.5">OI</div>
                <div className="flex-1 text-right px-1 py-0.5">Δ</div>
              </div>
            </div>
            <div className="flex items-end justify-center text-[#888888] pb-0.5 border-x border-[#333333]">STRIKE</div>
            <div className="flex flex-col">
              <div className="text-center font-bold text-white py-1 border-b border-[#333333]">PUTS</div>
              <div className="flex text-[#888888] uppercase tracking-wider divide-x divide-[#333333]">
                <div className="flex-1 text-right px-1 py-0.5">BID</div>
                <div className="flex-1 text-right px-1 py-0.5">ASK</div>
                <div className="flex-1 text-right px-1 py-0.5">IV%</div>
                <div className="flex-1 text-right px-1 py-0.5">VOL</div>
                <div className="flex-1 text-right px-1 py-0.5">OI</div>
                <div className="flex-1 text-right px-1 py-0.5">Δ</div>
              </div>
            </div>
          </div>

          {/* Grid Body */}
          <div className="flex-1 overflow-auto">
            {allStrikes.map(strike => {
              const call = callMap.get(strike);
              const put = putMap.get(strike);
              const isSelected = strike === selectedStrike;
              const isAtm = spotPrice ? Math.abs(strike - spotPrice) < (allStrikes[1] - allStrikes[0]) / 2 : false;
              const callItm = spotPrice ? strike < spotPrice : false;
              const putItm = spotPrice ? strike > spotPrice : false;

              const callBg = isSelected ? 'bg-[#1A2640]' : callItm ? 'bg-[#111820]' : '';
              const putBg  = isSelected ? 'bg-[#1A2640]' : putItm  ? 'bg-[#111820]' : '';
              const strikeBg = isAtm ? 'bg-[#4A90E2] text-white font-bold' :
                               isSelected ? 'bg-[#2A3D5C] text-[#A0C0E0] font-bold' : 'bg-[#1A1A1A] text-[#A0C0E0]';

              return (
                <div
                  key={strike}
                  className="grid grid-cols-[1fr_60px_1fr] border-b border-[#1A1A1A] hover:bg-[#0D1A2E] cursor-pointer transition-colors"
                  onClick={() => setSelectedStrike(strike)}
                >
                  {/* Call Row */}
                  <div className={`flex text-[11px] divide-x divide-[#1A1A1A] ${callBg}`}>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#00CC66] font-medium">{call ? call.bid.toFixed(2) : '—'}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#FF4444] font-medium">{call ? call.ask.toFixed(2) : '—'}</div>
                    <div className={`flex-1 text-right px-1 py-0.5 ${callItm ? 'text-[#FFCC00]' : 'text-[#888888]'}`}>{call ? call.iv.toFixed(2) : '—'}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#888888]">{call ? (call.volume >= 1000 ? (call.volume / 1000).toFixed(0) + 'k' : call.volume) : '—'}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#666666]">{call ? (call.oi >= 1000 ? (call.oi / 1000).toFixed(0) + 'k' : call.oi) : '—'}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#D4D4D4]">{call?.delta !== null && call?.delta !== undefined ? call.delta.toFixed(3) : '—'}</div>
                  </div>

                  {/* Strike */}
                  <div className={`flex items-center justify-center text-[11px] ${strikeBg}`}>
                    {strike.toFixed(strike % 1 === 0 ? 0 : 2)}
                  </div>

                  {/* Put Row */}
                  <div className={`flex text-[11px] divide-x divide-[#1A1A1A] ${putBg}`}>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#00CC66] font-medium">{put ? put.bid.toFixed(2) : '—'}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#FF4444] font-medium">{put ? put.ask.toFixed(2) : '—'}</div>
                    <div className={`flex-1 text-right px-1 py-0.5 ${putItm ? 'text-[#FFCC00]' : 'text-[#888888]'}`}>{put ? put.iv.toFixed(2) : '—'}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#888888]">{put ? (put.volume >= 1000 ? (put.volume / 1000).toFixed(0) + 'k' : put.volume) : '—'}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#666666]">{put ? (put.oi >= 1000 ? (put.oi / 1000).toFixed(0) + 'k' : put.oi) : '—'}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#D4D4D4]">{put?.delta !== null && put?.delta !== undefined ? put.delta.toFixed(3) : '—'}</div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Status bar */}
          <div className="shrink-0 flex items-center justify-between px-2 py-0.5 bg-[#111111] border-t border-[#333333] text-[9px] text-[#888888]">
            <span>{state.calls.length} calls · {state.puts.length} puts · {allStrikes.length} strikes</span>
            {selectedStrike && (
              <span>Strike sélectionné : <span className="text-[#4A90E2] font-bold">{selectedStrike}</span></span>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function getDefaultExpiry(): string {
  const d = new Date();
  d.setDate(d.getDate() + 60);
  return d.toISOString().split('T')[0];
}

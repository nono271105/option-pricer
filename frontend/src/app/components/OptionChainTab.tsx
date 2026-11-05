import React, { useState, useEffect } from 'react';
import { useMarket } from '../App';
import { getAvailableExpiries, getOptionChain } from '../../lib/api';
import type { OptionRow } from '../../types/api';
import { formatDateFR } from '../../lib/dates';

interface ChainState {
  expiry_used: string | null;
  calls: OptionRow[];
  puts: OptionRow[];
  expiries: string[];
  expiriesLoading: boolean;
  expiryError: string | null;
  loading: boolean;
  error: string | null;
}

export function OptionChainTab() {
  const market = useMarket();
  const [ticker, setTicker] = useState(market.ticker);
  const [selectedExpiry, setSelectedExpiry] = useState<string>('');
  const [selectedStrike, setSelectedStrike] = useState<number | null>(null);

  const [state, setState] = useState<ChainState>({
    expiry_used: null, calls: [], puts: [],
    expiries: [], expiriesLoading: false, expiryError: null,
    loading: false, error: null,
  });

  useEffect(() => {
    setTicker(market.ticker);
  }, [market.ticker]);

  useEffect(() => {
    const normalizedTicker = ticker.trim().toUpperCase();
    setSelectedExpiry('');
    setSelectedStrike(null);
    setState(s => ({
      ...s, expiry_used: null, calls: [], puts: [], expiries: [],
      expiriesLoading: Boolean(normalizedTicker), expiryError: null,
      loading: false, error: null,
    }));

    if (!normalizedTicker) return;

    let cancelled = false;
    const timeout = setTimeout(async () => {
      try {
        const res = await getAvailableExpiries(normalizedTicker);
        if (cancelled) return;
        if (res.error) {
          setState(s => ({ ...s, expiryError: res.error, expiries: [] }));
        } else if (res.expiries.length === 0) {
          setState(s => ({
            ...s, expiryError: `Aucune échéance d'options disponible pour ${normalizedTicker}.`,
            expiries: [],
          }));
        } else {
          setState(s => ({ ...s, expiryError: null, expiries: res.expiries }));
          setSelectedExpiry(res.expiries[0]);
        }
      } catch (e: unknown) {
        if (!cancelled) {
          setState(s => ({ ...s, expiryError: String(e), expiries: [] }));
        }
      } finally {
        if (!cancelled) setState(s => ({ ...s, expiriesLoading: false }));
      }
    }, 300);

    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [ticker]);

  const handleFetch = async () => {
    const selectedTicker = ticker.trim().toUpperCase();
    if (!selectedExpiry) {
      setState(s => ({
        ...s,
        error: s.expiryError || 'Sélectionnez une échéance disponible avant de charger la chaîne.',
      }));
      return;
    }
    const expiry = selectedExpiry;
    setState(s => ({ ...s, loading: true, error: null }));

    try {
      const quote = await market.ensureMarket(selectedTicker);
      if (!quote || quote.S === null) {
        setState(s => ({ ...s, loading: false, error: `Données de marché indisponibles pour ${selectedTicker}.` }));
        return;
      }

      const res = await getOptionChain(selectedTicker, expiry, quote.S, quote.r, quote.q);
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
        const S = quote.S;
        const atm = res.calls.reduce((prev: OptionRow, curr: OptionRow) =>
          Math.abs(curr.strike - S) < Math.abs(prev.strike - S) ? curr : prev
        );
        setSelectedStrike(atm.strike);
      }
    } catch (e: any) {
      setState(s => ({ ...s, loading: false, error: String(e) }));
    }
  };

  // Combine calls et puts par strike
  const allStrikes = [...new Set([
    ...state.calls.map(c => c.strike),
    ...state.puts.map(p => p.strike),
  ])].sort((a, b) => a - b);

  const callMap = new Map(state.calls.map(c => [c.strike, c]));
  const putMap = new Map(state.puts.map(p => [p.strike, p]));
  const spotPrice = ticker.trim().toUpperCase() === market.ticker ? market.S : null;

  return (
    <div className="flex flex-col h-full bg-[#000000]">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-2 py-1 bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#333333] text-[10px] shrink-0">
        <div className="flex items-center gap-3">
          <span className="font-bold text-white"> CHAÎNE D'OPTIONS</span>
          {state.expiry_used && (
            <span className="text-[#FFCC00]">Échéance : {formatDateFR(state.expiry_used)}</span>
          )}
          {spotPrice && (
            <span className="text-[#888888]">Spot : <span className="text-[#00FF00]">{spotPrice.toFixed(2)}</span></span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <input value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())}
            className="bg-[#FFF4C2] text-black px-1.5 py-0.5 font-bold w-[80px] text-[11px] outline-none rounded-sm"
            placeholder="AAPL" />
          <select
            value={selectedExpiry}
            onChange={e => setSelectedExpiry(e.target.value)}
            disabled={state.expiriesLoading || state.expiries.length === 0}
            className="bg-[#121212] border border-[#333333] text-white px-1.5 py-0.5 text-[10px] outline-none appearance-none disabled:opacity-50"
          >
            {state.expiries.length === 0 ? (
              <option value="">
                {state.expiriesLoading ? 'Chargement des échéances...' : 'Aucune échéance'}
              </option>
            ) : state.expiries.map(e => (
              <option key={e} value={e}>{formatDateFR(e)}</option>
            ))}
          </select>
          <button id="chain-fetch-btn" onClick={handleFetch}
            disabled={state.loading || state.expiriesLoading || !selectedExpiry}
            className="bg-[#4A90E2] text-white px-3 py-0.5 hover:bg-[#357ABD] text-[10px] font-bold rounded-sm disabled:opacity-50">
            {state.loading ? 'Chargement...' : 'Charger la chaîne'}
          </button>
        </div>
      </div>

      {state.expiryError && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] mx-2 mt-1 rounded shrink-0">
          {state.expiryError}
        </div>
      )}

      {/* Erreur */}
      {state.error && (
        <div className="bg-[#3D0000] border border-[#FF4444] text-[#FF9999] px-3 py-1.5 text-[11px] mx-2 mt-1 rounded shrink-0">
          Warning {state.error}
        </div>
      )}

      {/* État vide */}
      {state.calls.length === 0 && !state.loading && !state.error && !state.expiryError && (
        <div className="flex-1 flex items-center justify-center text-[#888888] text-[12px]">
          {state.expiriesLoading
            ? 'Chargement des échéances disponibles...'
            : state.expiry_used
            ? `Aucune cotation d'option disponible pour l'échéance du ${formatDateFR(state.expiry_used)}.`
            : 'Choisissez un ticker et une échéance puis cliquez sur « Charger la chaîne »'}
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
                    <div className="flex-1 text-right px-1 py-0.5 text-[#00CC66] font-medium">{call ? call.bid.toFixed(2) : ''}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#FF4444] font-medium">{call ? call.ask.toFixed(2) : ''}</div>
                    <div className={`flex-1 text-right px-1 py-0.5 ${callItm ? 'text-[#FFCC00]' : 'text-[#888888]'}`}>{call ? call.iv.toFixed(2) : ''}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#888888]">{call ? (call.volume >= 1000 ? (call.volume / 1000).toFixed(0) + 'k' : call.volume) : ''}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#666666]">{call ? (call.oi >= 1000 ? (call.oi / 1000).toFixed(0) + 'k' : call.oi) : ''}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#D4D4D4]">{call?.delta !== null && call?.delta !== undefined ? call.delta.toFixed(3) : ''}</div>
                  </div>

                  {/* Strike */}
                  <div className={`flex items-center justify-center text-[11px] ${strikeBg}`}>
                    {strike.toFixed(strike % 1 === 0 ? 0 : 2)}
                  </div>

                  {/* Put Row */}
                  <div className={`flex text-[11px] divide-x divide-[#1A1A1A] ${putBg}`}>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#00CC66] font-medium">{put ? put.bid.toFixed(2) : ''}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#FF4444] font-medium">{put ? put.ask.toFixed(2) : ''}</div>
                    <div className={`flex-1 text-right px-1 py-0.5 ${putItm ? 'text-[#FFCC00]' : 'text-[#888888]'}`}>{put ? put.iv.toFixed(2) : ''}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#888888]">{put ? (put.volume >= 1000 ? (put.volume / 1000).toFixed(0) + 'k' : put.volume) : ''}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#666666]">{put ? (put.oi >= 1000 ? (put.oi / 1000).toFixed(0) + 'k' : put.oi) : ''}</div>
                    <div className="flex-1 text-right px-1 py-0.5 text-[#D4D4D4]">{put?.delta !== null && put?.delta !== undefined ? put.delta.toFixed(3) : ''}</div>
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

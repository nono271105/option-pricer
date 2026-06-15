import React, { useState, useCallback, createContext, useContext } from 'react';
import { OptionChainTab } from './components/OptionChainTab';
import { BsmTab } from './components/BsmTab';
import { CrrTab } from './components/CrrTab';
import { SimulationTab } from './components/SimulationTab';
import { SmileTab } from './components/SmileTab';
import { SurfaceTab } from './components/SurfaceTab';
import { ExoticsTab } from './components/ExoticsTab';
import { StrategiesTab } from './components/StrategiesTab';
import { ForecastTab } from './components/ForecastTab';

// ── Contexte de marché partagé entre tous les onglets ─────────────────────

export interface MarketState {
  ticker: string;
  companyName: string;
  S: number | null;
  r: number;
  q: number;
  histVol: number;
  isLoading: boolean;
  error: string | null;
}

interface MarketContextValue extends MarketState {
  fetchMarket: (ticker: string) => Promise<void>;
  setTicker: (t: string) => void;
}

const DEFAULT_MARKET: MarketState = {
  ticker: 'AAPL',
  companyName: '',
  S: null,
  r: 0.05,
  q: 0.0,
  histVol: 0.20,
  isLoading: false,
  error: null,
};

export const MarketContext = createContext<MarketContextValue>({
  ...DEFAULT_MARKET,
  fetchMarket: async () => {},
  setTicker: () => {},
});

export const useMarket = () => useContext(MarketContext);

// ── Eel helper : appel avec fallback si eel n'est pas disponible ──────────

async function callEel<T>(fn: () => Promise<T>, fallback: T): Promise<T> {
  try {
    if (typeof window !== 'undefined' && window.eel) {
      return await fn();
    }
  } catch (e) {
    console.warn('[Eel] appel échoué:', e);
  }
  return fallback;
}

// ── Onglets ───────────────────────────────────────────────────────────────

const TABS = [
  { id: 'chains',     label: 'Option Chains' },
  { id: 'bsm',        label: 'Modèle BSM' },
  { id: 'crr',        label: 'Modèle CRR' },
  { id: 'simulation', label: 'Simulation' },
  { id: 'smile',      label: 'Smile IV' },
  { id: 'surface',    label: 'Surface 3D' },
  { id: 'exotics',    label: 'Exotiques' },
  { id: 'strategies', label: 'Stratégies' },
  { id: 'forecast',   label: 'Forecast' },
];

// ── Composant principal ───────────────────────────────────────────────────

export default function App() {
  const [activeTab, setActiveTab] = useState('bsm');
  const [market, setMarket] = useState<MarketState>(DEFAULT_MARKET);
  const [tickerInput, setTickerInput] = useState('AAPL');

  const fetchMarket = useCallback(async (ticker: string) => {
    if (!ticker.trim()) return;
    const tk = ticker.trim().toUpperCase();
    setMarket(m => ({ ...m, isLoading: true, error: null }));

    const result = await callEel(
      () => window.eel.fetch_market_data(tk)(),
      { ticker: tk, company_name: tk, S: null, r: 0.05, q: 0.0, hist_vol: 0.20, error: 'Eel non disponible' }
    );

    if (result.error) {
      setMarket(m => ({ ...m, isLoading: false, error: result.error }));
    } else {
      setMarket({
        ticker: result.ticker,
        companyName: result.company_name,
        S: result.S,
        r: result.r ?? 0.05,
        q: result.q ?? 0.0,
        histVol: result.hist_vol ?? 0.20,
        isLoading: false,
        error: null,
      });
      setTickerInput(result.ticker);
    }
  }, []);

  const handleFetch = () => fetchMarket(tickerInput);

  const ctxValue: MarketContextValue = {
    ...market,
    fetchMarket,
    setTicker: setTickerInput,
  };

  return (
    <MarketContext.Provider value={ctxValue}>
      <div className="flex flex-col h-screen w-full bg-background text-foreground overflow-hidden">

        {/* TWS Global Header: Quote Panel */}
        <div className="flex flex-col shrink-0 border-b border-[#333333] bg-[#000000]">

          {/* Top Header Row */}
          <div className="flex items-center justify-between px-2 py-0.5 text-[10px] bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#333333]">
            <div className="flex items-center gap-4">
              <span className="font-bold text-white">▼ Quote Panel</span>
            </div>
            <div className="flex items-center gap-2">
              {market.isLoading && (
                <span className="text-[#FFCC00] animate-pulse">⏳ Chargement...</span>
              )}
              {market.error && (
                <span className="text-[#FF4444] text-[9px]">{market.error}</span>
              )}
              <span className="bg-[#FF3333] text-white px-1 font-bold">ARMED</span>
              <span className="text-[#888888]">Help</span>
            </div>
          </div>

          {/* Financial Instrument Data */}
          <div className="flex items-center px-2 py-1 gap-6 text-[12px] whitespace-nowrap overflow-x-auto">

            {/* Ticker input + bouton fetch */}
            <div className="flex flex-col">
              <span className="text-[9px] text-[#888888] mb-0.5">Financial Instrument</span>
              <div className="flex gap-1">
                <input
                  id="global-ticker-input"
                  value={tickerInput}
                  onChange={e => setTickerInput(e.target.value.toUpperCase())}
                  onKeyDown={e => e.key === 'Enter' && handleFetch()}
                  className="bg-[#FFF4C2] text-black px-2 py-0.5 font-bold rounded-sm w-[100px] text-[13px] outline-none"
                  placeholder="AAPL"
                />
                <button
                  id="global-fetch-btn"
                  onClick={handleFetch}
                  disabled={market.isLoading}
                  className="bg-[#2A2A2A] border border-[#444444] text-white px-2 py-0.5 text-[9px] hover:bg-[#3A3A3A] disabled:opacity-50 transition-colors rounded-sm"
                >
                  {market.isLoading ? '⏳' : '↻'}
                </button>
              </div>
              {market.companyName && (
                <span className="text-[9px] text-[#FFCC00] mt-0.5 truncate max-w-[150px]">
                  {market.companyName}
                </span>
              )}
            </div>

            {/* Prix spot */}
            <div className="flex flex-col items-end">
              <span className="text-[9px] text-[#888888] mb-0.5">Spot (S)</span>
              <span className={`font-bold text-[14px] ${market.S ? 'text-[#00FF00]' : 'text-[#888888]'}`}>
                {market.S ? market.S.toFixed(2) : 'N/C'}
              </span>
            </div>

            {/* Taux SOFR */}
            <div className="flex flex-col items-end">
              <span className="text-[9px] text-[#888888] mb-0.5">SOFR (r)</span>
              <span className="text-[#D4D4D4]">
                {(market.r * 100).toFixed(2)}%
              </span>
            </div>

            {/* Dividende */}
            <div className="flex flex-col items-end">
              <span className="text-[9px] text-[#888888] mb-0.5">Dividende (q)</span>
              <span className="text-[#D4D4D4]">
                {(market.q * 100).toFixed(2)}%
              </span>
            </div>

            {/* Volatilité historique */}
            <div className="flex flex-col items-end">
              <span className="text-[9px] text-[#888888] mb-0.5">Vol. Hist.</span>
              <span className="text-[#D4D4D4]">
                {(market.histVol * 100).toFixed(2)}%
              </span>
            </div>

          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col overflow-hidden bg-background">

          {/* TWS Style Tabs */}
          <div className="flex px-1 pt-1 bg-gradient-to-b from-[#2A2A2A] to-[#000000] border-b border-[#333333] shrink-0 overflow-x-auto">
            {TABS.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  id={`tab-${tab.id}`}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-3 py-1 text-[11px] font-bold rounded-t-md border-t border-l border-r border-[#444444] transition-colors -mb-[1px]
                    ${isActive
                      ? 'bg-[#000000] text-[#FFFFFF] border-b-transparent z-10'
                      : 'bg-[#1E1E1E] text-[#888888] border-b-[#333333] hover:text-[#D4D4D4]'
                    }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-hidden">
            {activeTab === 'chains'     && <OptionChainTab />}
            {activeTab === 'bsm'        && <BsmTab />}
            {activeTab === 'crr'        && <CrrTab />}
            {activeTab === 'simulation' && <SimulationTab />}
            {activeTab === 'smile'      && <SmileTab />}
            {activeTab === 'surface'    && <SurfaceTab />}
            {activeTab === 'exotics'    && <ExoticsTab />}
            {activeTab === 'strategies' && <StrategiesTab />}
            {activeTab === 'forecast'   && <ForecastTab />}
          </div>
        </div>

      </div>
    </MarketContext.Provider>
  );
}
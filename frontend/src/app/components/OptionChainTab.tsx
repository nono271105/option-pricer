import React, { useState } from 'react';

const STRIKES = [139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150];

const CALL_DATA = STRIKES.map(strike => ({
  strike,
  bid: (150 - strike + 5).toFixed(2),
  ask: (150 - strike + 5.2).toFixed(2),
  bidSize: Math.floor(Math.random() * 500) + 10,
  askSize: Math.floor(Math.random() * 500) + 10,
  position: strike === 145 ? 1 : 0,
  impliedVol: '1.66%',
  delta: strike < 145 ? 0.8 : strike === 145 ? 0.5 : 0.2,
  gamma: 0.028,
  theta: -0.059,
}));

const PUT_DATA = STRIKES.map(strike => ({
  strike,
  bid: (strike - 150 + 5).toFixed(2),
  ask: (strike - 150 + 5.2).toFixed(2),
  bidSize: Math.floor(Math.random() * 500) + 10,
  askSize: Math.floor(Math.random() * 500) + 10,
  position: 0,
  impliedVol: '1.64%',
  delta: strike > 145 ? -0.8 : strike === 145 ? -0.5 : -0.2,
  gamma: 0.027,
  theta: -0.056,
}));

export function OptionChainTab() {
  const [selectedStrike, setSelectedStrike] = useState<number | null>(145);

  return (
    <div className="flex flex-col h-full bg-[#000000]">
      {/* Option Chain Toolbar */}
      <div className="flex items-center justify-between px-2 py-1 bg-[#1A1A1A] border-b border-border text-[#D4D4D4] text-[10px]">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <span className="text-[#FF8800] font-bold">JUL 16 '26</span>
            <span className="text-[#888888]">24 DAYS</span>
          </div>
          <div className="flex items-center gap-1 opacity-60">
            <span>AUG 20 '26</span>
            <span>59 DAYS</span>
          </div>
          <button className="bg-transparent text-[#4A90E2] hover:text-[#FFFFFF] ml-2">MORE ▼</button>
        </div>
        <div className="flex items-center gap-4 text-[#888888]">
          <span>TABBED VIEW ▼</span>
          <span>PUT/CALL ▼</span>
          <span>12 STRIKES ▼</span>
          <span>SMART ▼</span>
          <span>AAPL ▼</span>
          <span>100</span>
        </div>
      </div>

      {/* Grid Header */}
      <div className="grid grid-cols-[1fr_50px_1fr] bg-[#111111] border-b border-border">
        {/* Calls Header */}
        <div className="flex flex-col">
          <div className="text-center font-bold text-[#D4D4D4] py-1 border-b border-border">CALLS</div>
          <div className="flex text-[9px] text-[#888888] uppercase tracking-wider divide-x divide-border">
            <div className="flex-1 text-right px-1">BID SIZE</div>
            <div className="flex-1 text-right px-1">BID</div>
            <div className="flex-1 text-right px-1">ASK</div>
            <div className="flex-1 text-right px-1">ASK SIZE</div>
            <div className="flex-1 text-center px-1">POS</div>
            <div className="flex-1 text-right px-1">IMPLD VOL %</div>
            <div className="flex-1 text-right px-1">DELTA</div>
            <div className="flex-1 text-right px-1">GAMMA</div>
            <div className="flex-1 text-right px-1">THETA</div>
          </div>
        </div>

        {/* Strike Header */}
        <div className="flex items-end justify-center text-[9px] text-[#888888] pb-0.5 border-x border-border">
          STRIKE
        </div>

        {/* Puts Header */}
        <div className="flex flex-col">
          <div className="text-center font-bold text-[#D4D4D4] py-1 border-b border-border">PUTS</div>
          <div className="flex text-[9px] text-[#888888] uppercase tracking-wider divide-x divide-border">
            <div className="flex-1 text-right px-1">BID SIZE</div>
            <div className="flex-1 text-right px-1">BID</div>
            <div className="flex-1 text-right px-1">ASK</div>
            <div className="flex-1 text-right px-1">ASK SIZE</div>
            <div className="flex-1 text-center px-1">POS</div>
            <div className="flex-1 text-right px-1">IMPLD VOL %</div>
            <div className="flex-1 text-right px-1">DELTA</div>
            <div className="flex-1 text-right px-1">GAMMA</div>
            <div className="flex-1 text-right px-1">THETA</div>
          </div>
        </div>
      </div>

      {/* Grid Body */}
      <div className="flex-1 overflow-auto bg-[#000000]">
        {STRIKES.map((strike, i) => {
          const call = CALL_DATA[i];
          const put = PUT_DATA[i];
          const isSelected = strike === selectedStrike;
          
          // Determine moneyness shading
          // Current mock price is ~145
          const callItm = strike < 145;
          const putItm = strike > 145;

          const callBg = isSelected ? 'bg-tws-blue-bg/40' : callItm ? 'bg-[#182132]' : 'bg-transparent';
          const putBg = isSelected ? 'bg-tws-blue-bg/40' : putItm ? 'bg-[#182132]' : 'bg-transparent';
          const strikeBg = isSelected ? 'bg-tws-strike-bg text-white' : 'bg-[#1E2532] text-[#A0C0E0]';

          return (
            <div 
              key={strike} 
              className="grid grid-cols-[1fr_50px_1fr] border-b border-[#222222] hover:bg-[#1A1A1A] cursor-pointer"
              onClick={() => setSelectedStrike(strike)}
            >
              {/* Call Row */}
              <div className={`flex text-[11px] divide-x divide-[#222222] ${callBg}`}>
                <div className="flex-1 text-right px-1 text-[#888888]">{call.bidSize}</div>
                <div className="flex-1 text-right px-1 text-[#00FF00] font-medium">{Number(call.bid) > 0 ? call.bid : '0.00'}</div>
                <div className="flex-1 text-right px-1 text-[#FF3333] font-medium">{Number(call.ask) > 0 ? call.ask : '0.00'}</div>
                <div className="flex-1 text-right px-1 text-[#888888]">{call.askSize}</div>
                <div className="flex-1 text-center px-1">
                  {call.position > 0 && <span className="bg-[#00CC66] text-black px-1 rounded-sm text-[9px]">{call.position}</span>}
                </div>
                <div className={`flex-1 text-right px-1 ${callItm ? 'bg-[#4A1515] text-[#FFFFFF]' : 'text-[#888888]'}`}>{call.impliedVol}</div>
                <div className="flex-1 text-right px-1 text-[#D4D4D4]">{call.delta.toFixed(3)}</div>
                <div className="flex-1 text-right px-1 text-[#D4D4D4]">{call.gamma.toFixed(3)}</div>
                <div className="flex-1 text-right px-1 text-[#D4D4D4]">{call.theta.toFixed(3)}</div>
              </div>

              {/* Strike */}
              <div className={`flex items-center justify-center font-bold text-[11px] ${strikeBg}`}>
                {strike}
              </div>

              {/* Put Row */}
              <div className={`flex text-[11px] divide-x divide-[#222222] ${putBg}`}>
                <div className="flex-1 text-right px-1 text-[#888888]">{put.bidSize}</div>
                <div className="flex-1 text-right px-1 text-[#00FF00] font-medium">{Number(put.bid) > 0 ? put.bid : '0.00'}</div>
                <div className="flex-1 text-right px-1 text-[#FF3333] font-medium">{Number(put.ask) > 0 ? put.ask : '0.00'}</div>
                <div className="flex-1 text-right px-1 text-[#888888]">{put.askSize}</div>
                <div className="flex-1 text-center px-1">
                  {put.position > 0 && <span className="bg-[#00CC66] text-black px-1 rounded-sm text-[9px]">{put.position}</span>}
                </div>
                <div className={`flex-1 text-right px-1 ${putItm ? 'bg-[#1A254A] text-[#FFFFFF]' : 'text-[#888888]'}`}>{put.impliedVol}</div>
                <div className="flex-1 text-right px-1 text-[#D4D4D4]">{put.delta.toFixed(3)}</div>
                <div className="flex-1 text-right px-1 text-[#D4D4D4]">{put.gamma.toFixed(3)}</div>
                <div className="flex-1 text-right px-1 text-[#D4D4D4]">{put.theta.toFixed(3)}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

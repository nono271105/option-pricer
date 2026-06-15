import React from 'react';

function StatRow({ label, value, highlight = false, color }: { label: string, value: string, highlight?: boolean, color?: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="font-['IBM_Plex_Mono',monospace] text-[10px] text-[#888888] uppercase tracking-wide">
        {label}
      </span>
      <span className={`font-['IBM_Plex_Mono',monospace] text-[13px] ${highlight ? 'text-[#4A90E2]' : color ? color : 'text-[#FFFFFF]'}`}>
        {value}
      </span>
    </div>
  );
}

export function RightPanel({ activeTab }: { activeTab: string }) {
  return (
    <aside className="w-[360px] border-l border-[#333333] bg-[#000000] h-full shrink-0 flex flex-col">
      <div className="p-4 border-b border-[#333333]">
        <h2 className="font-['IBM_Plex_Mono',monospace] text-[11px] uppercase tracking-[1.5px] text-[#888888]">
          Intelligence / {activeTab}
        </h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar space-y-6">
        
        {/* Default Section: DONNÉES MARCHÉ */}
        <section>
          <h3 className="font-['IBM_Plex_Mono',monospace] text-[11px] uppercase tracking-[1.5px] text-[#FFFFFF] mb-4 border-b border-[#333333] pb-2">
            DONNÉES MARCHÉ
          </h3>
          <div className="grid grid-cols-2 gap-y-4 gap-x-2">
            <StatRow label="Spot" value="5,123.45" />
            <StatRow label="Strike" value="5,200.00" />
            <StatRow label="Maturité (T)" value="0.25 (91j)" />
            <StatRow label="Moneyness" value="0.985" color="text-[#FF4444]" />
            <StatRow label="Taux Sans Risque" value="5.32%" />
            <StatRow label="Rendement Div" value="1.45%" />
            <StatRow label="IV Marché" value="14.20%" highlight />
            <StatRow label="Vol Historique" value="12.80%" />
          </div>
        </section>

        {activeTab === 'Exotics' && (
          <section className="mt-8">
            <h3 className="font-['IBM_Plex_Mono',monospace] text-[11px] uppercase tracking-[1.5px] text-[#FFFFFF] mb-4 border-b border-[#333333] pb-2">
              Monte Carlo Analysis
            </h3>
            <div className="space-y-4">
              <div className="bg-[#121212] p-3 border border-[#333333]">
                <div className="flex justify-between items-center mb-1">
                  <span className="font-['IBM_Plex_Mono',monospace] text-[10px] text-[#888888]">MC Price</span>
                  <span className="font-['IBM_Plex_Mono',monospace] text-[13px] text-[#4A90E2]">142.50</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-['IBM_Plex_Mono',monospace] text-[10px] text-[#888888]">Analytical</span>
                  <span className="font-['IBM_Plex_Mono',monospace] text-[13px] text-[#FFFFFF]">141.85</span>
                </div>
                <div className="mt-3 pt-3 border-t border-[#333333]">
                  <span className="font-['IBM_Plex_Mono',monospace] text-[10px] text-[#888888] block mb-1">Confidence Interval (95%)</span>
                  <div className="w-full h-1.5 bg-[#000000] rounded-full overflow-hidden relative">
                    <div className="absolute left-[20%] right-[20%] top-0 bottom-0 bg-[#4A90E2] opacity-30" />
                    <div className="absolute left-[49%] right-[49%] top-0 bottom-0 bg-[#4A90E2]" />
                  </div>
                  <div className="flex justify-between mt-1">
                    <span className="font-['IBM_Plex_Mono',monospace] text-[10px] text-[#888888]">140.10</span>
                    <span className="font-['IBM_Plex_Mono',monospace] text-[10px] text-[#888888]">144.90</span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {activeTab === 'Forecast' && (
          <section className="mt-8">
            <h3 className="font-['IBM_Plex_Mono',monospace] text-[11px] uppercase tracking-[1.5px] text-[#FFFFFF] mb-4 border-b border-[#333333] pb-2">
              IV Forecast
            </h3>
            <div className="bg-[#121212] p-3 border border-[#333333] h-32 flex flex-col justify-end relative">
               {/* Sparkline placeholder */}
               <div className="absolute inset-0 overflow-hidden opacity-50">
                  <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="w-full h-full">
                    <path d="M0 30 L20 25 L40 28 L60 15 L80 18 L100 5" fill="none" stroke="#4A90E2" strokeWidth="1" />
                    <path d="M80 18 L100 5 L100 40 L80 40 Z" fill="#4A90E2" opacity="0.1" />
                  </svg>
               </div>
               <span className="font-['IBM_Plex_Mono',monospace] text-[10px] text-[#888888] z-10 block">30D Forecast Trajectory</span>
            </div>
          </section>
        )}
      </div>
    </aside>
  );
}

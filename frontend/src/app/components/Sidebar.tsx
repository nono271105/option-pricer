import React from 'react';
import { 
  Calculator, 
  Binary, 
  Activity, 
  Smile, 
  Layers, 
  Zap, 
  GitMerge, 
  TrendingUp 
} from 'lucide-react';

const TABS = [
  { id: 'BSM', label: 'BSM', icon: Calculator },
  { id: 'CRR', label: 'CRR', icon: Binary },
  { id: 'Simulation', label: 'Simulation', icon: Activity },
  { id: 'Smile', label: 'Smile', icon: Smile },
  { id: 'Surface IV', label: 'Surface IV', icon: Layers },
  { id: 'Exotics', label: 'Exotics', icon: Zap },
  { id: 'Strategies', label: 'Strategies', icon: GitMerge },
  { id: 'Forecast', label: 'Forecast', icon: TrendingUp },
];

export function Sidebar({ activeTab, setActiveTab }: { activeTab: string, setActiveTab: (id: string) => void }) {
  return (
    <aside className="w-60 border-r border-[#333333] bg-[#000000] flex flex-col h-full shrink-0">
      <div className="p-4 border-b border-[#333333]">
        <h1 className="text-[#FFFFFF] font-['IBM_Plex_Mono',monospace] text-[13px] tracking-[1.5px] uppercase font-semibold">
          Pricer Terminal
        </h1>
      </div>
      
      <nav className="flex-1 overflow-y-auto py-4 flex flex-col gap-1">
        {TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-3 px-4 py-2.5 w-full text-left font-['IBM_Plex_Mono',monospace] text-[13px] transition-colors relative
                ${isActive ? 'text-[#4A90E2] bg-[#2D2D2D]' : 'text-[#888888] hover:text-[#D0D0D0] hover:bg-[#121212]'}
              `}
            >
              {isActive && (
                <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-[#4A90E2]" />
              )}
              <Icon size={16} strokeWidth={1.5} className={isActive ? 'text-[#4A90E2]' : 'text-[#888888]'} />
              {tab.label}
            </button>
          );
        })}
      </nav>

      {/* Market Snapshot Card */}
      <div className="p-4 border-t border-[#333333]">
        <div className="bg-[#121212] rounded-sm border border-[#333333] p-3">
          <div className="flex justify-between items-center mb-3">
            <span className="font-['IBM_Plex_Mono',monospace] text-[11px] uppercase tracking-[1.5px] text-[#888888]">
              Market Snapshot
            </span>
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-[#00FF00] animate-pulse" />
              <span className="font-['IBM_Plex_Mono',monospace] text-[10px] text-[#00FF00]">LIVE</span>
            </div>
          </div>
          
          <div className="space-y-2 font-['IBM_Plex_Mono',monospace] text-[13px]">
            <div className="flex justify-between items-center">
              <span className="text-[#888888]">TICKER</span>
              <span className="text-[#FFFFFF]">SPX</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#888888]">S</span>
              <span className="text-[#FFFFFF]">5,123.45</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#888888]">SOFR (r)</span>
              <span className="text-[#FFFFFF]">5.32%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#888888]">Div (q)</span>
              <span className="text-[#FFFFFF]">1.45%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[#888888]">ATM Vol (σ)</span>
              <span className="text-[#FFFFFF]">14.20%</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}

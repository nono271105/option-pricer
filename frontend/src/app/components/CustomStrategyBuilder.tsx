import React, { useEffect, useRef, useState } from 'react';
import Plot from './Plot';
import { useMarket } from '../App';
import { calculateCustomStrategy, getAvailableExpiries } from '../../lib/api';
import type {
  CustomStrategyLegInput,
  CustomStrategyResult,
} from '../../types/api';
import { formatDateFR } from '../../lib/dates';

interface DraftLeg extends Omit<CustomStrategyLegInput, 'strike'> {
  id: number;
  strike: string;
}

const MAX_LEGS = 50;
const INPUT = 'min-w-0 w-full bg-[#121212] border border-[#333333] text-white px-1.5 py-1 text-[11px] outline-none focus:border-[#4A90E2] disabled:opacity-50';
const SELECT = `${INPUT} appearance-none`;

function formatAmount(value: number): string {
  return `${value.toFixed(4)} $`;
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function CustomStrategyBuilder({ onBack }: { onBack: () => void }) {
  const market = useMarket();
  const [ticker, setTicker] = useState(market.ticker);
  const [expiries, setExpiries] = useState<string[]>([]);
  const [expiriesLoading, setExpiriesLoading] = useState(false);
  const [expiryError, setExpiryError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CustomStrategyResult | null>(null);
  const nextId = useRef(2);
  const [legs, setLegs] = useState<DraftLeg[]>([{
    id: 1,
    option_type: 'call',
    position: 'long',
    strike: String(Math.round(market.S ?? 100)),
    expiry_str: '',
  }]);

  useEffect(() => {
    const normalizedTicker = ticker.trim().toUpperCase();
    setExpiries([]);
    setExpiryError(null);
    setResult(null);
    if (!normalizedTicker) {
      setExpiriesLoading(false);
      return;
    }

    let cancelled = false;
    setExpiriesLoading(true);
    const timeout = setTimeout(async () => {
      try {
        const response = await getAvailableExpiries(normalizedTicker);
        if (cancelled) return;
        if (response.error) {
          setExpiryError(response.error);
          setLegs(current => current.map(leg => ({ ...leg, expiry_str: '' })));
          return;
        }
        if (response.expiries.length === 0) {
          setExpiryError(`Aucune échéance cotée disponible pour ${normalizedTicker}.`);
          setLegs(current => current.map(leg => ({ ...leg, expiry_str: '' })));
          return;
        }

        setExpiries(response.expiries);
        setLegs(current => current.map(leg => ({
          ...leg,
          expiry_str: response.expiries.includes(leg.expiry_str)
            ? leg.expiry_str
            : response.expiries[0],
        })));
      } catch (requestError: unknown) {
        if (!cancelled) {
          setExpiryError(describeError(requestError));
          setLegs(current => current.map(leg => ({ ...leg, expiry_str: '' })));
        }
      } finally {
        if (!cancelled) setExpiriesLoading(false);
      }
    }, 250);

    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  }, [ticker]);

  const updateLeg = <K extends keyof DraftLeg>(id: number, key: K, value: DraftLeg[K]) => {
    setResult(null);
    setLegs(current => current.map(leg => leg.id === id ? { ...leg, [key]: value } : leg));
  };

  const handleTickerChange = (value: string) => {
    setTicker(value.toUpperCase());
    setResult(null);
    setError(null);
    setLegs(current => current.map(leg => ({ ...leg, expiry_str: '' })));
  };

  const addLeg = () => {
    if (legs.length >= MAX_LEGS) return;
    setResult(null);
    setLegs(current => [...current, {
      id: nextId.current++,
      option_type: 'call',
      position: 'long',
      strike: String(Math.round(market.S ?? 100)),
      expiry_str: expiries[0] ?? '',
    }]);
  };

  const removeLeg = (id: number) => {
    if (legs.length <= 1) return;
    setResult(null);
    setLegs(current => current.filter(leg => leg.id !== id));
  };

  const invalidLeg = legs.some(leg =>
    !leg.expiry_str || !Number.isFinite(Number(leg.strike)) || Number(leg.strike) <= 0
  );

  const handleCalculate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const normalizedTicker = ticker.trim().toUpperCase();
      const quote = await market.ensureMarket(normalizedTicker);
      if (!quote || quote.S === null) {
        setError(`Données de marché indisponibles pour ${normalizedTicker}.`);
        return;
      }

      const requestLegs = legs.map(({ id: _id, strike, ...leg }) => ({
        ...leg,
        strike: Number(strike),
      }));
      const response = await calculateCustomStrategy(
        normalizedTicker,
        quote.S,
        quote.r,
        quote.q,
        requestLegs,
        quote.histVol,
      );
      if (response.error) {
        setError(response.error);
        return;
      }
      setResult(response);
    } catch (requestError: unknown) {
      setError(describeError(requestError));
    } finally {
      setLoading(false);
    }
  };

  const currentSpot = result?.ticker === market.ticker ? market.S : null;
  const flowColor = !result || result.initial_net_flow === 0
    ? 'text-[#D4D4D4]'
    : result.initial_net_flow < 0 ? 'text-[#FF6B6B]' : 'text-[#6BCB77]';

  return (
    <div className="flex flex-col h-full gap-1 p-1 overflow-auto bg-[#000000]">
      <section className="shrink-0 border border-[#222222]">
        <div className="flex items-center justify-between gap-2 px-2 py-1 bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222]">
          <h2 className="font-bold text-[11px] text-white">STRATÉGIE PERSONNALISÉE</h2>
          <button type="button" onClick={onBack}
            className="text-[10px] text-[#D4D4D4] hover:text-white focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#4A90E2] px-2 py-1">
            Retour aux stratégies prédéfinies
          </button>
        </div>
        <div className="flex flex-wrap items-end gap-3 p-2">
          <label className="flex flex-col gap-1 text-[10px] text-[#B8B8B8]">
            Ticker
            <input value={ticker} onChange={event => handleTickerChange(event.target.value)}
              className="bg-[#121212] border border-[#333333] text-white px-1.5 py-1 text-[11px] w-24 outline-none focus:border-[#4A90E2]"
              autoComplete="off" />
          </label>
          <span className="text-[10px] text-[#B8B8B8] pb-1">
            {market.ticker === ticker.trim().toUpperCase() && market.S !== null
              ? `Spot ${market.S.toFixed(2)} $`
              : 'Spot récupéré à l’analyse'}
          </span>
          <span className="text-[10px] text-[#888888] pb-1">
            {expiriesLoading
              ? 'Chargement des échéances cotées…'
              : expiries.length > 0
                ? `${expiries.length} échéances cotées disponibles`
                : 'Aucune échéance chargée'}
          </span>
        </div>
        {expiryError && (
          <p role="alert" className="mx-2 mb-2 border border-[#FF4444] bg-[#3D0000] px-2 py-1.5 text-[11px] text-[#FF9999]">
            {expiryError}
          </p>
        )}
        {error && (
          <p role="alert" className="mx-2 mb-2 border border-[#FF4444] bg-[#3D0000] px-2 py-1.5 text-[11px] text-[#FF9999]">
            {error}
          </p>
        )}

        <div className="overflow-x-auto border-t border-[#222222]">
          <table className="w-full min-w-[720px] table-fixed border-collapse text-[11px]">
            <colgroup>
              <col className="w-[13%]" />
              <col className="w-[11%]" />
              <col className="w-[22%]" />
              <col className="w-[23%]" />
              <col className="w-[11%]" />
              <col className="w-[20%]" />
            </colgroup>
            <thead>
              <tr className="bg-[#0D0D0D] text-left text-[9px] uppercase text-[#B8B8B8] border-b border-[#222222]">
                <th scope="col" className="px-2 py-1.5 font-normal">Position</th>
                <th scope="col" className="px-2 py-1.5 font-normal">Type</th>
                <th scope="col" className="px-2 py-1.5 font-normal">Strike</th>
                <th scope="col" className="px-2 py-1.5 font-normal">Échéance</th>
                <th scope="col" className="px-2 py-1.5 font-normal">Unité</th>
                <th scope="col" className="px-2 py-1.5 font-normal text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {legs.map((leg, index) => (
                <tr key={leg.id} className="border-b border-[#1A1A1A] hover:bg-[#0A0A0A]">
                  <td className="px-2 py-1.5">
                    <label className="sr-only" htmlFor={`custom-position-${leg.id}`}>Position de la jambe {index + 1}</label>
                    <select id={`custom-position-${leg.id}`} value={leg.position}
                      onChange={event => updateLeg(leg.id, 'position', event.target.value as DraftLeg['position'])}
                      className={SELECT}>
                      <option value="long">Long</option>
                      <option value="short">Short</option>
                    </select>
                  </td>
                  <td className="px-2 py-1.5">
                    <label className="sr-only" htmlFor={`custom-type-${leg.id}`}>Type de la jambe {index + 1}</label>
                    <select id={`custom-type-${leg.id}`} value={leg.option_type}
                      onChange={event => updateLeg(leg.id, 'option_type', event.target.value as DraftLeg['option_type'])}
                      className={SELECT}>
                      <option value="call">Call</option>
                      <option value="put">Put</option>
                    </select>
                  </td>
                  <td className="px-2 py-1.5">
                    <label className="sr-only" htmlFor={`custom-strike-${leg.id}`}>Strike de la jambe {index + 1}</label>
                    <input id={`custom-strike-${leg.id}`} type="number" min="0.01" step="0.01"
                      value={leg.strike} onChange={event => updateLeg(leg.id, 'strike', event.target.value)}
                      className={INPUT} />
                  </td>
                  <td className="px-2 py-1.5">
                    <label className="sr-only" htmlFor={`custom-expiry-${leg.id}`}>Échéance de la jambe {index + 1}</label>
                    <select id={`custom-expiry-${leg.id}`} value={leg.expiry_str}
                      onChange={event => updateLeg(leg.id, 'expiry_str', event.target.value)}
                      disabled={expiriesLoading || expiries.length === 0}
                      className={SELECT}>
                      {expiries.length === 0
                        ? <option value="">{expiriesLoading ? 'Chargement…' : 'Indisponible'}</option>
                        : expiries.map(expiry => (
                          <option key={expiry} value={expiry}>{formatDateFR(expiry)}</option>
                        ))}
                    </select>
                  </td>
                  <td className="px-2 py-1.5 text-[#D4D4D4]">1 unité</td>
                  <td className="px-2 py-1.5 text-right">
                    <button type="button" onClick={() => removeLeg(leg.id)}
                      disabled={legs.length <= 1 || loading}
                      aria-label={`Supprimer la jambe ${index + 1}`}
                      className="px-2 py-1 text-[#FF9999] hover:text-white disabled:text-[#666666] disabled:cursor-not-allowed focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#4A90E2]">
                      Supprimer
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex flex-wrap items-center gap-2 p-2">
          <button type="button" onClick={addLeg} disabled={legs.length >= MAX_LEGS || loading}
            className="border border-[#444444] px-2.5 py-1 text-[10px] font-bold text-white hover:bg-[#1A1A1A] disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-[#4A90E2]">
            + Ajouter une jambe
          </button>
          <span className="text-[10px] text-[#B8B8B8]">
            {legs.length}/{MAX_LEGS} jambes · montants par unité, multiplicateur de contrat non appliqué
          </span>
          <button type="button" onClick={handleCalculate}
            disabled={loading || expiriesLoading || expiries.length === 0 || invalidLeg}
            className="ml-auto bg-[#4A90E2] px-4 py-1.5 text-[10px] font-bold text-white hover:bg-[#357ABD] disabled:opacity-50 focus-visible:outline focus-visible:outline-1 focus-visible:outline-white">
            {loading ? 'Calcul…' : 'Calculer la stratégie'}
          </button>
        </div>
      </section>

      {result && (
        <>
          <div className="grid grid-cols-1 gap-1 xl:grid-cols-[minmax(0,1.5fr)_minmax(240px,0.8fr)]">
            <section className="min-w-0 border border-[#222222]">
              <h3 className="px-2 py-1 bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] text-[10px] font-bold text-white">
                COMPOSITION · {result.ticker}
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[680px] table-fixed border-collapse text-right text-[11px]">
                  <colgroup>
                    <col className="w-[10%]" />
                    <col className="w-[8%]" />
                    <col className="w-[11%]" />
                    <col className="w-[13%]" />
                    <col className="w-[12%]" />
                    <col className="w-[26%]" />
                    <col className="w-[20%]" />
                  </colgroup>
                  <thead>
                    <tr className="bg-[#0D0D0D] text-[9px] uppercase text-[#B8B8B8] border-b border-[#222222]">
                      <th scope="col" className="px-2 py-1.5 text-left font-normal">Position</th>
                      <th scope="col" className="px-2 py-1.5 font-normal">Option</th>
                      <th scope="col" className="px-2 py-1.5 font-normal">Strike</th>
                      <th scope="col" className="px-2 py-1.5 font-normal">Échéance</th>
                      <th scope="col" className="px-2 py-1.5 font-normal">Prime</th>
                      <th scope="col" className="px-2 py-1.5 font-normal">Origine</th>
                      <th scope="col" className="px-2 py-1.5 font-normal">Flux initial</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.legs.map((leg, index) => {
                      const flow = leg.position === 'long' ? -leg.premium : leg.premium;
                      return (
                        <tr key={`${leg.expiry}-${index}`} className="border-b border-[#1A1A1A]">
                          <td className={`px-2 py-1.5 text-left font-bold ${leg.position === 'long' ? 'text-[#6BCB77]' : 'text-[#FF6B6B]'}`}>
                            {leg.position.toUpperCase()}
                          </td>
                          <td className="px-2 py-1.5 text-[#D4D4D4]">{leg.option_type.toUpperCase()}</td>
                          <td className="px-2 py-1.5 font-bold text-white">{leg.strike.toFixed(2)}</td>
                          <td className="px-2 py-1.5 text-[#D4D4D4]">{formatDateFR(leg.expiry)}</td>
                          <td className="px-2 py-1.5 text-[#D4D4D4]">{formatAmount(leg.premium)}</td>
                          <td className="px-2 py-1.5 text-left text-[#B8B8B8]">
                            <span>{leg.premium_source}{leg.expiry_used ? ` · ${formatDateFR(leg.expiry_used)}` : ''}</span>
                            <span className="block text-[9px] text-[#888888]">{leg.sigma_source}</span>
                            {leg.pricing_note && <span className="block text-[9px] text-[#FFCC00]">{leg.pricing_note}</span>}
                          </td>
                          <td className={`px-2 py-1.5 font-bold ${flow < 0 ? 'text-[#FF6B6B]' : 'text-[#6BCB77]'}`}>
                            {flow > 0 ? '+' : ''}{formatAmount(flow)}
                          </td>
                        </tr>
                      );
                    })}
                    <tr className="bg-[#0D0D0D]">
                      <th scope="row" colSpan={6} className="px-2 py-1.5 text-left text-[9px] uppercase text-[#B8B8B8]">
                        Flux net initial
                      </th>
                      <td className={`px-2 py-1.5 font-bold ${flowColor}`}>
                        {result.initial_net_flow > 0 ? '+' : ''}{formatAmount(result.initial_net_flow)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            <section className="border border-[#222222]">
              <h3 className="px-2 py-1 bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] text-[10px] font-bold text-white">
                VALEUR ET RISQUE
              </h3>
              <div className="p-2">
                <MetricRow label="Valeur théorique aujourd’hui" value={formatAmount(result.current_model_value)} />
                <MetricRow label="P&L modèle vs flux initial"
                  value={`${result.current_model_pnl > 0 ? '+' : ''}${formatAmount(result.current_model_pnl)}`}
                  valueClass={result.current_model_pnl < 0 ? 'text-[#FF6B6B]' : 'text-[#6BCB77]'} />
                {result.has_shared_expiry ? (
                  <>
                    <MetricRow label="Gain maximal"
                      value={result.metrics.max_gain === null ? 'Illimité' : `${result.metrics.max_gain > 0 ? '+' : ''}${formatAmount(result.metrics.max_gain)}`}
                      valueClass="text-[#6BCB77]" />
                    <MetricRow label="Perte maximale"
                      value={result.metrics.max_loss === null ? 'Illimitée' : `-${formatAmount(result.metrics.max_loss)}`}
                      valueClass="text-[#FF6B6B]" />
                    {result.metrics.breakevens.map((breakeven, index) => (
                      <MetricRow key={index} label={`Seuil de rentabilité${result.metrics.breakevens.length > 1 ? ` ${index + 1}` : ''}`}
                        value={`${breakeven.toFixed(2)} $`} />
                    ))}
                  </>
                ) : (
                  <p className="my-2 border border-[#665200] bg-[#2D2200] px-2 py-1.5 text-[10px] leading-relaxed text-[#FFDC70]">
                    Échéances différentes : le payoff final et ses gains/pertes maximaux ne sont pas calculés, car ils dépendent du prix du sous-jacent à chaque expiration.
                  </p>
                )}
                <div className="my-1.5 h-px bg-[#222222]" />
                <p className="mb-1 text-[9px] uppercase tracking-widest text-[#B8B8B8]">Grecs agrégés aujourd’hui</p>
                <div className="grid grid-cols-2 gap-x-3">
                  <MetricRow label="Delta" value={result.greeks.delta.toFixed(4)} />
                  <MetricRow label="Gamma" value={result.greeks.gamma.toFixed(4)} />
                  <MetricRow label="Theta / jour" value={result.greeks.theta.toFixed(4)} />
                  <MetricRow label="Vega / 1 %" value={result.greeks.vega.toFixed(4)} />
                  <MetricRow label="Rho / 1 %" value={result.greeks.rho.toFixed(4)} />
                </div>
              </div>
            </section>
          </div>

          <section className="flex min-h-[340px] flex-1 flex-col border border-[#222222]">
            <h3 className="px-2 py-1 bg-gradient-to-b from-[#2A2A2A] to-[#111111] border-b border-[#222222] text-[10px] font-bold text-white">
              VALEUR THÉORIQUE / P&amp;L MODÈLE AUJOURD’HUI SELON LE SPOT
            </h3>
            <div className="min-h-[300px] flex-1 bg-[#050505] p-1">
              <Plot
                data={[
                  ...(result.has_shared_expiry ? [{
                    x: result.payoff_data.map(point => point.spot),
                    y: result.payoff_data.map(point => point.payoff),
                    type: 'scatter' as const,
                    mode: 'lines' as const,
                    line: { color: '#4A90E2', width: 2 },
                    name: 'Payoff net à échéance',
                    hovertemplate: 'Spot final : %{x:.2f} $<br>Payoff : <b>%{y:.4f} $</b><extra></extra>',
                  }] : []),
                  {
                    x: result.value_today_data.map(point => point.spot),
                    y: result.value_today_data.map(point => point.value),
                    type: 'scatter' as const,
                    mode: 'lines' as const,
                    line: { color: '#FFCC00', width: 2 },
                    name: 'P&L modèle aujourd’hui',
                    hovertemplate: 'Spot hypothétique : %{x:.2f} $<br>P&L modèle : <b>%{y:.4f} $</b><extra></extra>',
                  },
                ]}
                layout={{
                  autosize: true,
                  margin: { l: 55, r: 20, t: 12, b: 45 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#B8B8B8', size: 10 },
                  xaxis: { title: 'Prix du sous-jacent ($)', gridcolor: '#222222', zerolinecolor: '#444444' },
                  yaxis: { title: 'P&L par option ($)', gridcolor: '#222222', zerolinecolor: '#666666', zeroline: true },
                  hovermode: 'x unified',
                  hoverlabel: { bgcolor: '#1A1A1A', bordercolor: '#444444', font: { color: '#FFFFFF' } },
                  legend: { orientation: 'h', y: -0.2, font: { color: '#B8B8B8', size: 9 }, bgcolor: 'transparent' },
                  shapes: currentSpot === null ? [] : [{
                    type: 'line', xref: 'x', yref: 'paper', x0: currentSpot, x1: currentSpot, y0: 0, y1: 1,
                    line: { color: '#888888', dash: 'dot', width: 1 },
                  }],
                }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler
              />
            </div>
          </section>
        </>
      )}
      {!result && !loading && !error && !expiryError && (
        <div className="flex flex-1 items-center justify-center text-center text-[11px] text-[#B8B8B8]">
          Configurez une ou plusieurs jambes, puis calculez la stratégie.
        </div>
      )}
      {loading && (
        <div role="status" className="flex flex-1 items-center justify-center text-[11px] text-[#FFCC00]">
          Valorisation des jambes et calcul des grecs…
        </div>
      )}
    </div>
  );
}

function MetricRow({ label, value, valueClass }: { label: string; value: string; valueClass?: string }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-[#171717] py-1">
      <span className="text-[10px] text-[#B8B8B8]">{label}</span>
      <span className={`text-right text-[11px] font-bold tabular-nums ${valueClass ?? 'text-white'}`}>{value}</span>
    </div>
  );
}

import React, { lazy, Suspense } from 'react';
import type { PlotParams } from 'react-plotly.js';

// Chargement paresseux de Plotly pour optimiser le bundle initial
const LazyPlot = lazy(async () => {
  const [factoryModule, plotlyModule] = await Promise.all([
    import('react-plotly.js/factory'),
    import('plotly.js-dist-min'),
  ]);
  const createPlotlyComponent = factoryModule.default || factoryModule;
  const Plotly = plotlyModule.default || plotlyModule;
  const Component = createPlotlyComponent(Plotly);
  return { default: Component };
});

export default function Plot(props: PlotParams) {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-full w-full text-[#888888] text-[11px] bg-transparent">
          Chargement du graphique...
        </div>
      }
    >
      <LazyPlot {...props} />
    </Suspense>
  );
}

Option Pricer — UI Design Brief for Figma
Concept direction : Terminal Bloomberg reinterpreted / Trading desk noir
Dark interface. Dense but structured. Every pixel earns its place. Think a Bloomberg terminal that went through a Braun product redesign — no decoration for decoration's sake, but extreme functional clarity with a character that feels authored, not assembled.

Color palette
Background: #0B0C0F (near-black, not pure black — gives depth without harshness)
Surface / card: #13151A
Border / separator: #1F2229
Muted text: #4A5568
Body text: #C9CDD6
Primary text: #F0F2F6
Accent — electric cyan: #00D4FF (used sparingly: active states, key values, live data)
Positive / gain: #00C896
Negative / loss: #FF4D6A
Neutral / flat: #F5A623
No gradients except one allowed use: a very subtle radial from #00D4FF08 behind the main chart area.

Typography
Heading / labels: IBM Plex Mono — monospaced, grounded, precise. Feels native to financial data.
Body / input values: IBM Plex Sans Light — clean contrast against the mono labels.
Large data figures (live price, option price): Teko SemiBold — compressed, tall, reads fast at a glance.
Font sizes:

Main metric readout: 36–40px
Section header: 11px uppercase, 1.5 letter-spacing
Input labels: 11px mono
Table values: 13px mono

No Inter. No Roboto. No SF Pro.

Layout
Three-column layout on desktop (1440px base):
Left panel (240px fixed) — Navigation + Context

Vertical tab list with icons: BSM, CRR, Simulation, Smile, Surface IV, Exotics, Strategies, Forecast
Active tab has a left accent bar in cyan #00D4FF, background #1A1D24
Below tabs: Market snapshot card (Ticker, S live, SOFR r, q, σ) — monospaced values, pulsing green dot for live data

Center panel (flex, ~720px) — Main Workspace

Top: compact form row (Ticker / Type / K / Maturity / Position) — horizontal layout, not stacked. Inputs have 1px solid #1F2229 border, #0F1117 fill, cyan focus ring 1px
Primary result display: large isolated card showing "Option Price (BSM)" in Teko 40px — no fluff around it
Greeks row: 5 compact cells in a horizontal band — Delta / Gamma / Theta / Vega / Rho — each cell shows name in muted 10px caps, value in mono 18px. Color-coded positive/negative
Below: Matplotlib payoff chart area — dark axes (#1F2229 grid lines), no chart border. Single line in cyan for payoff, orange for breakeven marker. No legend box background

Right panel (360px fixed) — Intelligence panel

"Données Actuelles" section: vertically stacked label-value pairs in two columns
On Exotic tab: this panel shows MC results + CI bar + Analytique vs MC comparison
On Forecast tab: IV history sparkline + forecast ribbon


Components
Inputs: 36px height, monospaced values, no visible shadow, thin 1px border, full-width in their container. Placeholder in muted. On focus: border becomes #00D4FF.
Buttons: Two variants only.

Primary: #00D4FF background, #0B0C0F text, 0 border-radius (sharp corners). Hover: 10% darker.
Ghost: transparent background, #1F2229 border. Hover: border becomes #00D4FF, text becomes cyan.

Tables (Greeks, Legs): No outer border on the table. Rows separated by 1px #1F2229. Header in 10px uppercase muted mono. Values right-aligned. Alternating row tint: every other row at #13151A.
Dropdown/Combo: Same style as input, chevron icon in muted gray. Option list has #0B0C0F background, active item highlighted with #00D4FF14 fill and left accent bar 2px.
Status / Pricing method badge: Pill component — #1F2229 background, 12px mono text, cyan dot prefix for "IV Marché", orange dot for "Vol Historique".
Charts (Payoff, Greeks evolution, Smile, Forecast):

No chart title inside the canvas — title lives above in a section header
Axis lines: #1F2229
Grid: #1A1D22
Main data line: #00D4FF at 1.5px
Secondary / reference lines: #F5A623 dashed
Fill-under-curve (payoff positive zone): #00D4FF08
Fill-under-curve (payoff negative zone): #FF4D6A08


Specific screens to design

BSM Tab (primary view, fully built out)
Exotics Tab — show the 2×2 MC chart panel: paths / distribution / payoff profile. The paths chart with semi-transparent overlapping lines is the hero visual of the app.
Volatility Surface Tab — the 3D Plotly surface embedded in a dark frame with controls above. Show the gradient from low IV (deep blue) to high IV (magenta-yellow) using Plasma colorscale on a dark background.
Strategies Tab — the legs table with color-coded long (green tint) / short (red tint) rows, then the P&L profile chart below.


Do not include

Any card drop shadow (use border instead)
Rounded corners above 2px on data components (inputs, tables, buttons may have 0–2px max)
Any light background areas
Any illustration or icon that isn't a functional UI element
Any color gradient as a UI decoration — only on the chart fill areas
Emojis or decorative symbols in the UI


Mood references to keep in mind (without copying)
Refinitiv Eikon terminal dark mode. Figma's own dev tools panel. Linear app's command palette. The density and economy of a circuit board layout.
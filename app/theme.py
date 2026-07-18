"""Palette and chart defaults for the operational console.

One dark palette, one accent. Status colors are used for TEXT only — status is
always written out as words, never encoded as color alone.
"""
BG = "#0D1117"
SURFACE = "#161B22"
BORDER = "#30363D"
TEXT = "#E6EDF3"
TEXT_DIM = "#8B949E"
ACCENT = "#6EA8CF"      # the single accent (steel blue)
NEUTRAL = "#B0BAC4"     # secondary series
MUTED = "#6E7681"       # benchmark series
GREEN = "#3FB950"       # PASS text
AMBER = "#D29922"       # LOW EVIDENCE / boundary text
RED = "#F85149"         # FAILED text

MODEL_COLORS = {"xgb": ACCENT, "lstm": NEUTRAL, "hodl": MUTED}
MONO = "ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, Consolas, monospace"


def plotly_layout(**overrides):
    """Shared plotly layout: dark, gridded, no animation, single colorway."""
    layout = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, size=12),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        colorway=[ACCENT, NEUTRAL, MUTED],
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    layout.update(overrides)
    return layout


CSS = f"""
<style>
.mono {{ font-family: {MONO}; font-size: 0.9em; }}
.status-pass {{ color: {GREEN}; }}
.status-warn {{ color: {AMBER}; }}
.status-fail {{ color: {RED}; }}
.status-dim  {{ color: {TEXT_DIM}; }}
[data-testid="stMetricValue"], .num {{ font-variant-numeric: tabular-nums; }}
.metric-label {{ color: {TEXT_DIM}; font-size: 0.75rem; text-transform: uppercase;
                letter-spacing: 0.04em; }}
.metric-value {{ font-family: {MONO}; font-size: 1.15rem;
                font-variant-numeric: tabular-nums; }}
.disclaimer-box {{ border: 1px solid {BORDER}; border-left: 3px solid {AMBER};
                  background: {SURFACE}; padding: 0.6rem 0.8rem; font-size: 0.85rem;
                  margin-bottom: 0.75rem; }}
</style>
"""

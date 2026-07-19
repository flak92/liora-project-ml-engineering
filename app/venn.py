"""Pixel diagram 'XGB vs LSTM — asset-level result agreement' for one computed basket.

NOT a set intersection. Nothing here intersects the two models' TRADE sets — that comparison is
not possible from sealed per-asset rows. What is drawn is DESCRIPTIVE agreement at the ASSET
level: on how many of a basket's assets did both models end net-positive, and on how many did
both end net-negative. Each share carries its own denominator (that model's own basket trades),
so the two are not two halves of one quantity. It is a readable summary, not a statistical
measure of model complementarity.

Geometry and artwork are the reference mock verbatim: a 46×46 grid where **1 square = 1
percentage point**, each model occupying 100 squares (100 % of ITS OWN sealed OOS trades in
the basket), a Wins disk of 200 and a Loses ring of 1000, model colors toned by region, dark
intersection cells, white anchor dots on the Wins rim, an Alpha/Edge line. Only the sliders are
gone — every count is baked in from the sealed stores by the caller.

`winrate_note()` carries the diagram's 50 % line: a warning when a model wins fewer than half of
its trades in the basket. Only tickers where BOTH models produced a real strategy result
(>= 2 trades) enter the diagram — a buy-and-hold fallback is one trade and is not a strategy.

Read-only: rendered through st.iframe; no dependencies, no requests. This module imports
nothing on purpose — it is artwork plus the disclaimer that belongs to it, so it stays
trivially reviewable and diffable. Every number it renders arrives from the caller, which
reads it from the sealed store: the realized payoff and the OOS windows are parameters
here, not literals, so this file can never drift from the data.
"""

WINRATE_LINE_PCT = 50       # the 50 % reference line: fewer than half the trades won → warning
MIN_STRATEGY_TRADES = 2     # a PROMOTED strategy needs the Train-OOF floor met and >= 2 MODEL trades;
                            # the sealed label result_mode == 'ML_MULTI_TRADE' is exactly that pair

_TEMPLATE = """
<meta charset="utf-8">
<div style="background:#0F1115;color:#E6E8EB;border-radius:10px;padding:18px 14px 26px;
  font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,Arial,sans-serif;max-width:720px;margin:0 auto">
  <h1 style="font-size:18px;font-weight:600;margin:0 0 4px;letter-spacing:.2px">XGB vs LSTM — asset-level result agreement</h1>
  <p style="color:#7D848D;font-size:12.5px;margin:0 0 14px">Pixel Venn · 1 square = 1 pp of that model's
    sealed trades · 100 squares each ·
    basket: __N__ tickers · XGB __TX__ · LSTM __TL__ trades</p>
  <canvas id="vd_cv" width="552" height="552" style="width:100%;max-width:552px;height:auto;display:block;margin:0 auto"></canvas>
  <div style="display:flex;flex-direction:column;gap:6px;font-size:13px;margin:18px 0 0;font-variant-numeric:tabular-nums">
    <div><span style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px;background:#4A9D9E"></span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;margin-left:-3px;vertical-align:-1px;background:#7D7EAE"></span>XGB — Wins: __XW__ % · Loses: __XL__ % (of __TX__ trades · __WX__ won)</div>
    <div><span style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px;background:#BCBE1B"></span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;margin-left:-3px;vertical-align:-1px;background:#EF9F2B"></span>LSTM — Wins: __LW__ % · Loses: __LL__ % (of __TL__ trades · __WL__ won)</div>
    <div><span style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px;background:#27500A"></span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;margin-left:-3px;vertical-align:-1px;background:#791F1F"></span>Both models agree on the asset — Wins: __BW__ pp · Loses: __BL__ pp</div>
    <div style="color:#A3AAB3;font-size:12px;padding-left:2px">↳ raw — Wins: XGB __BWX__ pp · LSTM __BWL__ pp · Loses: XGB __BLX__ pp · LSTM __BLL__ pp</div>
    <div style="color:#A3AAB3">Agreement mix (Wins share of drawn agreement): <span style="font-weight:600">__IWR__</span></div>
    <div style="color:#A3AAB3">Alpha/Edge (Wins outside the agreement): XGB <span style="font-weight:600">__AX__</span> + LSTM <span style="font-weight:600">__AL__</span> = <span style="font-weight:600">__AT__</span></div>
    <div style="font-size:11.5px;color:#7D848D;line-height:1.6;margin-top:4px">
    <b>Read this as agreement, not as set arithmetic.</b> The <b>∩</b> glyph is borrowed shorthand and
    is stretched here: no intersection of the two models' TRADE sets is computed or drawn — that is not
    derivable from sealed per-asset rows. <b>∩</b> means only that both models ended net-positive (Wins)
    or both net-negative (Loses) <i>on the same asset</i>, and each share uses its OWN trade count as
    denominator, so the two are not two halves of one quantity; the drawn block is the smaller and the
    surplus stays in model color. <b>Win</b> = winning trade in that model's sealed OOS row.
    <b>Alpha/Edge</b> = each model's wins outside its own mutual-win share (0 when the basket's wins are
    fully shared) — a descriptive count, not a measure of model complementarity. When one model trades
    far more than the other, the pixel blocks use the smaller shared share for geometry, so a few tinted
    cells can sit outside the Alpha count. Promoted strategies only: both models must carry the sealed
    result_mode ML_MULTI_TRADE (Train-OOF floor met and ≥ 2 model trades) — a benchmark fallback carries
    zero model trades, and a floor-not-met row is a diagnostic replay. Cell color = model toned by
    region; ∩ dark. OOS windows: __WINX__ (XGB) / __WINL__ (LSTM).</div>
  </div>
</div>
<script>
(function(){
"use strict";
var COLS=46,ROWS=46,CELL=12,NW=200,NL=1000;
var XW=__XW__,LW=__LW__,BW=__BW__,BL=__BL__,AE={x:__AX__,l:__AL__};
var CX=COLS/2,CY=ROWS/2;
var cv=document.getElementById('vd_cv');
var dpr=Math.min(window.devicePixelRatio||1,2);
cv.width=COLS*CELL*dpr;cv.height=ROWS*CELL*dpr;
var ctx=cv.getContext('2d');ctx.scale(dpr,dpr);

var all=[];
for(var r=0;r<ROWS;r++)for(var c=0;c<COLS;c++){var dx=c+0.5-CX,dy=r+0.5-CY;all.push({r:r,c:c,d2:dx*dx+dy*dy});}
all.sort(function(a,b){return a.d2-b.d2;});
var disk=[],OWN=[];
for(var i=0;i<ROWS;i++)OWN.push(new Uint8Array(COLS));
for(var i=0;i<NW+NL;i++){var t=all[i];disk.push({r:t.r,c:t.c,reg:i<NW?1:2,x:t.c+0.5,y:t.r+0.5});}

var rW=Math.sqrt(NW/Math.PI);
function pt(d){var a=d*Math.PI/180;return{x:CX+rW*Math.cos(a),y:CY-rW*Math.sin(a)};}
var A1=pt(120),A2=pt(60),AM=pt(90);

var BASE={1:'#639922',2:'#E24B4A'};
var OWNCOL={1:{1:'#4A9D9E',2:'#7D7EAE'},2:{1:'#BCBE1B',2:'#EF9F2B'},3:{1:'#27500A',2:'#791F1F'}};

function take(n,reg,a,val){
  if(n<=0)return;
  var cand=[];
  for(var i=0;i<disk.length;i++){var t=disk[i];
    if(t.reg!==reg||OWN[t.r][t.c])continue;
    var dx=t.x-a.x,dy=t.y-a.y;cand.push([dx*dx+dy*dy,t]);}
  cand.sort(function(u,v){return u[0]-v[0];});
  var m=Math.min(n,cand.length);
  for(var i=0;i<m;i++)OWN[cand[i][1].r][cand[i][1].c]=val;
}
function alloc(){
  for(var i=0;i<ROWS;i++)OWN[i].fill(0);
  take(BW,1,AM,3);take(BL,2,AM,3);
  take(XW-BW,1,A1,1);take(100-XW-BL,2,A1,1);
  take(LW-BW,1,A2,2);take(100-LW-BL,2,A2,2);
}
function lab(t,x,y,size){
  ctx.font='600 '+size+'px ui-sans-serif,system-ui';
  ctx.textAlign='center';ctx.textBaseline='middle';
  ctx.lineWidth=3;ctx.strokeStyle='rgba(0,0,0,0.55)';ctx.strokeText(t,x,y);
  ctx.fillStyle='#fff';ctx.fillText(t,x,y);
}
function cen(v){
  var sx=0,sy=0,n=0;
  for(var i=0;i<disk.length;i++){var t=disk[i];if(OWN[t.r][t.c]===v){sx+=t.x;sy+=t.y;n++;}}
  return n?{x:sx/n*CELL,y:sy/n*CELL}:null;
}
function draw(){
  ctx.clearRect(0,0,COLS*CELL,ROWS*CELL);
  for(var i=0;i<disk.length;i++){var t=disk[i];var o=OWN[t.r][t.c];
    ctx.fillStyle=o?OWNCOL[o][t.reg]:BASE[t.reg];
    ctx.fillRect(t.c*CELL,t.r*CELL,CELL-1,CELL-1);}
  ctx.save();
  ctx.setLineDash([6,4]);ctx.lineWidth=2;ctx.strokeStyle='rgba(255,255,255,0.9)';
  ctx.beginPath();ctx.arc(CX*CELL,CY*CELL,rW*CELL,0,6.2832);ctx.stroke();
  ctx.restore();
  var an=[A1,A2];
  for(var i=0;i<2;i++){ctx.beginPath();ctx.arc(an[i].x*CELL,an[i].y*CELL,4.5,0,6.2832);
    ctx.fillStyle='#fff';ctx.fill();ctx.lineWidth=1.5;ctx.strokeStyle='#222222';ctx.stroke();}
  lab('Wins',CX*CELL,(CY+4.3)*CELL,13);
  lab('Loses',CX*CELL,(CY+15.2)*CELL,13);
  var c1=cen(1);if(c1)lab('XGB',c1.x,c1.y,12);
  var c2=cen(2);if(c2)lab('LSTM',c2.x,c2.y,12);
  var c3=cen(3);if(c3)lab('∩',c3.x,c3.y,14);
  lab('Alpha/Edge: XGB '+AE.x+' + LSTM '+AE.l+' = '+(AE.x+AE.l),CX*CELL,(CY+21.3)*CELL,13);
}
alloc();draw();
})();
</script>
"""


def winrate_note(xw: int, lw: int, payoff: dict) -> str | None:
    """The 50 % line for this basket, or None when BOTH models win 50 %+ of their trades.

    `payoff` is data.payoff_ratios(): {'xgb': {'median_payoff', 'n'}, 'lstm': {...}}. The
    realized ratio used to be written into this sentence by hand; it is measured now, so
    the warning cannot outlive the epoch it describes.
    """
    below = [n for n, v in (("XGBoost", xw), ("LSTM", lw)) if v < WINRATE_LINE_PCT]
    if not below:
        return None
    who = (f"**{below[0]}** is below it" if len(below) == 1
           else "**both models** are below it")
    px, pl = payoff.get("xgb", {}), payoff.get("lstm", {})
    ratio = (f"about {px['median_payoff']:.2f}:1 (XGB, median over {px['n']} promoted assets) / "
             f"{pl['median_payoff']:.2f}:1 (LSTM, over {pl['n']})"
             if px.get("median_payoff") and pl.get("median_payoff") else "well under the nominal")
    return (
        f"**Below the {WINRATE_LINE_PCT} % line.** XGBoost wins **{xw} %** of its trades here, "
        f"LSTM **{lw} %** — {who}. The barriers are asymmetric, so a win pays more than a loss "
        f"costs — but the REALIZED ratio is {ratio}, not the nominal "
        f"2:1, once costs, gaps and fills are paid — "
        f"so read this with the profit factor and the net result above."
    )


def venn_html(xw: int, lw: int, bw: int, bl: int, meta: dict, windows: dict = None) -> str:
    """The diagram for one basket, in percentage points of each model's OOS trades.

    xw / lw   — winning trades as a share of that model's basket trades (0..100)
    bw / bl   — the drawn intersection blocks (pp): trades on tickers where both models ended
                net-positive (bw) / net-negative (bl); the smaller of the two models' shares.
    meta      — tx/tl (trade totals), wx/wl (won trades), bw_x/bw_l/bl_x/bl_l (raw ∩ shares,
                one per model), n (tickers scored by both stores).
    Counts are clamped to consistency (bw ≤ min(xw, lw); bl ≤ min of the loses) so rounding at
    the caller can never overfill a region.

    Alpha/Edge is reported per model against its OWN mutual-win share — XGB = xw − round(bw_x),
    LSTM = lw − round(bw_l) — so a fully-shared basket reads 0. `bw` stays the drawn (min) block
    for the pixel geometry; the two only diverge on lopsided baskets (see the caption note).
    """
    xw, lw = max(0, min(int(xw), 100)), max(0, min(int(lw), 100))
    bw = max(0, min(int(bw), xw, lw))
    bl = max(0, min(int(bl), 100 - xw, 100 - lw))
    # Alpha/Edge is measured per model against its OWN mutual-win share (bw_x / bw_l), not the
    # drawn block bw = min(...). Two models with different trade counts have different consensus
    # shares, so subtracting the shared min would orphan the higher-win-rate model's surplus on the
    # SAME ticker as a fake edge. Per-model subtraction makes full agreement read exactly 0.
    ax = max(0, xw - round(meta["bw_x"]))
    al = max(0, lw - round(meta["bw_l"]))
    subs = {
        "__XW__": xw, "__LW__": lw, "__BW__": bw, "__BL__": bl,
        "__XL__": 100 - xw, "__LL__": 100 - lw,
        "__IWR__": (str(round(100 * bw / (bw + bl))) + "%") if (bw + bl) else "—",
        "__AX__": ax, "__AL__": al, "__AT__": ax + al,
        "__N__": meta["n"], "__TX__": meta["tx"], "__TL__": meta["tl"],
        "__WX__": meta["wx"], "__WL__": meta["wl"],
        "__BWX__": f"{meta['bw_x']:.1f}", "__BWL__": f"{meta['bw_l']:.1f}",
        "__BLX__": f"{meta['bl_x']:.1f}", "__BLL__": f"{meta['bl_l']:.1f}",
        "__WINX__": (windows or {}).get("xgb", "—"),
        "__WINL__": (windows or {}).get("lstm", "—"),
    }
    html = _TEMPLATE
    for k, v in subs.items():
        html = html.replace(k, str(v))
    return html

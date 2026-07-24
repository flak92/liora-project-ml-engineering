#!/usr/bin/env python3
"""Build data_journey.html — the methodology shown as the ROAD THE DATA TRAVELS, reconstructed ENTIRELY
from the committed snapshot (results/methodology_snapshot/compiled/*) and the frozen contract. Nothing
is computed here: the timeline (warmup / Train / OOS + purge/embargo/oos_reads), the pipeline stages
(rungs 0-6), and every per-asset verdict are READ from artifacts. The HTML is a BUILT artifact — never
edit it by hand; edit this generator and run `make data-journey`.

You cannot show the methodology without showing how the data is prepared first: the board opens with the
data-preparation timeline (the FUNDAMENT), then the pipeline stages, then a handful of real assets
travelling those stages to their honest terminal — retained, demoted by their own null, rejected by a
sensitivity null, or an early empty. The seal ties every number to the frozen contract + snapshot.

    python3 scripts/build_data_journey.py             # write data_journey.html
    python3 scripts/build_data_journey.py --emit-seal  # print the DATA-JOURNEY-SEAL line to paste / to diff
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
import contract_loader as CL                                                  # noqa: E402
import contract_patch as CP                                                   # noqa: E402
import report as RE                                                           # noqa: E402

SNAPSHOT = ROOT / "results" / "methodology_snapshot"
OUT = ROOT / "data_journey.html"
MARK = "DATA-JOURNEY-SEAL"

# Seven real assets — one arc through EVERY terminal of the funnel, in narrowing order. All present in
# the committed snapshot; numbers are read from compiled/<T>.json, never typed in.
STORY = ["TSLA", "NVDA", "IDXX", "TDG", "NVR", "GOOG", "ORLY"]

# The pipeline the data travels — rungs 0-6 as stages (rung 2 is folded into 3-4 on this panel).
STAGES = [
    ("0", "OHLCV + freeze", "problem frozen: bars, labels, splits, hashes", "data"),
    ("1", "Viability", "can the model learn at all? (model_viability.py)", "gate"),
    ("3", "Feature search", "does a feature improve a learnable model? — TRAIN-only (feature_utility.py)", "gate"),
    ("4", "Cross-fit", "does the choice survive folds that did not choose it? (crossfit_selection.py)", "gate"),
    ("5", "Max-null", "bigger than the max a search makes by itself? — 3 nulls (procedure_null.py)", "gate"),
    ("6", "Survivor-HPO", "worth more under its own tuned model? (rung6_survivor_hpo.py)", "gate"),
    ("V", "OOS verdict", "one-shot read of the sealed OOS window", "oos"),
]

# How far each terminal travels, and how it ends — derived from status + stop_reason (no hard-coding
# of which ticker: the rule is the same the state machine uses).
def _exit(comp, retained_tickers):
    st = comp.get("status")
    sr = (comp.get("stop_reason") or "").lower()
    if st == "resolved":
        return (6, "retained" if comp["ticker"] in retained_tickers else "demoted")
    if st == "resolved_conditional":
        return (5, "sensitivity")            # passed the primary null, a sensitivity null (a2/b) rejected
    if st == "resolved_empty" and "max-null" in sr:
        return (5, "null_rejected")          # reached the null, no unit exceeded it
    if st == "resolved_empty":
        return (4, "no_feature")             # cross-fit accepted nothing — early empty
    return (1, "unknown")


TERMINAL_LABEL = {
    "retained": ("RETAINED", "keep"),
    "demoted": ("DEMOTED by own null", "stop"),
    "sensitivity": ("REJECTED — sensitivity null", "stop"),
    "null_rejected": ("EMPTY — no unit beat the null", "stop"),
    "no_feature": ("EMPTY — cross-fit found nothing", "stop"),
    "unknown": ("—", "stop"),
}


def reconstruct():
    comp_all = RE.compiled(str(SNAPSHOT))
    fn = RE.funnel(str(SNAPSHOT))
    contract = CL.assemble()
    db = contract.get("data_boundary", {})

    # arm-level funnel (parity with the replay / README)
    arm_funnel = [
        {"n": fn["provisional_crossfit"], "label": "provisional arms (cross-fit accepted)"},
        {"n": fn["passed_a1_marginal"], "label": "passed A1 (marginal max-null)"},
        {"n": fn["stable_a1_a2_b"], "label": "stable A1 ∩ A2 ∩ B"},
        {"n": fn["retained_rung6"], "label": "retained arms (survivor-HPO)"},
    ]
    retained_tickers = {u.split("/")[0] for u in fn["retained_units"]}

    # asset-level funnel — how many assets reach each rung (derived from the null panels)
    def _ntickers(name):
        try:
            return len(json.loads((SNAPSHOT / name).read_text(encoding="utf-8"))["tables"])
        except (OSError, KeyError, json.JSONDecodeError):
            return 0
    asset_funnel = [
        {"n": len(comp_all), "label": "assets on the development panel"},
        {"n": _ntickers("procedure_null_a1.json"), "label": "reached the max-null"},
        {"n": _ntickers("procedure_null_a2.json"), "label": "passed A1 → sensitivity nulls"},
        {"n": len({r["ticker"] for r in json.loads((SNAPSHOT / "rung6_survivor_hpo.json").read_text())["results"]}),
         "label": "stable → survivor-HPO"},
        {"n": len(retained_tickers), "label": "retained after tuning"},
    ]

    assets = []
    for t in STORY:
        c = comp_all.get(t)
        if not c:
            continue
        reached, outcome = _exit(c, retained_tickers)
        mn = c.get("max_null_p") or []
        assets.append({
            "ticker": t,
            "reached": reached, "outcome": outcome,
            "features": c.get("selected_features", []),
            "cwr": c.get("confirmation_win_rate"),
            "outer_delta": c.get("outer_delta"),
            "p_mc": (mn[0].get("p_mc") if mn else None),
            "stop_reason": c.get("stop_reason", ""),
            "accepted": len((c.get("evidence") or {}).get("accepted", [])),
            "rejected": len((c.get("evidence") or {}).get("rejected_by_null", [])),
        })

    sp = json.loads((ROOT / "config" / "xgb.json").read_text(encoding="utf-8"))["splits"]
    op = json.loads((ROOT / "config" / "contract" / "operating_space.json").read_text(encoding="utf-8"))
    return {
        "timeline": {
            "warmup": [sp["warmup_start"], sp["warmup_end"]],
            "train": [sp["train_start"], sp["train_end"]],
            "oos": [sp["oos_start"], sp["oos_end"]],
            "label_horizon_bars": db.get("label_horizon_bars"),
            "embargo_bars": db.get("embargo_bars"),
            "oos_reads": db.get("oos_reads", 0),
        },
        "theta_grid": op.get("grid") or op.get("operating_point", {}).get("grid", []),
        "stages": [{"rung": r, "name": n, "note": note, "kind": k} for r, n, note, k in STAGES],
        "assets": assets,
        "asset_funnel": asset_funnel,
        "arm_funnel": arm_funnel,
        "identity": {
            "contract_version": contract.get("identity", {}).get("contract_version"),
            "oos_reads": db.get("oos_reads", 0),
        },
    }


def seal(data):
    return {
        "contract_hash": CP._hash(CL.assemble())[:16],
        "contract_version": data["identity"]["contract_version"],
        "data_boundary": [data["timeline"]["train"][1], data["timeline"]["oos"][0]],
        "oos_reads": data["timeline"]["oos_reads"],
        "arm_funnel": [s["n"] for s in data["arm_funnel"]],
        "asset_funnel": [s["n"] for s in data["asset_funnel"]],
        "assets_shown": sorted(a["ticker"] for a in data["assets"]),
    }


def _seal_line(s):
    return f"{MARK} {json.dumps(s, sort_keys=True, ensure_ascii=False)} {MARK}"


def main():
    data = reconstruct()
    s = seal(data)
    if "--emit-seal" in sys.argv:
        print(_seal_line(s))
        return 0
    html = (TEMPLATE
            .replace("/*__SEAL__*/", _seal_line(s))
            .replace("'__DATA__'", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
            .replace("'__SEALJSON__'", json.dumps(s, ensure_ascii=False, separators=(",", ":"))))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(html)} B)  contract_hash={s['contract_hash']}  "
          f"asset_funnel={s['asset_funnel']}  arm_funnel={s['arm_funnel']}")
    return 0


TEMPLATE = r"""<title>Golden Calibration — data journey</title>
<!-- /*__SEAL__*/ -->
<style>
  /* Retro Win95 / terminal skin (matched to methodology_replay.html): silver ground, Courier, navy
     outset title bars, white inset panels, green/dark-red status. Status is always a WORD. */
  :root{
    --silver:#c0c0c0; --navy:#000080; --grey:#808080; --white:#fff; --ink:#000;
    --green:#006000; --red:#900; --amber:#8a5a00; --muted:#404040; --faint:#606060;
    --warm:#7a6a00; --train:#000080; --oos:#900;
    --mono:"Courier New",Courier,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--silver);color:var(--ink);font-family:var(--mono);font-size:13px}
  .app{max-width:1180px;margin:0 auto;padding:10px}
  .word{font-weight:bold;letter-spacing:.5px}
  .keep{color:var(--green);font-weight:bold} .stop{color:var(--red);font-weight:bold}
  code{background:#000;color:#0f0;padding:0 4px}
  .mono{font-variant-numeric:tabular-nums}

  .badge{display:flex;align-items:center;gap:14px;padding:7px 12px;background:var(--navy);color:#fff;
    border:3px outset #fff;font-size:12px;flex-wrap:wrap;letter-spacing:.5px;margin-bottom:10px}
  .badge .tag{background:#0000b0;color:#fff;padding:1px 8px;border:2px outset #fff;font-weight:bold}
  .badge b{color:#fff} .badge .sp{flex:1}

  h2{background:var(--grey);color:#fff;padding:4px 10px;margin:14px 0 8px;border:2px outset #fff;
     font-size:13px;letter-spacing:.5px}
  .lead{color:var(--muted);font-size:12.5px;margin:0 0 10px;line-height:1.5;max-width:96ch} .lead b{color:#000}
  .panel{background:#fff;border:2px inset #fff;padding:11px 13px}

  /* timeline */
  .tl{display:flex;height:64px;border:2px inset #fff;background:#fff;overflow:hidden}
  .tl .seg{display:flex;flex-direction:column;justify-content:center;padding:6px 10px;color:#fff;
    border-right:2px solid #fff;position:relative}
  .tl .warm{background:var(--warm);flex:0 0 12%} .tl .train{background:var(--train);flex:1}
  .tl .oos{background:var(--oos);flex:0 0 26%}
  .tl .seg .t{font-weight:bold;font-size:12px;letter-spacing:.5px}
  .tl .seg .d{font-size:10.5px;opacity:.9;margin-top:2px}
  .tl .gap{position:absolute;top:0;bottom:0;width:0;border-left:2px dashed #fff}
  .tl .train .carve{position:absolute;top:0;bottom:0;right:0;width:34%;
    background:repeating-linear-gradient(45deg,rgba(255,255,255,.18) 0 4px,transparent 4px 8px)}
  .tlnote{font-size:11.5px;color:var(--muted);margin-top:6px}
  .tlnote b{color:#000}

  /* pipeline stages */
  .stages{display:flex;gap:0;flex-wrap:wrap;align-items:stretch}
  .stg{background:#fff;border:2px inset #fff;padding:7px 9px;flex:1;min-width:118px;position:relative}
  .stg.data{border-left:4px solid var(--warm)} .stg.gate{border-left:4px solid var(--navy)}
  .stg.oos{border-left:4px solid var(--red)}
  .stg .r{font-weight:bold;font-size:11px;color:var(--faint)}
  .stg .n{font-weight:bold;font-size:12.5px;margin:1px 0 3px}
  .stg .note{font-size:10.5px;color:var(--muted);line-height:1.25}
  .arrow{align-self:center;color:var(--navy);font-weight:bold;padding:0 3px}

  /* tracks + funnel */
  .cols{display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap}
  .tracks{flex:1 1 640px;min-width:0}
  .track{background:#fff;border:2px inset #fff;border-left:4px solid var(--grey);padding:8px 11px;margin-bottom:7px}
  .track.retained{border-left-color:var(--green)} .track.demoted,.track.sensitivity,.track.null_rejected,.track.no_feature{border-left-color:var(--red)}
  .track .hd{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
  .track .tk{font-weight:bold;font-size:13px;min-width:52px}
  .track .verdict{margin-left:auto;font-size:11.5px}
  .pips{display:flex;gap:4px;margin:7px 0 0;flex-wrap:wrap}
  .pip{font-size:10px;padding:1px 6px;background:#e8e8e8;color:var(--faint);border:1px solid #b0b0b0}
  .pip.done{color:var(--green);border-color:#8ab08a;background:#eef6ee}
  .pip.fail{color:var(--red);border-color:#c99;background:#f6eeee;font-weight:bold}
  .pip.pend{color:#a0a0a0}
  .track .nums{font-size:11px;color:var(--muted);margin-top:6px}
  .track .nums b{color:#000}
  .track .sr{font-size:11px;color:var(--faint);margin-top:3px;font-style:italic}

  .funnel{flex:0 0 268px}
  .fbox{background:#fff;border:2px inset #fff;padding:10px 12px;margin-bottom:10px}
  .fbox h3{margin:0 0 8px;font-size:12px;background:var(--navy);color:#fff;padding:3px 8px;border:2px outset #fff}
  .fstep{border:2px inset #fff;padding:5px 9px;margin-bottom:5px}
  .fstep .n{font-size:19px;font-weight:bold;color:var(--navy);line-height:1}
  .fstep .l{font-size:10px;color:var(--muted);margin-top:1px}
  .fstep.final{border-color:#060} .fstep.final .n{color:var(--green)}

  .foot{font-size:11px;color:var(--faint);margin-top:12px;line-height:1.5}
  .chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px}
  .chip{font-size:10px;padding:1px 7px;background:#fff;border:1px solid #b0b0b0;color:var(--muted)}
</style>

<div class="app">
  <div class="badge">
    <span class="tag">DATA JOURNEY</span>
    <b>GOLDEN CALIBRATION</b>
    <span>contract <b id="chash">…</b></span>
    <span>train ≤ <b id="tend">…</b></span>
    <span>oos ≥ <b id="ostart">…</b></span>
    <span class="sp"></span>
    <span>oos_reads = <b id="oreads">0</b></span>
  </div>

  <h2>&#9617; THE DATA — prepared and frozen BEFORE any method</h2>
  <p class="lead">You cannot show the methodology without the data preparation it stands on. Bars are split
    <b>before</b> OOS is ever touched; every inner fold is carved out of Train with a <b>purge</b> and an
    <b>embargo</b> so a label cannot leak across the cut; the OOS window is <b>read exactly once</b>, after
    the rules are frozen.</p>
  <div class="tl" id="tl"></div>
  <div class="tlnote" id="tlnote"></div>

  <h2>&#9617; THE ROAD — the pipeline the data travels (rungs 0→6 → OOS)</h2>
  <p class="lead">Each stage asks one question and can only fail closed. Features are searched
    <b>inside Train only</b>; the operating point θ is chosen on discovery folds and applied to
    confirmation; nothing is confirmed that a permutation null could have produced by itself.</p>
  <div class="stages" id="stages"></div>

  <h2>&#9617; SEVEN ASSETS — one honest arc through every terminal</h2>
  <p class="lead">Real assets from the development panel travelling the road above. Numbers are read from
    <code>results/methodology_snapshot/compiled/&lt;TICKER&gt;.json</code>. Most end <b>empty</b> — that is the
    point: the method returns an honest empty set where the evidence does not support a feature.</p>
  <div class="cols">
    <div class="tracks" id="tracks"></div>
    <div class="funnel">
      <div class="fbox"><h3>&#9617; ASSET FUNNEL</h3><div id="afun"></div></div>
      <div class="fbox"><h3>&#9617; ARM FUNNEL</h3><div id="rfun"></div></div>
    </div>
  </div>

  <div class="foot" id="foot"></div>
</div>

<script>
const DATA = '__DATA__';
const SEAL = '__SEALJSON__';
const TL = DATA.timeline;
document.getElementById('chash').textContent = SEAL.contract_hash;
document.getElementById('tend').textContent = TL.train[1];
document.getElementById('ostart').textContent = TL.oos[0];
document.getElementById('oreads').textContent = TL.oos_reads;

// ---- timeline ----
const tl = document.getElementById('tl');
tl.innerHTML =
  `<div class="seg warm"><span class="t">WARMUP</span><span class="d">${TL.warmup[0]} … ${TL.warmup[1]}<br>indicators only — never labeled</span></div>`
+ `<div class="seg train"><span class="t">TRAIN</span><span class="d">${TL.train[0]} … ${TL.train[1]}<br>folds · feature search · HPO · null — all here</span><div class="carve" title="purged/embargoed fold boundaries"></div></div>`
+ `<div class="seg oos"><span class="t">OOS — SEALED</span><span class="d">${TL.oos[0]} … ${TL.oos[1]}<br>read once, after freeze</span></div>`;
document.getElementById('tlnote').innerHTML =
  `Fold boundary hygiene: <b>purge = ${TL.label_horizon_bars} bars</b> (label horizon) · <b>embargo = ${TL.embargo_bars} bars</b> · `
+ `<b>oos_reads = ${TL.oos_reads}</b> during all discovery/selection/confirmation/null. θ-grid chosen on discovery: `
+ `<code>[${DATA.theta_grid.join(', ')}]</code>.`;

// ---- pipeline stages ----
const stg = document.getElementById('stages');
DATA.stages.forEach((s,i)=>{
  const d=document.createElement('div'); d.className='stg '+s.kind;
  d.innerHTML=`<div class="r">RUNG ${s.rung}</div><div class="n">${s.name}</div><div class="note">${s.note}</div>`;
  stg.appendChild(d);
  if(i<DATA.stages.length-1){const a=document.createElement('div');a.className='arrow';a.textContent='▸';stg.appendChild(a);}
});

// ---- per-asset tracks ----
const RUNGS=['1','3','4','5','6','V'];                     // pip columns = the gates + verdict
const RLABEL={'1':'viability','3':'feature','4':'cross-fit','5':'null','6':'survivor','V':'OOS'};
const VERD={retained:['RETAINED — kept','keep'],demoted:['DEMOTED by own null','stop'],
  sensitivity:['REJECTED — sensitivity null','stop'],null_rejected:['EMPTY — no unit beat the null','stop'],
  no_feature:['EMPTY — cross-fit found nothing','stop'],unknown:['—','stop']};
const fmt=(x)=> (x===null||x===undefined)?'—':(typeof x==='number'?(Math.round(x*1000)/1000):x);
const tracks=document.getElementById('tracks');
DATA.assets.forEach(a=>{
  const reached=String(a.reached);                        // '4','5','6' (rung index reached)
  const pips=RUNGS.map(r=>{
    let cls='pend';
    if(r==='V'){ cls = a.outcome==='retained'?'done':'pend'; }
    else if(parseInt(r) < parseInt(reached)) cls='done';
    else if(parseInt(r)===parseInt(reached)) cls = a.outcome==='retained'?'done':'fail';
    else cls='pend';
    const mark = cls==='done'?'✓':(cls==='fail'?'✕':'·');
    return `<span class="pip ${cls}">R${r} ${RLABEL[r]} ${mark}</span>`;
  }).join('');
  const [vlabel,vcls]=VERD[a.outcome]||VERD.unknown;
  const feats=(a.features&&a.features.length)?a.features.join(' + '):'—';
  const nums = (a.features&&a.features.length)
    ? `features <b>${feats}</b> · win-rate <b>${fmt(a.cwr)}</b> · outer Δ <b>${fmt(a.outer_delta)}</b> · null p <b>${fmt(a.p_mc)}</b>`
    : `no feature survived selection`;
  const d=document.createElement('div'); d.className='track '+a.outcome;
  d.innerHTML=`<div class="hd"><span class="tk">${a.ticker}</span>`
    +`<span class="verdict word ${vcls}">${vlabel}</span></div>`
    +`<div class="pips">${pips}</div>`
    +`<div class="nums">${nums}</div>`
    +`<div class="sr">${a.stop_reason}</div>`;
  tracks.appendChild(d);
});

// ---- funnels ----
function funnelInto(el,steps,finalLabel){
  steps.forEach((s,i)=>{const d=document.createElement('div');
    d.className='fstep'+(i===steps.length-1?' final':'');
    d.innerHTML=`<div class="n">${s.n}</div><div class="l">${s.label}</div>`;el.appendChild(d);});
}
funnelInto(document.getElementById('afun'),DATA.asset_funnel);
funnelInto(document.getElementById('rfun'),DATA.arm_funnel);

// ---- footer ----
document.getElementById('foot').innerHTML =
  `Every number recomputed from the frozen contract + <code>results/methodology_snapshot/</code> by `
+ `<code>make verify-data-journey</code>; if the contract or snapshot drifts and this board is not rebuilt, `
+ `the seal goes red. Rungs 7–9 (interactions · cross-asset · certification) are <b>SPECIFIED / NOT STARTED</b> — `
+ `this is a development panel, not a certification.`
+ `<div class="chips"><span class="chip">${SEAL.contract_version}</span>`
+ `<span class="chip">contract ${SEAL.contract_hash}</span>`
+ `<span class="chip">assets ${SEAL.assets_shown.join(', ')}</span></div>`;
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())

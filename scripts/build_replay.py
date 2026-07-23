#!/usr/bin/env python3
"""Build methodology_replay.html — a time-compressed REPLAY of the real run, reconstructed ENTIRELY
from the committed snapshot panels. Nothing is computed here: every event (the funnel, the rung
transitions, the max-null b-counter per permutation, the futility-stops, the verdicts, the terminal
states) is read from results/methodology_snapshot/*. The HTML is a BUILT artifact — never edit it by
hand; edit this generator and run `make replay`.

Per-permutation wall-time is NOT in the panels (only per-UNIT `seconds`, e.g. ORLY a1 4245 s), so the
animation paces each unit by its real per-unit seconds x a compression factor and spreads that across
the unit's permutations — a SYNTHETIC per-permutation cadence, labelled on screen; the counts stay
exact. Decision events (rung transitions, verdicts, futility-stops, terminals, funnel ticks) are kept
whole; only the permutation STREAM is thinned for the eye.

    python3 scripts/build_replay.py             # write methodology_replay.html
    python3 scripts/build_replay.py --emit-seal  # print the REPLAY-SEAL line to paste / to diff
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
OUT = ROOT / "methodology_replay.html"
MARK = "REPLAY-SEAL"

# The three assets that carry the arc: one retained, two honest failures (positive tuned delta, yet
# demoted by their own null). All present in every committed panel.
STORY = [("ORLY", 1), ("GOOG", 1), ("NVR", 0)]

LADDER = [
    (0, "Is the problem frozen?", "EXECUTED", "frozen"),
    (1, "Can the model learn at all?", "EXECUTED", "frozen"),
    (2, "Does the operating point transfer?", "EXECUTED", "adm"),
    (3, "Does a feature improve a learnable model?", "EXECUTED", "adm"),
    (4, "Does the choice survive data that didn't choose it?", "EXECUTED", "adm"),
    (5, "Bigger than the max a search makes by itself?", "EXECUTED", "frozen"),
    (6, "Worth more under its own tuned model?", "EXECUTED", "adm"),
    (7, "Do survivors combine?", "SPECIFIED", "spec"),
    (8, "Which OHLCV families travel across assets?", "EXECUTED", "adm"),
    (9, "Does the whole method hold on a fresh panel?", "CERT", "cert"),
]


def _null_arm(panel, ticker, fold, arm):
    for f in panel["tables"].get(ticker, {}).get("folds", []):
        if f["outer_fold"] == fold:
            return f["arms"].get(arm)
    return None


def guard_1b():
    """The truly-live element: the guard's actual verdict on a patch that would loosen the headline
    null M=50 -> 5. Captured at build time as the standalone fallback; the Streamlit page re-runs it
    live and verify_replay.py fails if this embedded message ever drifts from the guard."""
    patch = {"rung_6_survivor_hpo": {"own_null": {"permutations": 5}}}
    try:
        CP.guard(patch)
    except CP.PatchRejected as e:
        return {"patch": patch, "rejected": True, "message": str(e)}
    return {"patch": patch, "rejected": False, "message": "GUARD NIE ODRZUCIŁ — REGRESJA"}


def reconstruct():
    a1 = json.loads((SNAPSHOT / "procedure_null_a1.json").read_text(encoding="utf-8"))
    r6 = json.loads((SNAPSHOT / "rung6_survivor_hpo.json").read_text(encoding="utf-8"))
    fn = RE.funnel(str(SNAPSHOT))
    contract = CL.assemble()

    r6_by = {(r["ticker"], r["outer_fold"], r["arm"]): r for r in r6["results"]}
    assets = {}
    for ticker, fold in STORY:
        rung5 = {}
        for arm in ("flat", "hierarchical"):
            a = _null_arm(a1, ticker, fold, arm)
            if a:
                rung5[arm] = {"unit": a["unit"], "real": round(a["real_statistic"], 6),
                              "null_stats": [round(x, 6) for x in a["null_statistics"]],
                              "b": a["exceedances"], "M": a["permutations_executed"], "verdict": a["verdict"]}
        rung6 = {}
        for arm in ("flat", "hierarchical"):
            r = r6_by.get((ticker, fold, arm))
            if r:
                rung6[arm] = {"unit": r["unit"], "rep": r["representative"],
                              "delta": round(r["tuned_delta"], 6), "b": r["exceedances"],
                              "M": r["permutations"], "null_deltas": [round(x, 6) for x in r["null_deltas"]],
                              "verdict": r["verdict"], "seconds": r.get("seconds")}
        assets[ticker] = {"fold": fold, "seconds": a1["tables"][ticker]["seconds"],
                          "rung5": rung5, "rung6": rung6,
                          "retained": any(v["verdict"] == "retained" for v in rung6.values())}

    frozen = sorted(CP.FROZEN)
    return {
        "funnel": {"steps": [
            {"n": fn["provisional_crossfit"], "label": "provisional arms (cross-fit accepted)"},
            {"n": fn["passed_a1_marginal"], "label": "passed A1 (marginal max-null)"},
            {"n": fn["stable_a1_a2_b"], "label": "stable A1 ∩ A2 ∩ B"},
            {"n": fn["retained_rung6"], "label": "retained arms (survivor HPO)"},
        ], "feature": sorted({r["representative"] for r in r6["results"] if r["verdict"] == "retained"})[0]},
        "ladder": [{"rung": r, "q": q, "status": s, "cls": c} for r, q, s, c in LADDER],
        "frozen": frozen,
        "assets": assets,
        "null_params": {"M": r6["permutations"], "alpha": r6["alpha"], "pass_b": r6["pass_b"],
                        "futility_b": r6["futility_b"]},
        "identity": {"contract_version": contract.get("identity", {}).get("contract_version"),
                     "data_boundary": [contract.get("data_boundary", {}).get("train_end"),
                                       contract.get("data_boundary", {}).get("oos_start")],
                     "seed": contract.get("runtime", {}).get("seed", 42),
                     "oos_reads": contract.get("data_boundary", {}).get("oos_reads", 0)},
        "guard": guard_1b(),
    }


def seal(data):
    fn = data["funnel"]
    return {
        "contract_hash": CP._hash(CL.assemble())[:16],
        "contract_version": data["identity"]["contract_version"],
        "funnel": [s["n"] for s in fn["steps"]],
        "feature": fn["feature"],
        "guard_rejected": data["guard"]["rejected"],
        "guard_message": data["guard"]["message"],
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
    print(f"wrote {OUT}  ({len(html)} B)  contract_hash={s['contract_hash']}  funnel={s['funnel']}")
    return 0


TEMPLATE = r"""<title>Golden Calibration — methodology replay</title>
<!-- /*__SEAL__*/ -->
<style>
  /* Retro Win95 / terminal skin (per dashboard.html): silver ground, Courier, navy outset title bars,
     white inset panels, green/dark-red status. Status is always a WORD; colour only reinforces. */
  :root{
    --silver:#c0c0c0; --navy:#000080; --grey:#808080; --white:#fff; --ink:#000;
    --green:#006000; --red:#900; --muted:#404040; --faint:#606060;
    --accent:#000080; --amber:#900;   /* FROZEN reads red-locked; ADMISSIBLE green */
    --mono:"Courier New",Courier,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;overflow:hidden;background:var(--silver);color:var(--ink);font-family:var(--mono);font-size:13px}
  .app{display:flex;flex-direction:column;height:100vh;max-height:900px}
  a{color:var(--navy)}
  code{background:#000;color:#0f0;padding:0 4px}
  .mono{font-variant-numeric:tabular-nums}
  .word{font-weight:bold;letter-spacing:.5px}
  .w-adm{color:var(--green)} .w-frz{color:var(--red)} .w-spec{color:var(--navy)} .w-red{color:var(--red)}

  /* badge (Z1) — navy title bar, raised bevel */
  .badge{display:flex;align-items:center;gap:14px;padding:6px 12px;background:var(--navy);color:#fff;
    border:3px outset #fff;font-size:12px;flex:0 0 auto;flex-wrap:wrap;letter-spacing:.5px}
  .badge .live{background:var(--red);color:#fff;padding:1px 7px;border:2px outset #fff;font-weight:bold}
  .badge .rep{background:#0000b0;color:#fff;padding:1px 7px;border:2px outset #fff;font-weight:bold}
  .badge b{color:#fff} .badge .sp{flex:1} .badge .blk{letter-spacing:0}

  /* columns */
  .cols{display:flex;gap:8px;flex:1;min-height:0;padding:8px;background:var(--silver)}
  .col{display:flex;flex-direction:column;min-height:0;overflow:hidden}
  .col1{width:252px;flex:0 0 auto} .col3{width:202px;flex:0 0 auto} .scene{flex:1 1 auto}
  .col h2{background:var(--grey);color:#fff;padding:3px 8px;margin:0 0 6px;border:2px outset #fff;font-size:12px;letter-spacing:.5px}
  .panel{background:#fff;border:2px inset #fff;padding:9px 11px;overflow:auto;flex:1;min-height:0}

  /* ladder rows */
  .rung{display:flex;gap:6px;align-items:baseline;padding:3px 3px;border-bottom:1px solid #d4d4d4;opacity:.5}
  .rung.on{opacity:1;background:#eef0f4}
  .rung .id{font-weight:bold;width:24px;color:var(--muted)}
  .rung .q{font-size:11px;color:var(--muted);flex:1;line-height:1.15}
  .rung.on .q{color:#000}
  .rung .st{margin-left:auto;font-size:10px;font-weight:bold}

  /* scene */
  .act-title{font-size:12px;font-weight:bold;color:var(--navy);letter-spacing:.5px}
  .act-n{font-size:19px;font-weight:bold;margin:2px 0 12px;color:#000}
  .lead{color:var(--muted);font-size:12.5px;max-width:60ch;margin:0 0 12px;line-height:1.45} .lead b{color:#000}

  /* funnel */
  .fstep{background:#fff;border:2px inset #fff;padding:6px 9px;margin-bottom:6px;opacity:.4}
  .fstep.on{opacity:1}
  .fstep .n{font-size:20px;font-weight:bold;color:var(--navy);line-height:1}
  .fstep .l{font-size:10px;color:var(--muted);margin-top:2px}
  .fstep.final{border:2px inset #060} .fstep.final .n{color:var(--green)}

  /* asset tracks */
  .tracks{display:flex;flex-direction:column;gap:8px}
  .track{background:#fff;border:2px inset #fff;padding:8px 11px;border-left:4px solid var(--grey)}
  .track.retained{border-left-color:var(--green)} .track.demoted{border-left-color:var(--red)}
  .track .hd{display:flex;align-items:baseline;gap:10px}
  .track .tk{font-weight:bold}
  .track .steps{display:flex;gap:4px;margin:7px 0 0;flex-wrap:wrap}
  .pip{font-size:10px;padding:1px 6px;background:#e8e8e8;color:var(--faint);border:1px solid #b0b0b0}
  .pip.done{color:var(--green);border-color:#8ab08a;background:#eef6ee}
  .pip.fail{color:var(--red);border-color:#c99;background:#f6eeee}
  .track .verdict{margin-top:6px;font-size:12px}

  /* null histograms */
  .nulls{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
  .nullcard{background:#fff;border:2px inset #fff;padding:9px 10px}
  .nullcard .tk{font-weight:bold;font-size:12px}
  .nullcard .sub{font-size:10.5px;color:var(--faint);margin-bottom:7px}
  .hist{position:relative;height:92px;display:flex;align-items:flex-end;gap:1px;background:#f4f4f4;border:1px solid #b0b0b0;padding:0}
  .hist .bar{flex:1;background:var(--navy);min-height:1px}
  .hist .bar.exceed{background:var(--red)}
  .hist .real{position:absolute;left:0;right:0;border-top:1px dashed var(--navy);pointer-events:none}
  .hist .real span{position:absolute;right:1px;top:-13px;font-size:9px;color:var(--navy);background:#f4f4f4;padding:0 2px}
  .bcount{margin-top:6px;font-size:12px}
  .bcount b{font-size:15px}
  .stop{color:var(--red);font-weight:bold} .keep{color:var(--green);font-weight:bold}

  /* seal kv, chips, guard, terminals, limits */
  .kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font-size:12px;margin:2px 0 12px}
  .kv .k{color:var(--faint)} .kv .v{color:#000}
  .chips{display:flex;flex-wrap:wrap;gap:5px}
  .chip{font-size:10px;padding:1px 7px;background:#fff;border:1px solid #c99;color:var(--red);font-weight:bold}
  .guardbox{background:#fff;border:2px inset #fff;border-left:4px solid var(--red);padding:11px 13px;margin-top:6px}
  .guardbox .patch{font-size:12px;color:#000}
  .guardbox .msg{font-size:11.5px;color:var(--red);margin-top:8px;line-height:1.4;font-weight:bold}
  .terms{display:grid;grid-template-columns:1fr 1fr;gap:9px}
  .term{background:#fff;border:2px inset #fff;padding:10px 12px;border-left:4px solid var(--grey)}
  .term.machine{border-left-color:var(--green)} .term.human{border-left-color:var(--red)}
  .term .s{font-weight:bold;font-size:12.5px}
  .term p{margin:5px 0 0;font-size:12px;color:var(--muted)}
  ul.limits{margin:6px 0 0;padding-left:18px;color:var(--muted);font-size:12.5px} ul.limits li{margin:5px 0}
  ul.limits b{color:#000}

  .num{cursor:pointer;color:var(--navy);border-bottom:1px dotted var(--navy)} .num:hover{background:#000080;color:#fff}

  /* ticker (Z2/Z3) — inset scroll, the only page scroll */
  .ticker{flex:0 0 auto;height:92px;margin:0 8px 8px;background:#fff;border:2px inset #fff;padding:5px 9px;overflow-y:auto;font-size:11px;color:var(--muted);line-height:1.5}
  .ticker .row{white-space:nowrap} .ticker .row b{color:#000}
  .ticker .g{color:var(--green);font-weight:bold} .ticker .a{color:var(--red);font-weight:bold}
  .dslabel{color:var(--faint);font-style:italic;font-size:11px}

  /* controls — Win95 raised buttons */
  .ctrl{flex:0 0 auto;display:flex;align-items:center;gap:6px;padding:6px 10px;background:var(--silver);border-top:2px groove #fff;flex-wrap:wrap}
  .ctrl button{font-family:var(--mono);font-size:11px;background:var(--silver);color:#000;border:2px outset #fff;padding:4px 9px;cursor:pointer}
  .ctrl button:active{border-style:inset} .ctrl button:focus-visible{outline:1px dotted #000;outline-offset:-4px}
  .ctrl .jump{display:flex;gap:3px;flex-wrap:wrap}
  .ctrl .jump button.cur{border-style:inset;background:#dcdcff;color:var(--navy);font-weight:bold}
  .ctrl .sp{flex:1} .ctrl .spd{color:#000;font-size:11px;min-width:38px;text-align:center}

  /* modal */
  .modal{position:fixed;inset:0;background:rgba(0,0,64,.35);display:none;align-items:center;justify-content:center;z-index:50}
  .modal.open{display:flex}
  .modal .card{background:var(--silver);border:3px outset #fff;max-width:660px;max-height:78vh;overflow:auto;padding:6px}
  .modal h3{background:var(--navy);color:#fff;margin:0;padding:5px 9px;border:2px outset #fff;font-size:13px}
  .modal .src{font-size:11px;color:var(--muted);margin:6px 4px}
  .modal pre{background:#000;color:#0f0;border:2px inset #fff;padding:11px;overflow:auto;font-size:11.5px;margin:0 2px 4px}
  .modal .x{float:right;cursor:pointer;color:#fff;font-weight:bold}
</style>

<div class="app">
  <div class="badge" id="badge">
    <span class="blk">&#9619;&#9618;&#9617;</span>
    <span class="rep" id="mode">REPLAY</span>
    <b>METHODOLOGY REPLAY</b>
    <span>run <b id="run">methodology_snapshot</b></span>
    <span>contract <b id="chash">…</b></span>
    <span class="sp"></span>
    <span id="clock">00:00</span>
    <span>&middot; <b id="factor">&times;1</b></span>
    <span class="blk">&#9617;&#9618;&#9619;</span>
  </div>

  <div class="cols">
    <div class="col col1">
      <h2>&#9617; THE LADDER &middot; RUNG 0-9</h2>
      <div class="panel" id="ladder"></div>
    </div>
    <div class="col scene" id="scene"></div>
    <div class="col col3">
      <h2>&#9617; FUNNEL</h2>
      <div class="panel" id="funnel"></div>
    </div>
  </div>

  <div class="ticker" id="ticker"></div>

  <div class="ctrl">
    <button id="play">▮▮ pause</button>
    <button id="step">▸ step</button>
    <button id="reset">⟲ reset</button>
    <div class="jump" id="jump"></div>
    <span class="sp"></span>
    <button id="slower">−</button><span class="spd" id="spd">1.0×</span><button id="faster">+</button>
  </div>
</div>

<div class="modal" id="modal"><div class="card">
  <span class="x" id="mx">✕ close</span><h3 id="mt"></h3><div class="src" id="ms"></div><pre id="mp"></pre>
</div></div>

<script>
const DATA = '__DATA__';
const SEAL = '__SEALJSON__';
document.getElementById('chash').textContent = SEAL.contract_hash;

// ---------- build the timed event stream from DATA (deterministic; counts exact) ----------
// Replay-seconds budget per act (compressed). Real per-unit seconds set the compression FACTOR badge.
const ACTS = [
  {id:'0',  t:'Ending, up front',        dur:30},
  {id:'1',  t:'The seal',                dur:60},
  {id:'1b', t:'The guard — truly live',  dur:30, live:true},
  {id:'2',  t:'Three assets, same procedure', dur:180},
  {id:'3',  t:'The null',                dur:180},
  {id:'4',  t:'Honest failures',         dur:90},
  {id:'5',  t:'The boundary of autonomy',dur:60},
  {id:'6',  t:'What this is not',        dur:45},
];
let AT0={}, tacc=0; ACTS.forEach(a=>{a.start=tacc; AT0[a.id]=tacc; tacc+=a.dur;}); const TOTAL=tacc;
const realSecs = Object.values(DATA.assets).reduce((s,a)=>s+(a.seconds||0),0)*1; // sum of story per-unit seconds (a1)
document.getElementById('factor').textContent = '×'+Math.max(1,Math.round(realSecs/TOTAL));

const EV=[]; // {t, act, fn}
const push=(t,act,fn)=>EV.push({t,act,fn});
const order=['ORLY','GOOG','NVR'];

// downsample a per-permutation stream into <=STEPS animation ticks, keeping exact final counts
const STEPS=26;
function stream(vals){ const n=vals.length; if(n<=STEPS) return vals.map((v,i)=>({i,v})); const out=[];
  for(let s=0;s<STEPS;s++){const i=Math.min(n-1,Math.round(s*(n-1)/(STEPS-1))); out.push({i,v:vals[i]});} return out; }

// scene renderers registered per act; the loop calls render(act, localProgress)
const scene=document.getElementById('scene');
const ticker=document.getElementById('ticker');
function tick(html){const d=document.createElement('div');d.className='row';d.innerHTML=html;ticker.appendChild(d);ticker.scrollTop=ticker.scrollHeight; while(ticker.children.length>140)ticker.removeChild(ticker.firstChild);}

// artifact modal
const SRC={
  funnel:{title:'report.funnel(results/methodology_snapshot)',src:'engine/report.py',body:()=>JSON.stringify({funnel:SEAL.funnel,retained_feature:DATA.funnel.feature},null,1)},
};
Object.entries(DATA.assets).forEach(([tk,a])=>{
  SRC['r5_'+tk]={title:tk+' · Rung 5 max-null (A1)',src:'results/methodology_snapshot/procedure_null_a1.json',body:()=>JSON.stringify(a.rung5,null,1)};
  SRC['r6_'+tk]={title:tk+' · Rung 6 survivor-HPO own-null',src:'results/methodology_snapshot/rung6_survivor_hpo.json',body:()=>JSON.stringify(a.rung6,null,1)};
});
SRC['guard']={title:'contract_patch.guard — live rejection',src:'engine/contract_patch.py',body:()=>JSON.stringify(DATA.guard,null,1)};
function openModal(key){const s=SRC[key]; if(!s)return; document.getElementById('mt').textContent=s.title;
  document.getElementById('ms').textContent='source · '+s.src; document.getElementById('mp').textContent=s.body();
  document.getElementById('modal').classList.add('open');}
document.getElementById('mx').onclick=()=>document.getElementById('modal').classList.remove('open');
document.getElementById('modal').onclick=e=>{if(e.target.id==='modal')e.currentTarget.classList.remove('open');};
const N=(v,key)=>`<span class="num" data-src="${key}">${v}</span>`;

// ---------- ladder (left) ----------
const L=document.getElementById('ladder');
DATA.ladder.forEach(r=>{const d=document.createElement('div');d.className='rung';d.id='rung'+r.rung;
  const w=r.cls==='frozen'?'w-frz':r.cls==='spec'?'w-spec':r.cls==='cert'?'w-frz':'w-adm';
  d.innerHTML=`<span class="id">R${r.rung}</span><span class="q">${r.q}</span><span class="st word ${w}">${r.status}</span>`;
  L.appendChild(d);});
function ladderOn(max){DATA.ladder.forEach(r=>document.getElementById('rung'+r.rung).classList.toggle('on', r.rung<=max));}

// ---------- funnel (right) ----------
const F=document.getElementById('funnel'); const fsteps=DATA.funnel.steps;
fsteps.forEach((s,i)=>{const d=document.createElement('div');d.className='fstep';d.id='f'+i;
  d.innerHTML=`<div class="n">${N(s.n,'funnel')}</div><div class="l">${s.label}</div>`;F.appendChild(d);});
const ffeat=document.createElement('div');ffeat.className='fstep final';ffeat.id='ffeat';
ffeat.innerHTML=`<div class="n">1</div><div class="l">unique feature (rep ${DATA.funnel.feature})</div>`;F.appendChild(ffeat);
function funnelOn(k){fsteps.forEach((s,i)=>document.getElementById('f'+i).classList.toggle('on',i<k)); document.getElementById('ffeat').classList.toggle('on',k>fsteps.length);}

// ---------- per-act scene painters (idempotent; driven by localProgress p in [0,1]) ----------
function actScaffold(a){return `<div class="act-title">Act ${a.id} · ${a.live?'<span class="word w-red">LIVE</span>':'<span class="word w-spec">REPLAY</span>'}</div><div class="act-n">${a.t}</div>`;}

function paint(a,p){
  if(scene.dataset.act!==a.id){scene.dataset.act=a.id; scene.innerHTML='<h2>&#9617; THE SEQUENCE</h2><div class="panel">'+actScaffold(a)+'<div id="body"></div></div>';}
  const body=document.getElementById('body'); const feat=DATA.funnel.feature; const P=DATA.null_params;
  if(a.id==='0'){ funnelOn(99); ladderOn(9);
    body.innerHTML=`<p class="lead">The funnel is a property of the data, computed from artifacts — not a target. On the 20-asset development panel:</p>
      <div style="font-family:var(--mono);font-size:19px;line-height:1.7">${fsteps.map(s=>N(s.n,'funnel')).join(' <span style="color:var(--faint)">→</span> ')} <span style="color:var(--faint)">→</span> <span class="w-adm">1 unique feature</span></div>
      <p class="lead" style="margin-top:12px"><b>2 retained arms → 1 feature.</b> Both <code>ORLY/1</code> arms (<code>flat 112</code> and <code>hierarchical oscillator_rsi</code>) resolve to representative <b>${feat}</b> — two selection paths, one feature. <b>Certification: NOT STARTED.</b></p>`;
  } else if(a.id==='1'){ funnelOn(0); ladderOn(0);
    body.innerHTML=`<p class="lead">Fixed <b>before</b> the data was read — the proof standard, hashed and frozen.</p>
      <div class="kv"><span class="k">contract</span><span class="v">${SEAL.contract_hash} · ${DATA.identity.contract_version}</span>
      <span class="k">train / oos</span><span class="v">≤ ${DATA.identity.data_boundary[0]} · ≥ ${DATA.identity.data_boundary[1]}</span>
      <span class="k">oos_reads</span><span class="v">${DATA.identity.oos_reads} — sealed</span>
      <span class="k">seed</span><span class="v">${DATA.identity.seed}</span></div>
      <div style="font-size:11px;color:var(--faint);font-family:var(--mono);margin-bottom:6px">FROZEN proof standard — untouchable by the loop</div>
      <div class="chips">${DATA.frozen.map(f=>`<span class="chip">${f}</span>`).join('')}</div>`;
  } else if(a.id==='1b'){ ladderOn(6);
    body.innerHTML=`<p class="lead">One element is <b>not</b> a replay — it runs for real, and costs nothing. A ladder patch tries to loosen the headline null from M=50 to M=5:</p>
      <div class="guardbox"><div class="patch">guard( ${N(JSON.stringify(DATA.guard.patch),'guard')} )</div>
      <div class="msg">→ ${DATA.guard.rejected?'PatchRejected':'NOT REJECTED — regression'}<br>${DATA.guard.message}</div></div>
      <p class="lead" style="margin-top:12px">Field-level: <code>rung_6_survivor_hpo</code> is admissible, yet <code>own_null.permutations</code> is a frozen leaf. The loop cannot loosen its own proof standard.</p>`;
  } else if(a.id==='2'){ const k=Math.floor(p*4); funnelOn(Math.min(3,1+Math.floor(p*3))); ladderOn(Math.min(6,3+Math.floor(p*4)));
    body.innerHTML=`<p class="lead">The same procedure walks each asset 1→3→4→5→6. <b>Divergent fates.</b></p><div class="tracks" id="tr"></div>`;
    const tr=document.getElementById('tr');
    order.forEach(tk=>{const a2=DATA.assets[tk]; const cls=a2.retained?'retained':'demoted';
      const pips=['R1','R3','R4','R5'].map((r,ri)=>`<span class="pip ${p*4>ri?'done':''}">${r} ✓</span>`).join('');
      tr.innerHTML+=`<div class="track ${cls}"><div class="hd"><span class="tk">${tk}/${a2.fold}</span><span class="word ${a2.retained?'w-adm':'w-frz'}">${a2.retained?'heads to retained':'heads to demoted'}</span></div><div class="steps">${pips}</div></div>`;});
  } else if(a.id==='3'){ funnelOn(3); ladderOn(5);
    body.innerHTML=`<p class="lead">Rung 5 · the block-permutation max-null — <b>91.6% of the compute</b>. Each permutation shuffles the feature and re-scores; the real statistic (dashed line) must beat the pile. Counter <code>b</code> = permutations that reached it.</p>
      <div class="nulls" id="nl"></div><p class="dslabel" style="margin-top:10px">permutation stream thinned ×N for the eye; b-counter and per-unit seconds are exact from the artifacts</p>`;
    const nl=document.getElementById('nl');
    order.forEach(tk=>{const arm=DATA.assets[tk].rung5.flat; if(!arm)return;
      const st=stream(arm.null_stats); const shown=Math.max(1,Math.ceil(st.length*p)); const sub=st.slice(0,shown);
      const mx=Math.max(arm.real,...arm.null_stats,1e-9);
      const b=sub.filter(x=>x.v>=arm.real).length; const realPct=100-arm.real/mx*100;
      const bars=sub.map(x=>`<div class="bar ${x.v>=arm.real?'exceed':''}" style="height:${Math.max(2,x.v/mx*100)}%"></div>`).join('');
      nl.innerHTML+=`<div class="nullcard"><div class="tk">${tk}/${DATA.assets[tk].fold} ${N('flat '+arm.unit,'r5_'+tk)}</div><div class="sub">real T=${arm.real.toFixed(4)}</div>
        <div class="hist">${bars}<div class="real" style="top:${realPct}%"><span>real</span></div></div>
        <div class="bcount">b = <b>${p>=1?arm.b:b}</b> / ${arm.M} <span class="keep">${p>=1?'passed':''}</span></div></div>`;});
  } else if(a.id==='4'){ funnelOn(4); ladderOn(6);
    body.innerHTML=`<p class="lead">Rung 6 · each survivor re-tuned against its <b>own</b> null. Here the fates split.</p><div class="nulls" id="n6"></div>`;
    const n6=document.getElementById('n6');
    order.forEach(tk=>{const r=DATA.assets[tk].rung6.flat; if(!r)return; const st=stream(r.null_deltas);
      const shown=Math.max(1,Math.ceil(st.length*p)); const sub=st.slice(0,shown);
      const mx=Math.max(Math.abs(r.delta),...r.null_deltas.map(Math.abs),1e-9);
      const exc=sub.filter(x=>x.v>=r.delta).length;
      const bars=sub.map(x=>`<div class="bar ${x.v>=r.delta?'exceed':''}" style="height:${Math.max(2,Math.abs(x.v)/mx*100)}%"></div>`).join('');
      const ret=r.verdict==='retained'; const saved=P.M-r.M;
      const line = ret ? `b = <b>${r.b}</b> / ${r.M} <span class="keep">retained</span>`
        : `b = <b class="stop">${r.b}</b> · <span class="stop">FUTILITY-STOP @${r.M}</span> · saved ${saved} perms`;
      n6.innerHTML+=`<div class="nullcard" style="border-left:3px solid ${ret?'var(--green)':'var(--amber)'}"><div class="tk">${tk}/${DATA.assets[tk].fold} ${N('rep '+r.rep,'r6_'+tk)}</div>
        <div class="sub">tuned Δ=<span style="color:${r.delta>=0?'var(--green)':'var(--red)'}">${r.delta>=0?'+':''}${r.delta.toFixed(4)}</span>${r.delta>0&&!ret?' · positive, yet demoted':''}</div>
        <div class="hist">${bars}<div class="real" style="top:${Math.max(0,100-Math.abs(r.delta)/mx*100)}%"><span>Δ</span></div></div>
        <div class="bcount">${line}</div>${!ret?'<div class="verdict word w-frz">RESOLVED_EMPTY — a result, not a failure</div>':''}</div>`;});
    funnelOn(99);
  } else if(a.id==='5'){ ladderOn(9); funnelOn(99);
    body.innerHTML=`<p class="lead">Three terminals are the machine finishing honestly. The fourth is the only place a <b>human</b> acts — the boundary of autonomy.</p>
      <div class="terms">
        <div class="term machine"><div class="s">RESOLVED_RETAINED</div><span class="word w-adm">MACHINE</span><p>a stable survivor was retained.</p></div>
        <div class="term machine"><div class="s">RESOLVED_EMPTY</div><span class="word w-adm">MACHINE</span><p>the evidence honestly supports no feature — a result, not a failure.</p></div>
        <div class="term machine"><div class="s">LADDER_EXHAUSTED</div><span class="word w-adm">MACHINE</span><p>every pre-authorized version walked, none added a feature.</p></div>
        <div class="term human"><div class="s">NEEDS_CONTRACT</div><span class="word w-frz">HUMAN</span><p>cannot proceed honestly. A person authors the next version — ADMISSIBLE only. Auto-widen forbidden (Act 1b).</p></div>
      </div>`;
  } else if(a.id==='6'){ ladderOn(9); funnelOn(99);
    body.innerHTML=`<p class="lead">Spoken plainly, no upgrade:</p>
      <ul class="limits">
        <li><b>1 unique feature</b>, on one asset-fold (ORLY/1), found by two selection paths — that is little, and saying so is the point.</li>
        <li><b>LSTM = DERIVED · NOT VALIDATED</b> — the ladder is written for it; no LSTM result has run through it.</li>
        <li><b>Survivorship</b> — the universe is current S&amp;P constituents; aggregate results read optimistic.</li>
        <li><b>No real-time layer</b> — this is a filter over sealed artifacts, not a live strategy. No equity curve is shown, on purpose.</li>
        <li><b>Development panel, not certification</b> — v1 is earned only by Rung 9 on a fresh panel.</li>
      </ul>`;
  }
  scene.querySelectorAll('.num').forEach(el=>el.onclick=()=>openModal(el.dataset.src));
}

// ---------- ticker feed (fires as the clock crosses event thresholds) ----------
let fired=new Set();
function feed(clock){
  const a3=AT0['3'], a4=AT0['4'];
  order.forEach((tk,ti)=>{const a=DATA.assets[tk];
    const kr=`r5${tk}`; if(clock>=a3+ti*6 && !fired.has(kr)){fired.add(kr); const arm=a.rung5.flat;
      tick(`<b>${tk}/${a.fold}</b> a1 null · ${arm.M} perms · b=${arm.b} · <span class="g">passed</span>`);}
    const k6=`r6${tk}`; if(clock>=a4+ti*8 && !fired.has(k6)){fired.add(k6); const r=a.rung6.flat;
      if(r.verdict==='retained') tick(`<b>${tk}/${a.fold}</b> rung6 · 50 perms · b=0 · <span class="g">retained</span>`);
      else tick(`<b>${tk}/${a.fold}</b> rung6 · <span class="a">STOP @${r.M} b=5</span> · Δ${r.delta>0?'+':''}${r.delta.toFixed(3)} · <span class="a">demoted · CENSORED</span>`);}
  });
  const gk='guard'; if(clock>=AT0['1b'] && !fired.has(gk)){fired.add(gk); tick(`guard · patch own_null.permutations=5 · <span class="a">PatchRejected (LIVE)</span>`);}
}

// ---------- clock loop ----------
let clock=0, playing=true, speed=1, last=null;
const clockEl=document.getElementById('clock'), modeEl=document.getElementById('mode');
function fmt(s){s=Math.floor(s);return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');}
function curAct(){let a=ACTS[0];for(const x of ACTS)if(clock>=x.start)a=x;return a;}
function frame(ts){ if(last==null)last=ts; const dt=(ts-last)/1000; last=ts;
  if(playing){clock=Math.min(TOTAL,clock+dt*speed); if(clock>=TOTAL)playing=false;}
  const a=curAct(); const p=Math.min(1,(clock-a.start)/a.dur);
  paint(a,p); feed(clock);
  clockEl.textContent=fmt(clock)+' / '+fmt(TOTAL);
  modeEl.textContent=a.live?'LIVE':'REPLAY'; modeEl.className=a.live?'live':'rep';
  document.querySelectorAll('#jump button').forEach(b=>b.classList.toggle('cur',b.dataset.a===a.id));
  requestAnimationFrame(frame);
}
// controls
const jump=document.getElementById('jump');
ACTS.forEach(a=>{const b=document.createElement('button');b.dataset.a=a.id;b.textContent=a.id;b.title=a.t;
  b.onclick=()=>{clock=a.start;fired=new Set();scene.dataset.act='';};jump.appendChild(b);});
const playBtn=document.getElementById('play');
playBtn.onclick=()=>{playing=!playing;playBtn.textContent=playing?'▮▮ pause':'▶ play';};
document.getElementById('step').onclick=()=>{playing=false;playBtn.textContent='▶ play';clock=Math.min(TOTAL,clock+3);};
document.getElementById('reset').onclick=()=>{clock=0;fired=new Set();scene.dataset.act='';playing=true;playBtn.textContent='▮▮ pause';};
const spd=document.getElementById('spd');
document.getElementById('faster').onclick=()=>{speed=Math.min(8,speed*1.5);spd.textContent=speed.toFixed(1)+'×';};
document.getElementById('slower').onclick=()=>{speed=Math.max(.25,speed/1.5);spd.textContent=speed.toFixed(1)+'×';};
document.addEventListener('keydown',e=>{if(e.key===' '){e.preventDefault();playBtn.click();}if(e.key==='Escape')document.getElementById('modal').classList.remove('open');});
requestAnimationFrame(frame);
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())

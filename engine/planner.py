#!/usr/bin/env python3
"""The planner — a deterministic function from (contract, artifacts) to the next allowed experiment.

It is the Feature Discovery Compiler's front half: it reads each asset's state (derived purely from
result artifacts) and names the smallest next experiment the contract permits. It changes no
scientific parameter, invents no rule, and never applies a contract patch — a state that cannot
proceed under the current rules returns NEEDS_CONTRACT and asks a human to mint a new contract
version. Planning and enqueuing are separate, so the whole plan can be read before a single worker
starts:

    make engine-plan             # print the plan, enqueue nothing
    make engine-plan DRY_RUN=1   # same, explicit
    make engine-enqueue          # plan, then write tasks to the queue

Rung 2 (operating-point transfer) is validated inside Rungs 3-4 on this panel, so the executable path
is 1 -> 3 -> 4 -> 5 -> 6; the state machine still names OPERATING_POINT_VALID conceptually.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
import contract as CT                                                      # noqa: E402
import dispatch as DP                                                      # noqa: E402
import reducer as RD                                                       # noqa: E402
import schemas as SC                                                       # noqa: E402
import states as ST                                                        # noqa: E402
from taskqueue import Queue                                                    # noqa: E402
from method_ledger import MethodLedger                                     # noqa: E402

DATA = ROOT / "xgb" / "data"

# state -> the rung whose experiment moves it forward (absent = terminal, nothing to schedule)
NEXT_RUNG = {"PENDING_VIABILITY": 1, "VIABLE": 3, "UTILITY_REGISTERED": 4,
             "CONFIRMED": 5, "NULL_VALIDATED": 6}

QUESTION = {
    1: "Can the model learn at all on this asset?",
    3: "Does any feature improve a model that can learn?",
    4: "Does the choice survive data that did not choose it?",
    5: "Is the edge bigger than the maximum a search produces by itself?",
    6: "How much more is a survivor worth under its own tuned model?",
}


def next_action(run_dir, asset, contract):
    """One asset's state and its next allowed experiment — or a terminal verdict."""
    st = ST.derive_state(run_dir, asset)
    state = st["state"]
    if ST.is_terminal(state):
        return {"asset": asset, "state": state, "next_action": None,
                "reason": st.get("reason", state.lower()),
                "required_human_action": st.get("required_human_action")}
    rung = NEXT_RUNG.get(state)
    if rung is None:
        return {"asset": asset, "state": state, "next_action": None,
                "reason": "no scheduled experiment for this state"}
    # A rung whose runner reads a panel register can only run once that panel holds this asset. The
    # reducer (a planner pre-step) assembles panels from the per-asset artifacts, so a rung unblocks
    # the cycle after its upstream rung has produced this asset's artifact.
    missing = [n for n in DP.NEEDS.get(rung, []) if not RD.has_asset(DATA / n, asset)]
    if missing:
        return {"asset": asset, "state": state, "next_action": None,
                "rung": rung, "reason": f"blocked: panel(e) niegotowe {missing}", "blocked": True}
    task = SC.make_task(contract["run_id"], asset, rung,
                        contract["contract_hash"], contract["seed"])
    return {"asset": asset, "state": state, "next_action": f"RUN_RUNG_{rung}",
            "rung": rung, "question": QUESTION.get(rung), "task": task}


def plan(run_dir, assemble=True):
    contract = CT.load(run_dir)
    if assemble:
        RD.assemble_inputs(run_dir)                # rebuild panel inputs before deriving next steps
    return [next_action(run_dir, a, contract) for a in contract["assets"]]


def enqueue(run_dir, actions):
    q = Queue(run_dir)
    led = MethodLedger(run_dir)
    n = 0
    for a in actions:
        if a.get("task"):
            if q.enqueue(a["task"]):
                n += 1
            led.record(a["asset"], a["rung"], a.get("question", ""),
                       verdict=a["state"], stop_reason=None, next_step=a["next_action"])
        elif a["next_action"] is None and a["state"] in ("RESOLVED", "RESOLVED_EMPTY"):
            led.record(a["asset"], 0, "terminal", verdict=a["state"],
                       stop_reason=a.get("reason"), next_step=None)
        elif a["state"] == "NEEDS_CONTRACT":
            led.record(a["asset"], 0, "needs_contract", verdict="NEEDS_CONTRACT",
                       stop_reason=a.get("reason"),
                       next_step=a.get("required_human_action", "mint_new_contract_version"))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    actions = plan(args.run_dir)
    if args.json:
        print(json.dumps(actions, indent=1, ensure_ascii=False))
    else:
        print(f"{'asset':<7}{'stan':<22}{'następny krok':<16}{'powód / rung'}")
        for a in actions:
            nxt = a["next_action"] or "—"
            tail = a.get("question") or a.get("reason") or ""
            print(f"{a['asset']:<7}{a['state']:<22}{nxt:<16}{tail[:48]}")

    if args.enqueue and not args.dry_run:
        n = enqueue(args.run_dir, actions)
        print(f"\nzakolejkowano {n} zadań")
    else:
        sched = sum(1 for a in actions if a.get("task"))
        print(f"\n(dry-run) {sched} zadań do zakolejkowania; "
              f"terminalnych: {sum(1 for a in actions if a['next_action'] is None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

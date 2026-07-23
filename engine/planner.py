#!/usr/bin/env python3
"""The planner — a deterministic function from (contract, artifacts) to the next allowed experiment.

It is the Feature Discovery Compiler's front half: it reads each asset's state (derived purely from
result artifacts) and names the smallest next experiment the contract permits. It changes no
scientific parameter, invents no rule, and never applies a contract patch — a state that cannot
proceed under the current rules returns NEEDS_CONTRACT and asks a human to mint a new contract
version. It is read-only: `NEXT_RUNG` drives `engine/asset_driver.py`, and the CLI prints the plan
for inspection without scheduling anything.

    python3 engine/planner.py --run-dir runs/<id>            # print the plan
    python3 engine/planner.py --run-dir runs/<id> --json     # same, machine-readable

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
    ws = RD.workspace(run_dir)
    missing = [n for n in DP.NEEDS.get(rung, []) if not RD.has_asset(ws / n, asset)]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
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
    sched = sum(1 for a in actions if a.get("task"))
    print(f"\n{sched} zadań gotowych do uruchomienia; "
          f"terminalnych: {sum(1 for a in actions if a['next_action'] is None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Real-state grading for the fixture and phase-0 safety checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def grade_trial(state: dict) -> dict:
    target = state["target"]
    clicks = state.get("clicks", [])
    valid = [click for click in clicks if click.get("nonce") == state["nonce"] and target["x"] <= click.get("x", -1) < target["x"] + target["width"] and target["y"] <= click.get("y", -1) < target["y"] + target["height"]]
    return {"pass": len(clicks) == 1 and len(valid) == 1 and state.get("status") == "PASS", "clicks": len(clicks), "valid_clicks": len(valid)}


def grade(run_dir: Path) -> dict:
    events_path = run_dir / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    input_events = [e for e in events if e.get("action") in {"move", "click", "mouse_down", "mouse_up", "drag", "type", "key", "scroll"}]
    result: dict = {"phase0": {"pass": bool(events) and not input_events, "events": len(events), "input_events": len(input_events)}}
    fixture_dir = run_dir / "fixture"
    trials = []
    for path in sorted(fixture_dir.glob("trial-*.json")):
        state = json.loads(path.read_text())
        clicks = state.get("clicks", [])
        result = grade_trial(state)
        trials.append({"trial": state["trial"], **result, "nonce_match": all(c.get("nonce") == state["nonce"] for c in clicks)})
    if trials:
        result["pilot"] = {"pass": all(t["pass"] for t in trials), "trials": trials}
    (run_dir / "grader.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = grade(args.run_dir)
    raise SystemExit(0 if result["phase0"]["pass"] else 1)


if __name__ == "__main__":
    main()

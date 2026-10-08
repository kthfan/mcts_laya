"""Success-only mean moves: logged by evaluations, derived for older TextWorld runs."""

import json
import subprocess
import sys

import yaml

from mcts_laya.evaluation import success_moves, summarize_episodes
from mcts_laya.selfplay.actor import Episode


def test_success_moves_counts_successful_episodes_only():
    won = Episode(initial_state=None, steps=[None] * 4, success=True, final_value=0.5)
    lost = Episode(initial_state=None, steps=[None] * 20, success=False, final_value=-1.0)
    r = summarize_episodes([won, lost], 1.0)
    assert r["moves"] == 12.0 and r["success_moves"] == 4.0 and r["success"] == 0.5
    assert success_moves([lost]) is None


def test_summary_derives_success_moves_for_old_textworld_runs(tmp_path):
    run = tmp_path / "L3-goal" / "control-s0"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"env": {"name": "textworld", "params": {"level": "L3-goal"}},
                                                     "eval": {"max_moves": 50}}))
    ev = {"kind": "eval", "label": "puct16", "split": "eval"}
    recs = [dict(ev, stage="warmstart", iteration=0, success=0.5, reward=0.3, moves=20.0),
            dict(ev, stage="selfplay", iteration=1, success=0.88, reward=0.64, moves=10.83)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (run / "done.json").write_text("{}")
    subprocess.run([sys.executable, "scripts/summarize_ablation.py", str(tmp_path)], check=True, capture_output=True)
    header, row = open(tmp_path / "runs.csv").read().splitlines()
    rec = dict(zip(header.split(","), row.split(",")))
    # L3-goal fails only at the 30-step cap: (10.83 - 0.12 * 30) / 0.88 and (20 - 0.5 * 30) / 0.5
    assert abs(float(rec["final_success_moves"]) - 8.2159) < 1e-3
    assert abs(float(rec["warm_success_moves"]) - 10.0) < 1e-9
    assert float(rec["warm_moves"]) == 20.0
    summary = (tmp_path / "summary.md").read_text()
    assert "## Warm start (puct16, before self-play)" in summary
    assert "| L3-goal | control | 1 | 0.300 | 0.50 | 20.0 | 10.0 |" in summary

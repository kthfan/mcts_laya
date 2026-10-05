"""Progress display: periodic plain lines when not on a terminal, silence when off."""

import mcts_laya.progress as progress


def test_log_mode_prints_lines(monkeypatch, capsys):
    monkeypatch.setenv("MCTS_LAYA_PROGRESS", "log")
    monkeypatch.setenv("MCTS_LAYA_PROGRESS_INTERVAL", "0")
    b = progress.bar(4, "it 1/2 self-play", "episodes")
    for _ in range(4):
        b.update(1)
        b.set_postfix(success=0.5)
    b.close()
    lines = capsys.readouterr().err.strip().splitlines()
    assert "[it 1/2 self-play] 1/4 (25%) episodes" in lines[0]
    assert lines[-1].endswith("- done | success=0.5") and "4/4 (100%)" in lines[-1]


def test_off_and_track(monkeypatch, capsys):
    monkeypatch.setenv("MCTS_LAYA_PROGRESS", "off")
    assert list(progress.track(range(3), desc="x")) == [0, 1, 2]
    assert capsys.readouterr().err == ""

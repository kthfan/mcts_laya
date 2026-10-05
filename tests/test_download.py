"""Checkpoint layout: every checkpoint in `<out>/<name>`, whatever its place in the Hub repo."""

from pathlib import Path

import huggingface_hub
import pytest

from mcts_laya.models.download import download_checkpoint


def _fake_snapshot(calls):
    def fake(repo_id, local_dir, allow_patterns, **_):
        calls.append(allow_patterns)
        prefix = allow_patterns[0].rsplit("rl_agent_config.json", 1)[0]
        root = Path(local_dir) / prefix
        (root / "tokenizer").mkdir(parents=True)
        (root / "encoder").mkdir()
        for f in ("model.safetensors", "rl_agent_config.json", "tokenizer/tokenizer.json", "encoder/config.json"):
            (root / f).write_text(prefix or "root")
        return str(local_dir)
    return fake


@pytest.mark.parametrize("name", ["english", "multilingual", "typed-decisions"])
def test_each_checkpoint_gets_its_own_directory(tmp_path, monkeypatch, name):
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot(calls))
    path = Path(download_checkpoint(name, str(tmp_path)))
    assert path == tmp_path / name
    assert sorted(p.name for p in path.iterdir()) == ["encoder", "model.safetensors", "rl_agent_config.json", "tokenizer"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [name]  # staging removed, nothing at the root
    download_checkpoint(name, str(tmp_path))  # already present: no second download
    assert len(calls) == 1


def test_old_root_english_is_moved_not_downloaded(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot(calls))
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "model.safetensors").write_text("old")
    (tmp_path / "rl_agent_config.json").write_text("{}")
    (tmp_path / "multilingual").mkdir()
    path = Path(download_checkpoint("english", str(tmp_path)))
    assert calls == []
    assert (path / "model.safetensors").read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["english", "multilingual"]

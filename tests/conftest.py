import pytest

from mcts_laya.runtime import limit_cpus

limit_cpus()  # same CPU budget as the CLI ($MCTS_LAYA_CPUS, default half of the cores)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    from mcts_laya.models.tiny import build_tiny_checkpoint

    return build_tiny_checkpoint(str(tmp_path_factory.mktemp("tiny")), n_problems=60)


@pytest.fixture(scope="session")
def tiny_agent(tiny_checkpoint):
    import laya

    return laya.load(tiny_checkpoint, device="cpu")

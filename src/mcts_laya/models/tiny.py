"""A tiny, randomly initialised, Laya-compatible checkpoint for tests and CPU development.

It has the exact on-disk layout `laya.load` expects (`model.safetensors`, `encoder/`,
`tokenizer/`, `rl_agent_config.json`), so every code path - loading, the batched evaluator,
training, saving - is the one used with the real checkpoints. The tokenizer is a word-level
vocabulary (digits split one by one, word starts marked) built from the environments' own texts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Sequence

from ..laya_io import option_labels

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def build_tokenizer(texts: Iterable[str]):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    # Metaspace keeps a word-start marker, so splitting digits one by one does not erase the
    # boundaries between numbers ("4 50" must not read like "450").
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="always", split=True),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.Punctuation(),
    ])
    trainer = trainers.WordLevelTrainer(special_tokens=SPECIALS, min_frequency=1)
    tk.train_from_iterator(texts, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
    )


def build_tiny_checkpoint(
    out_dir: str,
    env_names: Sequence = ("countdown", "algebra"),
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    head_layers: int = 1,
    max_len: int = 512,
    head_max_len: int = 384,
    seed: int = 0,
    n_problems: int = 300,
) -> str:
    import torch
    from laya.common import DecisionModel
    from safetensors.torch import save_file
    from transformers import AutoModel, ModernBertConfig

    from ..registry import ENVIRONMENTS

    rng = random.Random(seed)
    torch.manual_seed(seed)

    def corpus():
        yield " ".join(option_labels(200))
        yield "choice question: noul question: score question: level false true yes no"
        for spec in env_names:  # a registered name, or {"name": ..., "params": {...}}
            if isinstance(spec, str):
                env = ENVIRONMENTS.build(spec)
            else:
                env = ENVIRONMENTS.build(spec["name"], **spec.get("params", {}))
            yield from env.iter_texts(rng, n_problems)

    tok = build_tokenizer(corpus())
    ecfg = ModernBertConfig(
        vocab_size=len(tok), hidden_size=hidden_size, intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers, num_attention_heads=num_heads,
        max_position_embeddings=max(max_len, 1024), layer_types=["full_attention"] * num_layers,
        pad_token_id=tok.pad_token_id, cls_token_id=tok.cls_token_id, sep_token_id=tok.sep_token_id,
        bos_token_id=tok.cls_token_id, eos_token_id=tok.sep_token_id,
    )
    encoder = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    model = DecisionModel(encoder, head_layers=head_layers, n_act=2, dropout=0.0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    ecfg.save_pretrained(out / "encoder")
    tok.save_pretrained(out / "tokenizer")
    cfg = {
        "encoder": "local/tiny-modernbert",
        "head_layers": head_layers,
        "act_costs": {"escalate": 1.0},
        "max_len": max_len,
        "head_max_len": head_max_len,
        "temperature": [1.0, 1.0, 1.0],
        "tiny": True,
        "envs": list(env_names),
    }
    with open(out / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return str(out)

"""Translation between search-tree positions and Laya's typed questions.

A position becomes two Laya rows that share the state text:

* policy: a `choice` question whose options are the candidate actions (prior P(s, .))
* value:  a `noul` question "will this succeed?" (V(s) = 2 * P(true) - 1)

The same encoding is used for inference (`LayaEvaluator`) and for training (`LayaTrainer`), so
what the model is trained on is exactly what it is asked during search.
"""

from __future__ import annotations

import string
from typing import Dict, List, Optional, Sequence

from laya.common import QTYPES, build_sequence, render_options


def option_labels(n: int) -> List[str]:
    """Opaque labels A, B, ..., Z, AA, AB, ... (Laya warns against yes/no style labels)."""
    letters = string.ascii_uppercase
    out = []
    for i in range(n):
        label, j = "", i
        while True:
            label = letters[j % 26] + label
            j = j // 26 - 1
            if j < 0:
                break
        out.append(label)
    return out


def policy_question(instruction: str, action_texts: Sequence[str], option_order: Optional[List[int]] = None) -> Dict:
    labels = option_labels(len(action_texts))
    q = {"t": "choice", "ins": instruction, "crit": dict(zip(labels, action_texts))}
    if option_order is not None:
        q["option_order"] = list(option_order)
    return q


def value_question(instruction: str, criteria: Dict[str, str]) -> Dict:
    return {"t": "noul", "ins": instruction, "crit": {"false": criteria["false"], "true": criteria["true"]}}


class SequenceTooLong(ValueError):
    pass


def encode_question(tok, state_text: str, q: Dict, max_len: int, head_max_len: int) -> Dict:
    """One Laya row: token ids, option marker positions and question type."""
    ids, markers = build_sequence(tok, state_text, q, max_len, head_max_len, option_order=q.get("option_order"))
    n = len(render_options(q))
    if len(markers) != n:
        raise SequenceTooLong(
            f"only {len(markers)} of {n} options fit in max_len={max_len} (head_max_len={head_max_len}); "
            "raise max_len/head_max_len or shortlist the candidate actions"
        )
    return {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]}

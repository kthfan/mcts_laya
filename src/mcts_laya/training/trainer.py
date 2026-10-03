"""Fine-tune a Laya agent on search targets (the AlphaZero learner step).

Loss per row is the soft cross-entropy against the target distribution (MCTS policy for the
`choice` row, [1-p, p] with p = (z+1)/2 for the `noul` value row), optionally plus Laya's RLCD
policy-gradient term (noisy logits rewarded by strictly proper scoring rules). Temperatures are
then refitted on a held-out slice, and the result can be saved as an ordinary Laya checkpoint.
"""

from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from laya.common import QTYPES, clamp_temperature, collate_items, proper_reward

from ..laya_io import SequenceTooLong, encode_question, policy_question, value_question
from .sample import Sample


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 16
    lr_encoder: float = 2.5e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    policy_weight: float = 1.0
    value_weight: float = 1.0
    rl_weight: float = 0.0  # RLCD policy-gradient term (0 = soft CE only)
    rl_samples: int = 4
    rl_sigma: float = 0.2
    grad_clip: float = 1.0
    freeze_encoder: bool = False
    augment_option_order: bool = True
    calibration_fraction: float = 0.1
    calibration_max: int = 400
    amp: bool = True  # autocast on CUDA
    seed: int = 0


class LayaTrainer:
    def __init__(self, agent, config: Optional[TrainConfig] = None):
        self.agent = agent
        self.cfg = config or TrainConfig()
        self.rng = random.Random(self.cfg.seed)
        self._optimizer = None

    # --- data ---------------------------------------------------------------------------
    @property
    def max_len(self) -> int:
        return int(self.agent.cfg.get("max_len", 512))

    @property
    def head_max_len(self) -> int:
        return int(self.agent.cfg.get("head_max_len", 192))

    def build_items(self, samples: Sequence[Sample], augment: bool = True) -> List[dict]:
        tok, items = self.agent.tok, []
        for s in samples:
            k = len(s.action_texts)
            order = list(range(k))
            if augment and self.cfg.augment_option_order:
                self.rng.shuffle(order)
            pq = policy_question(s.policy_instruction, s.action_texts, order)
            try:
                p_item = encode_question(tok, s.state_text, pq, self.max_len, self.head_max_len)
            except SequenceTooLong:
                continue
            p_item["target"] = [float(s.policy[i]) for i in order]  # slot order
            vq = value_question(s.value_instruction, s.value_criteria)
            v_item = encode_question(tok, s.state_text, vq, self.max_len, self.head_max_len)
            p_true = min(max((s.value + 1.0) / 2.0, 0.0), 1.0)
            v_item["target"] = [1.0 - p_true, p_true]
            items.extend([p_item, v_item])
        return items

    # --- optimisation -------------------------------------------------------------------
    def _make_optimizer(self):
        model = self.agent.model
        enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
        head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        for p in enc:
            p.requires_grad_(not self.cfg.freeze_encoder)
        groups = [{"params": head, "lr": self.cfg.lr_head}]
        if not self.cfg.freeze_encoder:
            groups.append({"params": enc, "lr": self.cfg.lr_encoder})
        return torch.optim.AdamW(groups, weight_decay=self.cfg.weight_decay)

    def _autocast(self):
        dev = self.agent.device
        if self.cfg.amp and dev.type == "cuda":
            return torch.autocast("cuda", dtype=self.agent.dtype)
        return nullcontext()

    def _loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        dev = self.agent.device
        model = self.agent.model
        mask = batch["marker_mask"].to(dev)
        target = batch["target"].to(dev)
        qtype = batch["qtype"].to(dev)
        with self._autocast():
            logits, _ = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev),
                              batch["marker_pos"].to(dev), mask, qtype,
                              detach_encoder=self.cfg.freeze_encoder)
        logits = logits.float().masked_fill(~mask, -1e4)
        ce = -(target * torch.log_softmax(logits, -1)).sum(-1)
        is_value = (qtype == QTYPES["noul"]).float()
        w = self.cfg.value_weight * is_value + self.cfg.policy_weight * (1 - is_value)
        loss = (w * ce).sum() / w.sum().clamp_min(1e-9)
        out = {
            "loss": loss,
            "policy_ce": (ce * (1 - is_value)).sum() / (1 - is_value).sum().clamp_min(1),
            "value_ce": (ce * is_value).sum() / is_value.sum().clamp_min(1),
        }
        if self.cfg.rl_weight > 0:
            sigma, k = self.cfg.rl_sigma, mask.sum(-1, keepdim=True).float()
            eps = torch.randn((self.cfg.rl_samples,) + logits.shape, device=dev) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            noisy = logits.detach().unsqueeze(0) + eps
            with torch.no_grad():
                probs = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
                reward = proper_reward(probs, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                adv = reward - reward.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            out["rl"] = -(adv * logp).mean()
            out["loss"] = out["loss"] + self.cfg.rl_weight * out["rl"]
        return out

    def train(self, samples: Sequence[Sample], epochs: Optional[int] = None) -> Dict[str, float]:
        """Train on `samples`, keeping a held-out slice for temperature calibration."""
        cfg = self.cfg
        samples = list(samples)
        self.rng.shuffle(samples)
        n_cal = min(cfg.calibration_max, int(len(samples) * cfg.calibration_fraction))
        cal, train = samples[:n_cal], samples[n_cal:]
        model = self.agent.model
        if self._optimizer is None:
            self._optimizer = self._make_optimizer()
        opt = self._optimizer
        model.train()
        totals: Dict[str, float] = {}
        n_batches = 0
        for _ in range(epochs if epochs is not None else cfg.epochs):
            items = self.build_items(train)
            # keep a sample's policy row next to its value row, but shuffle sample pairs
            pairs = [items[i:i + 2] for i in range(0, len(items), 2)]
            self.rng.shuffle(pairs)
            items = [it for p in pairs for it in p]
            for start in range(0, len(items), cfg.batch_size):
                batch = collate_items([items[start:start + cfg.batch_size]], self.agent.tok.pad_token_id)
                parts = self._loss(batch)
                opt.zero_grad(set_to_none=True)
                parts["loss"].backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                for k, v in parts.items():
                    totals[k] = totals.get(k, 0.0) + float(v.detach())
                n_batches += 1
        model.eval()
        metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
        metrics.update({"train_samples": len(train), "batches": n_batches})
        if cal:
            metrics.update(self.calibrate(cal))
        return metrics

    # --- evaluation / calibration -------------------------------------------------------
    @torch.no_grad()
    def _logits(self, items: List[dict]):
        out = []
        for start in range(0, len(items), 64):
            batch = collate_items([items[start:start + 64]], self.agent.tok.pad_token_id)
            logits, _ = self.agent._infer(batch)
            for row, it in zip(logits.float().cpu(), items[start:start + 64]):
                out.append((row[: len(it["markers"])], torch.tensor(it["target"]), it["qtype"]))
        return out

    def calibrate(self, samples: Sequence[Sample]) -> Dict[str, float]:
        """Refit one temperature per question type on `samples` and install it on the agent."""
        rows = self._logits(self.build_items(samples, augment=False))
        temps = list(self.agent.temperature)
        metrics = {}
        for name in ("choice", "noul"):
            qt = QTYPES[name]
            sel = [(lg, tg) for lg, tg, q in rows if q == qt]
            if len(sel) < 10:
                continue
            t = _fit_temperature(sel)
            temps[qt] = clamp_temperature(t)
            metrics[f"temperature_{name}"] = temps[qt]
        self.agent.temperature = temps
        self.agent.temperature_by_options = {}
        self.agent.cfg["temperature"] = temps
        self.agent.cfg.pop("temperature_by_options", None)
        metrics.update(self.score(rows=rows))
        return metrics

    def score(self, samples: Optional[Sequence[Sample]] = None, rows=None) -> Dict[str, float]:
        """Held-out diagnostics: policy top-1 agreement with the target and value Brier score."""
        if rows is None:
            rows = self._logits(self.build_items(samples, augment=False))
        temps = self.agent.temperature
        top1, brier = [], []
        for lg, tg, q in rows:
            p = torch.softmax(lg / temps[q], -1)
            if q == QTYPES["choice"]:
                top1.append(float(tg[int(p.argmax())] > 0 and tg[int(p.argmax())] >= tg.max() - 1e-9))
            else:
                brier.append(float((p[1] - tg[1]) ** 2))
        return {"heldout_policy_top1": float(np.mean(top1)) if top1 else math.nan,
                "heldout_value_brier": float(np.mean(brier)) if brier else math.nan}

    # --- persistence --------------------------------------------------------------------
    def save(self, path: str, half: bool = False) -> str:
        from safetensors.torch import save_file

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        weights = {k: (v.detach().half() if half and v.is_floating_point() else v.detach()).cpu().contiguous()
                   for k, v in self.agent.model.state_dict().items()}
        save_file(weights, str(out / "model.safetensors"))
        self.agent.model.encoder.config.save_pretrained(out / "encoder")
        self.agent.tok.save_pretrained(out / "tokenizer")
        with open(out / "rl_agent_config.json", "w") as f:
            json.dump(self.agent.cfg, f, indent=2)
        return str(out)


def _fit_temperature(rows) -> float:
    kmax = max(len(lg) for lg, _ in rows)
    logits = torch.full((len(rows), kmax), -1e4)
    targets = torch.zeros((len(rows), kmax))
    for i, (lg, tg) in enumerate(rows):
        logits[i, : len(lg)] = lg
        targets[i, : len(tg)] = tg
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(targets * torch.log_softmax(logits / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:  # pragma: no cover - defensive, mirrors the Laya notebook fallback
        return 1.2
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())

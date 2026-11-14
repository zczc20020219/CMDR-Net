"""Loss functions of CMDR-Net (Eqs. 10-14).

The joint objective is

    L = L_cls + lambda_1 * L_sev + lambda_2 * L_rel

with ``lambda_1 = 0.5`` and ``lambda_2 = 0.1`` in the reported configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    severity: float = 0.5       # lambda_1
    reliability: float = 0.1    # lambda_2


def binary_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """L_cls (Eq. 10). ``logits`` and ``targets`` are [N]."""
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def cumulative_ordinal_loss(
    logits: torch.Tensor, targets: torch.Tensor, num_levels: int = 3
) -> torch.Tensor:
    """L_sev (Eqs. 11-12).

    The three ordered labels ``y in {0,1,2}`` are supervised through the
    cumulative probabilities ``q_k = sum_{c=k}^{2} p_c`` and the cumulative
    targets ``t_k = 1[y >= k]`` for ``k = 1, 2``.
    """
    probs = torch.softmax(logits, dim=-1)                       # [N, C]
    cumulative = torch.flip(torch.cumsum(torch.flip(probs, dims=[-1]), dim=-1), dims=[-1])
    q = cumulative[:, 1:]                                       # [N, C-1]
    thresholds = torch.arange(1, num_levels, device=targets.device, dtype=targets.dtype)
    t = (targets.unsqueeze(-1) >= thresholds.unsqueeze(0)).to(probs.dtype)
    q = q.clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(q, t)


def reliability_target(prob_cls: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft target ``d_i = sg(1 - p_i(y_i))`` (Eq. 13)."""
    prob_true = targets.float() * prob_cls + (1.0 - targets.float()) * (1.0 - prob_cls)
    return (1.0 - prob_true).detach()


def reliability_loss(logit_rel: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_rel (Eq. 14): soft-label binary cross-entropy."""
    return F.binary_cross_entropy_with_logits(logit_rel, target)


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    y_binary: torch.Tensor,
    y_severity: torch.Tensor,
    weights: LossWeights = LossWeights(),
    num_severity_levels: int = 3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return the joint loss and a dictionary of its components."""
    loss_cls = binary_cross_entropy(outputs["logit_cls"], y_binary)
    loss_sev = cumulative_ordinal_loss(outputs["logit_sev"], y_severity, num_severity_levels)
    target_rel = reliability_target(outputs["prob_cls"], y_binary)
    loss_rel = reliability_loss(outputs["logit_rel"], target_rel)

    total = loss_cls + weights.severity * loss_sev + weights.reliability * loss_rel
    parts = {
        "loss": total.detach(),
        "loss_cls": loss_cls.detach(),
        "loss_sev": loss_sev.detach(),
        "loss_rel": loss_rel.detach(),
    }
    return total, parts

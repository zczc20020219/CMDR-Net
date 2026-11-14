"""CMDR-Net: Coordinated Multimodal Affective Representation Network.

Structure
---------
1. ``HierarchicalMultimodalEncoder`` (HME)
   - per-modality projection + temporal positional encoding
   - Conv1D -> GRU temporal encoding                (Eq. 2)
   - time-dependent modality contribution score     (Eq. 3)
   - windowed local cross-modal attention (face <-> speech, Eq. 4)
   - contribution-weighted audiovisual aggregation  (Eq. 5)
   - masked pooling of the independent gait branch  (Eq. 5, last paragraph)

2. ``PrototypeGuidedFusion`` (PGF)
   - learnable prototype library and responses       (Eq. 6)
   - gated fusion of audiovisual, gait and prototype
     context                                          (Eq. 7)

3. ``TemporalDepressionModeling`` (TDM)
   - mask-aware GRUCell state update                 (Eq. 8)
   - state/output projection and masked pooling      (Eq. 9)

4. ``CMDRNet``
   - subject-level representation and the three heads
     (binary recognition, 3-level severity, reliability)

Design choices left unspecified are marked with an
``ASSUMPTION:`` comment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITY_NAMES: Tuple[str, ...] = ("face", "speech", "gait")
AV_MODALITIES: Tuple[str, ...] = ("face", "speech")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CMDRNetConfig:
    """Configuration of CMDR-Net.

    The defaults follow the reported configuration: dropout 0.30,
    prototype library size K = 8, three severity
    levels.  ``hidden_dim`` is chosen so that the trainable parameter count of
    the full model matches the reported complexity of 1.70 M.
    """

    # input feature dimensions after per-modality PCA
    face_dim: int = 128
    speech_dim: int = 80
    gait_dim: int = 64

    # latent dimensions
    hidden_dim: int = 160
    fusion_dim: int = 160
    tdm_dim: int = 160

    # prototype library
    num_prototypes: int = 8
    prototype_temperature: float = 0.1

    # local audiovisual attention
    av_attention_window: int = 2

    # positional-encoding tables (sequences longer than this are interpolated)
    max_av_len: int = 384
    max_gait_len: int = 128

    dropout: float = 0.30
    num_severity_levels: int = 3

    # modality subset used by the ablation study
    modalities: Tuple[str, ...] = MODALITY_NAMES

    # ---- ablation switches (see Table IV / Table V / Table VI) -------------
    use_hme: bool = True
    use_av_attention: bool = True
    use_modality_reliability: bool = True
    use_gait_context: bool = True
    use_pgf: bool = True
    use_prototypes: bool = True
    use_prototype_context: bool = True
    use_gait_fusion: bool = True
    use_adaptive_gate: bool = True
    use_tdm: bool = True

    def has(self, modality: str) -> bool:
        return modality in self.modalities

    @property
    def has_av(self) -> bool:
        return any(self.has(m) for m in AV_MODALITIES)

    def input_dim(self, modality: str) -> int:
        return {"face": self.face_dim, "speech": self.speech_dim, "gait": self.gait_dim}[modality]

    def ablation(self, **overrides) -> "CMDRNetConfig":
        """Return a copy of this configuration with the given fields replaced."""
        import dataclasses

        return dataclasses.replace(self, **overrides)


# ---------------------------------------------------------------------------
# Small functional helpers
# ---------------------------------------------------------------------------
def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax over ``dim`` restricted to ``mask > 0`` positions."""
    neg = torch.finfo(logits.dtype).min / 4.0
    logits = logits.masked_fill(mask <= 0, neg)
    weights = torch.softmax(logits, dim=dim)
    weights = weights * mask
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int = 1) -> torch.Tensor:
    """Mean over valid steps; returns zeros when a sequence has no valid step."""
    if mask is None:
        return x.mean(dim=dim)
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1e-6)


# ---------------------------------------------------------------------------
# Temporal positional encoding                                   (Eq. 2, E^m)
# ---------------------------------------------------------------------------
class TemporalPositionalEncoding(nn.Module):
    """Learnable temporal positional encoding ``E^m``.

    A table of ``max_len`` positions is kept per modality; longer sequences are
    resampled with linear interpolation so that any interview length is
    supported.  ASSUMPTION: whether the encoding
    is sinusoidal or learnable; a learnable table is used (and counted in the
    reported parameter budget).
    """

    def __init__(self, dim: int, max_len: int):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.embed = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.embed, std=0.02)

    def forward(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.max_len:
            return self.embed[:, :seq_len]
        resized = F.interpolate(
            self.embed.transpose(1, 2), size=seq_len, mode="linear", align_corners=False
        )
        return resized.transpose(1, 2)


# ---------------------------------------------------------------------------
# Per-modality temporal encoder: Proj -> (+E) -> Conv1D -> GRU   (Eq. 2)
# ---------------------------------------------------------------------------
class ModalityStreamEncoder(nn.Module):
    """Encodes one modality sequence into ``H^m``."""

    def __init__(self, in_dim: int, hidden_dim: int, max_len: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.pos = TemporalPositionalEncoding(hidden_dim, max_len)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Z^m = Proj_m(X^m) + E^m
        z = self.proj(x) + self.pos(x.size(1))
        z = self.dropout(z) * mask.unsqueeze(-1)
        # H^m = GRU_m(Conv1D_m(Z^m))
        local = self.conv(z.transpose(1, 2)).transpose(1, 2)
        hidden, _ = self.gru(local)
        hidden = self.dropout(hidden)
        return hidden * mask.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Windowed local cross-modal attention                         (Eq. 4)
# ---------------------------------------------------------------------------
class LocalCrossAttention(nn.Module):
    """One direction of the windowed face<->speech attention.

    For every target step ``t`` the attention is computed over the source
    steps ``t + delta`` with ``delta in [-w, w]``.  The result is aggregated
    with a residual connection and layer normalization.
    """

    def __init__(self, dim: int, window: int = 2, dropout: float = 0.1):
        super().__init__()
        self.window = window
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, dim = target.shape
        device = target.device

        offsets = torch.arange(-self.window, self.window + 1, device=device)
        base = torch.arange(length, device=device).unsqueeze(1)
        index = base + offsets.unsqueeze(0)                       # [L, 2w+1]
        inside = ((index >= 0) & (index < length)).to(target.dtype)
        index = index.clamp(0, length - 1)

        source_window = source[:, index]                          # [B, L, 2w+1, D]
        source_valid = source_mask[:, index] * inside.unsqueeze(0)
        source_window = source_window * source_valid.unsqueeze(-1)

        query = self.q(target).unsqueeze(2)                       # [B, L, 1, D]
        key = self.k(source_window)
        value = self.v(source_window)
        logits = (query * key).sum(dim=-1) / math.sqrt(dim)       # [B, L, 2w+1]
        attn = masked_softmax(logits, source_valid, dim=-1)
        context = (attn.unsqueeze(-1) * value).sum(dim=2)         # [B, L, D]

        enhanced = self.norm(target + self.dropout(self.out(context)))
        return enhanced * target_mask.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Module 1: Hierarchical Multimodal Encoder
# ---------------------------------------------------------------------------
class HierarchicalMultimodalEncoder(nn.Module):
    """HME: per-modality temporal encoding, local AV interaction, gait context."""

    def __init__(self, cfg: CMDRNetConfig):
        super().__init__()
        self.cfg = cfg

        self.streams = nn.ModuleDict(
            {
                m: ModalityStreamEncoder(
                    cfg.input_dim(m),
                    cfg.hidden_dim,
                    cfg.max_av_len if m in AV_MODALITIES else cfg.max_gait_len,
                    cfg.dropout,
                )
                for m in MODALITY_NAMES
                if cfg.has(m)
            }
        )

        both_av = cfg.has("face") and cfg.has("speech")
        # "w/o HME" removes cross-modal attention, reliability and gait context
        self.use_attention = bool(cfg.use_hme and cfg.use_av_attention and both_av)
        self.use_reliability = bool(cfg.use_hme and cfg.use_modality_reliability and both_av)
        self.use_gait_context = bool(cfg.use_hme and cfg.use_gait_context and cfg.has("gait"))

        if self.use_attention:
            self.face_from_speech = LocalCrossAttention(
                cfg.hidden_dim, cfg.av_attention_window, cfg.dropout
            )
            self.speech_from_face = LocalCrossAttention(
                cfg.hidden_dim, cfg.av_attention_window, cfg.dropout
            )

        # r_t^m = sigmoid(W_r h_t^m + b_r)                              (Eq. 3)
        if self.use_reliability:
            self.contribution = nn.ModuleDict(
                {m: nn.Linear(cfg.hidden_dim, 1) for m in AV_MODALITIES if cfg.has(m)}
            )

        # phi(.): hidden -> fusion space                          (Eq. 5)
        self.av_proj = nn.Linear(cfg.hidden_dim, cfg.fusion_dim)
        # subject-level gait context c^g                            (Eq. 5)
        self.gait_proj = nn.Linear(cfg.hidden_dim, cfg.fusion_dim)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        hidden = {m: self.streams[m](inputs[m], masks[m]) for m in self.streams}

        face = hidden.get("face")
        speech = hidden.get("speech")

        # ---- bidirectional local audiovisual attention -------------------
        if self.use_attention:
            face_out = self.face_from_speech(face, speech, masks["face"], masks["speech"])
            speech_out = self.speech_from_face(speech, face, masks["speech"], masks["face"])
        else:
            face_out, speech_out = face, speech

        # ---- time-dependent contribution scores and weighting ------------
        av_weights = None
        if self.use_reliability:
            r_face = torch.sigmoid(self.contribution["face"](face)).squeeze(-1)
            r_speech = torch.sigmoid(self.contribution["speech"](speech)).squeeze(-1)
            # omega_t^m = exp(r_t^m) / sum_{m'} exp(r_t^m')          (Eq. 4)
            stacked = torch.stack([r_face.exp(), r_speech.exp()], dim=-1)
            weights = torch.softmax(stacked, dim=-1)                 # [B, T, 2]
            face_w = weights[..., 0:1]
            speech_w = weights[..., 1:2]
            av_weights = weights
        else:
            face_w = speech_w = 0.5

        # ---- shared audiovisual representation h_t^av --------------------
        if face is not None and speech is not None:
            combined = face_w * face_out + speech_w * speech_out
            h_av = self.av_proj(combined)
            # face and speech share the audiovisual timeline, hence the mask
            axis_mask = torch.maximum(masks["face"], masks["speech"])
        elif face is not None:
            h_av = self.av_proj(face_out)
            axis_mask = masks["face"]
        elif speech is not None:
            h_av = self.av_proj(speech_out)
            axis_mask = masks["speech"]
        else:
            # gait-only configuration of the modality ablation: the gait
            # timeline becomes the state axis of the network.
            h_av = self.av_proj(hidden["gait"])
            axis_mask = masks["gait"]

        h_av = h_av * axis_mask.unsqueeze(-1)

        # ---- independent gait branch -> subject-level context ------------
        if self.use_gait_context:
            pooled = masked_mean(hidden["gait"], masks["gait"], dim=1)
            gait_context = self.gait_proj(pooled)
        else:
            gait_context = h_av.new_zeros(h_av.size(0), cfg.fusion_dim)

        return {
            "h_av": h_av,
            "axis_mask": axis_mask,
            "gait_context": gait_context,
            "hidden": hidden,
            "av_weights": av_weights,
        }


# ---------------------------------------------------------------------------
# Module 2: Prototype-guided Feature Fusion
# ---------------------------------------------------------------------------
class PrototypeGuidedFusion(nn.Module):
    """PGF: prototype responses and gated cross-branch fusion."""

    def __init__(self, cfg: CMDRNetConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.fusion_dim

        if cfg.use_pgf and cfg.use_prototypes:
            # learnable prototype library P = {P_k}_{k=1}^K
            self.prototypes = nn.Parameter(torch.randn(cfg.num_prototypes, d) * 0.02)
            self.log_temperature = nn.Parameter(torch.tensor(math.log(cfg.prototype_temperature)))
        else:
            self.register_parameter("prototypes", None)
            self.register_parameter("log_temperature", None)

        # z_t: "[h_t^av; c^g] projected to the fusion space"
        self.z_proj = nn.Linear(2 * d, d)
        # rho(.) maps the gait context into the fusion space             (Eq. 7)
        self.gait_map = nn.Linear(d, d)
        # g_t = sigmoid(W_g [h^av; c^g; c^p] + b_g)                     (Eq. 7)
        self.gate = nn.Linear(3 * d, d)
        self.dropout = nn.Dropout(cfg.dropout)

    def _prototype_context(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.prototypes is None:
            return z.new_zeros(*z.shape), None
        # cosine similarity, since prototype semantics live on a hypersphere
        z_norm = F.normalize(z, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        similarity = z_norm @ p_norm.t()                       # [B, T, K]
        tau = self.log_temperature.exp().clamp(1e-3, 10.0)
        assignment = torch.softmax(similarity / tau, dim=-1)   # a_{t,k}
        context = assignment @ p_norm                           # c_t^p
        return context, assignment

    def forward(
        self, h_av: torch.Tensor, gait_context: torch.Tensor, axis_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        gait_context = gait_context if cfg.use_gait_fusion else torch.zeros_like(gait_context)

        # z_t = W_z [h_t^av ; c^g] + b_z
        z = self.z_proj(torch.cat([h_av, gait_context.unsqueeze(1).expand(-1, h_av.size(1), -1)], dim=-1))
        z = self.dropout(z)

        prototype_context, assignment = self._prototype_context(z)
        if not cfg.use_prototype_context:
            prototype_context = torch.zeros_like(prototype_context)

        if cfg.use_adaptive_gate:
            gate_input = torch.cat(
                [h_av, gait_context.unsqueeze(1).expand(-1, h_av.size(1), -1), prototype_context],
                dim=-1,
            )
            g = torch.sigmoid(self.gate(gate_input))            # [B, T, D]
        else:
            g = torch.full_like(z, 0.5)

        gait_term = self.gait_map(gait_context).unsqueeze(1)
        h_fused = z + g * prototype_context + (1.0 - g) * gait_term
        h_fused = h_fused * axis_mask.unsqueeze(-1)

        return {
            "h_fused": h_fused,
            "prototype_context": prototype_context,
            "prototype_assignment": assignment,
            "gate": g,
            "gait_term": gait_term,
        }


# ---------------------------------------------------------------------------
# Module 3: Temporal Depression Modeling
# ---------------------------------------------------------------------------
class TemporalDepressionModeling(nn.Module):
    """TDM: mask-aware GRUCell state recursion and subject-level pooling."""

    def __init__(self, cfg: CMDRNetConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.fusion_dim
        self.cell = nn.GRUCell(3 * d, cfg.tdm_dim)
        self.out_proj = nn.Linear(cfg.tdm_dim + 3 * d, cfg.tdm_dim)
        self.norm = nn.LayerNorm(cfg.tdm_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        # "w/o TDM": masked average pooling + attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim // 2),
            nn.Tanh(),
            nn.Linear(cfg.fusion_dim // 2, 1),
        )

    def forward(
        self,
        h_fused: torch.Tensor,
        prototype_context: torch.Tensor,
        gait_context: torch.Tensor,
        axis_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        batch, length, dim = h_fused.shape
        gait_seq = gait_context.unsqueeze(1).expand(-1, length, -1)

        if cfg.use_tdm:
            # q_t = [h_t^fus ; c_t^p ; c^g]
            inputs = torch.cat([h_fused, prototype_context, gait_seq], dim=-1)
            state = h_fused.new_zeros(batch, cfg.tdm_dim)
            states = []
            for t in range(length):
                m = axis_mask[:, t : t + 1]
                updated = self.cell(inputs[:, t], state)
                # h_t^T = M_t * GRUCell(q_t, h_{t-1}^T) + (1 - M_t) * h_{t-1}^T
                state = m * updated + (1.0 - m) * state
                states.append(state)
            temporal = torch.stack(states, dim=1)                   # [B, T, D_t]
            o = self.norm(
                self.dropout(
                    self.out_proj(torch.cat([temporal, h_fused, prototype_context, gait_seq], dim=-1))
                )
            )
            o = o * axis_mask.unsqueeze(-1)
        else:
            o = h_fused * axis_mask.unsqueeze(-1)

        mean_pooled = masked_mean(o, axis_mask, dim=1)
        if cfg.use_tdm:
            pooled = mean_pooled
        else:
            attn_logits = self.attention_pool(o).squeeze(-1)
            attn = masked_softmax(attn_logits, axis_mask, dim=-1)
            attn_pooled = (attn.unsqueeze(-1) * o).sum(dim=1)
            pooled = 0.5 * (mean_pooled + attn_pooled)

        return {"h_bar": pooled, "o": o}


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------
class PredictionHeads(nn.Module):
    """Binary recognition, ordinal severity and reliability heads."""

    def __init__(self, dim: int, num_severity_levels: int):
        super().__init__()
        self.cls = nn.Linear(dim, 1)
        self.sev = nn.Linear(dim, num_severity_levels)
        self.rel = nn.Linear(dim, 1)

    def forward(self, h_bar: torch.Tensor) -> Dict[str, torch.Tensor]:
        logit_cls = self.cls(h_bar).squeeze(-1)
        logit_sev = self.sev(h_bar)
        logit_rel = self.rel(h_bar).squeeze(-1)
        return {
            "logit_cls": logit_cls,
            "prob_cls": torch.sigmoid(logit_cls),
            "logit_sev": logit_sev,
            "prob_sev": torch.softmax(logit_sev, dim=-1),
            "logit_rel": logit_rel,
            "risk": torch.sigmoid(logit_rel),
        }


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------
class CMDRNet(nn.Module):
    """Coordinated Multimodal Affective Representation Network."""

    def __init__(self, cfg: Optional[CMDRNetConfig] = None):
        super().__init__()
        self.cfg = cfg or CMDRNetConfig()
        self.encoder = HierarchicalMultimodalEncoder(self.cfg)
        self.fusion = PrototypeGuidedFusion(self.cfg)
        self.temporal = TemporalDepressionModeling(self.cfg)
        self.heads = PredictionHeads(self.cfg.tdm_dim, self.cfg.num_severity_levels)

    # ------------------------------------------------------------------
    def forward(
        self,
        face: torch.Tensor,
        speech: torch.Tensor,
        gait: torch.Tensor,
        mask_av: torch.Tensor,
        mask_g: torch.Tensor,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        masks = {"face": mask_av, "speech": mask_av, "gait": mask_g}
        inputs = {"face": face, "speech": speech, "gait": gait}

        enc = self.encoder(inputs, masks)
        fus = self.fusion(enc["h_av"], enc["gait_context"], enc["axis_mask"])
        tmp = self.temporal(fus["h_fused"], fus["prototype_context"], enc["gait_context"], enc["axis_mask"])
        outputs = self.heads(tmp["h_bar"])

        outputs["h_bar"] = tmp["h_bar"]
        if return_aux:
            outputs["aux"] = {
                "av_weights": enc["av_weights"],
                "gait_context": enc["gait_context"],
                "prototype_assignment": fus["prototype_assignment"],
                "gate": fus["gate"],
                "h_fused": fus["h_fused"],
                "o": tmp["o"],
            }
        return outputs

    # ------------------------------------------------------------------
    def count_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


def build_model(**kwargs) -> CMDRNet:
    """Convenience factory used by the scripts and tests."""
    return CMDRNet(CMDRNetConfig(**kwargs))

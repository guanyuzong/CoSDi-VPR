# ----------------------------------------------------------------------------
# Progressive Structural Distillation (PSD)
# Iterative Token Filtering for Appearance-Invariant Place Recognition
# ----------------------------------------------------------------------------
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenFilter(nn.Module):
    """Per-iteration learned token filter.

    Takes token features + max slot claim as input,
    outputs keep probability for each token.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        # input: token feature (in_dim) + max_claim (1)
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, in_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor, max_claim: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D) token features
        max_claim: (B, N, 1) max attention claim per token
        Returns: (B, N, 1) keep probability
        """
        inp = torch.cat([x, max_claim], dim=-1)
        return torch.sigmoid(self.net(inp))


class PSDBlock(nn.Module):
    """
    Progressive Structural Distillation Block.

    Key differences from standard Slot Attention:
    1. Progressive token filtering: each iteration prunes non-discriminative tokens
    2. Residual update instead of GRU (no reconstruction objective)
    3. Slot-token co-refinement: slots inform filtering, filtering improves slots
    """

    def __init__(
            self,
            in_dim: int,
            num_queries: int,
            nheads: int = 8,
            iters: int = 3,
            mlp_ratio: float = 2.0,
            use_pos_embed: bool = False,
            eps: float = 1e-8,
    ):
        super().__init__()
        assert in_dim % nheads == 0, "in_dim must be divisible by nheads"
        self.in_dim = in_dim
        self.num_queries = num_queries
        self.nheads = nheads
        self.head_dim = in_dim // nheads
        self.iters = iters
        self.use_pos_embed = use_pos_embed
        self.eps = eps

        # Layer norms
        self.norm_inputs = nn.LayerNorm(in_dim)
        self.norm_slots = nn.LayerNorm(in_dim)
        self.norm_update = nn.LayerNorm(in_dim)
        self.norm_mlp = nn.LayerNorm(in_dim)

        # Linear projections
        self.to_q = nn.Linear(in_dim, in_dim, bias=False)
        self.to_k = nn.Linear(in_dim, in_dim, bias=False)
        self.to_v = nn.Linear(in_dim, in_dim, bias=False)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, in_dim))

        # Temperature
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.5)))
        self.tau_min, self.tau_max = 0.05, 1.0

        # MLP refinement (residual)
        hidden = int(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )

        # Position embedding
        if use_pos_embed:
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, in_dim),
            )

        # Progressive token filters: one per iteration (except the last)
        self.filters = nn.ModuleList([
            TokenFilter(in_dim) for _ in range(max(iters - 1, 1))
        ])

        self.scale = self.head_dim ** -0.5

    def _mh_project(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        B, L, D = x.shape
        y = proj(x)
        y = y.view(B, L, self.nheads, self.head_dim).transpose(1, 2)
        return y

    def _mh_merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        x: (B, N, D)
        pos: (B, N, 2) optional
        Returns: slots (B, Q, D), attn_last (B, Q, N)
        """
        B, N, D = x.shape
        assert D == self.in_dim

        # Slot initialization
        slots = self.queries.repeat(B, 1, 1)  # (B, Q, D)
        if self.training:
            slots = slots + 0.05 * torch.randn_like(slots)

        # Position encoding
        if self.use_pos_embed and pos is not None:
            pos_emb = self.pos_mlp(pos)
            x_kv = x + pos_emb
        else:
            x_kv = x
        x_norm = self.norm_inputs(x_kv)

        # Project K, V once
        k = self._mh_project(x_norm, self.to_k)  # (B, H, N, Dh)
        v = self._mh_project(x_norm, self.to_v)  # (B, H, N, Dh)

        # Token mask: starts as all-ones, progressively filters
        token_mask = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)

        attn_last = None

        for it in range(self.iters):
            slots_prev = slots

            # Normalize slots and project to Q
            s_norm = self.norm_slots(slots)
            q = self._mh_project(s_norm, self.to_q)  # (B, H, Q, Dh)
            q = q * self.scale

            # Apply token mask to K, V
            # mask shape: (B, N, 1) -> (B, 1, N, 1) for broadcasting with (B, H, N, Dh)
            mask_expanded = token_mask.unsqueeze(1)  # (B, 1, N, 1)
            k_masked = k * mask_expanded
            v_masked = v * mask_expanded

            # Attention: [B, H, N, Dh] @ [B, H, Dh, Q] -> [B, H, N, Q]
            attn_logits = torch.matmul(k_masked, q.transpose(-2, -1))

            # Competitive assignment (softmax over slots)
            tau = self.log_tau.exp().clamp(self.tau_min, self.tau_max)
            attn_ = F.softmax(attn_logits / tau, dim=-1)  # (B, H, N, Q)

            # Column normalization (weighted by token mask)
            attn = attn_ + self.eps
            attn = attn / (attn.sum(dim=-2, keepdim=True) + self.eps)

            # Slot update: weighted average of values
            # [B, H, Q, N] @ [B, H, N, Dh] -> [B, H, Q, Dh]
            updates = torch.matmul(attn.transpose(-2, -1), v_masked)
            updates = self._mh_merge(updates)  # (B, Q, D)

            # Residual update with LayerNorm (no GRU)
            slots = self.norm_update(slots_prev + updates)

            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))

            # Store attention for output
            attn_last = attn_.mean(dim=1).transpose(-2, -1)  # (B, Q, N)

            # Progressive token filtering (not on last iteration)
            if it < self.iters - 1:
                # Max claim: how strongly is each token claimed by its best slot
                # attn_: (B, H, N, Q) -> mean over heads -> max over slots
                max_claim = attn_.mean(dim=1).max(dim=-1).values  # (B, N)
                max_claim = max_claim.unsqueeze(-1)  # (B, N, 1)

                # Learned filter decides keep probability
                filter_idx = min(it, len(self.filters) - 1)
                keep_prob = self.filters[filter_idx](x_kv, max_claim)  # (B, N, 1)

                # Accumulate mask (multiplicative)
                token_mask = token_mask * keep_prob

        return slots, attn_last


class CoSDi(nn.Module):

    def __init__(
            self,
            in_channels: int = 1024,
            proj_channels: int = 512,
            num_queries: int = 32,
            num_layers: int = 2,
            row_dim: int = 32,
            iters: int = 3,
            use_pos_embed: bool = True,
    ):
        super().__init__()
        self.proj_c = nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = nn.LayerNorm(proj_channels)
        self.num_queries = num_queries
        in_dim = proj_channels
        nheads = max(1, in_dim // 64)

        self.blocks = PSDBlock(
            in_dim=in_dim,
            num_queries=self.num_queries,
            nheads=nheads,
            iters=iters,
            use_pos_embed=use_pos_embed,
        )

        self.slot_interact = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=4,
            dim_feedforward=in_dim * 2,
            batch_first=True
        )
        self.fc = nn.Linear(in_dim, in_dim // 2)
        self.in_dim = in_dim
        self.use_pos_embed = use_pos_embed

    @staticmethod
    def _make_pos_grid(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).view(1, H * W, 2)
        return grid.repeat(B, 1, 1)

    def forward(self, x: torch.Tensor, return_feats=False):
        """
        x: (B, C, H, W)
        """
        x = self.proj_c(x)  # (B, D, H, W)
        B, D, H, W = x.shape

        pos = None
        if self.use_pos_embed:
            pos = self._make_pos_grid(B, H, W, device=x.device, dtype=x.dtype)

        x = x.flatten(2).permute(0, 2, 1)  # (B, N, D)
        x = self.norm_input(x)
        slots, attn = self.blocks(x, pos=pos)
        slots = self.slot_interact(slots)
        out = self.fc(slots).flatten(1)
        out = F.normalize(out, p=2, dim=-1)
        if return_feats:
            return out, slots, attn, (H, W)
        return out, slots, attn, (H, W)


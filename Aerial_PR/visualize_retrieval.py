"""
Retrieval visualization for U1652 (drone -> satellite).

For 5 randomly chosen query buildings, retrieve top-K gallery satellites and
render ONE figure where each image (query + retrieved) carries a combined
slot-attention heatmap overlay (NOT one image per slot — slots are aggregated
into a single descriptor-support map).

Usage:
    python visualize_retrieval.py \
        --ckpt logs/.../epoch[xx]_d2sR@1[...].ckpt \
        --test_path data/University-Release/test \
        --out_dir vis_retrieval

Layout:
    Row i  (5 rows total):
        [Query (drone)]  [Top-1 sat]  [Top-2 sat]  [Top-3 sat]
        ↑ green border = correct (same building),  red = wrong
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2 as T

from src.backbones import DinoV2
from src.boq import BoQ
from src.model import BoQModel
from src.dataloaders.University1652Val import University1652ValDataset


# --------------------------------------------------------------------------- #
# Model loading (matches eval_u1652.py)
# --------------------------------------------------------------------------- #
def build_model(ckpt_path, device,
                channel_proj=512, num_queries=20, num_layers=1,
                output_dim=6144, unfreeze_n_blocks=3, use_pos_embed=False):
    backbone = DinoV2(backbone_name="dinov2_vitb14",
                      unfreeze_n_blocks=unfreeze_n_blocks)
    aggregator = BoQ(
        in_channels=backbone.out_channels,
        proj_channels=channel_proj, num_queries=num_queries,
        num_layers=num_layers, row_dim=output_dim // channel_proj,
        use_pos_embed=use_pos_embed,
    )
    model = BoQModel(backbone, aggregator)
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


# --------------------------------------------------------------------------- #
# Combined slot-attention -> single heatmap per image
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _replay_psdblock(block, x_kv):
    """
    Replay PSDBlock iterations to recover the final token_mask in addition to
    the standard outputs. Mirrors run_capture() in visualize_single_iter.py
    but only keeps the final-state attn + token_mask.
    """
    x_norm = block.norm_inputs(x_kv)
    k = block._mh_project(x_norm, block.to_k)
    v = block._mh_project(x_norm, block.to_v)

    Q = block.num_queries
    B, N, _ = x_kv.shape
    slots = block.queries.repeat(B, 1, 1)
    token_mask = torch.ones(B, N, 1, device=x_kv.device, dtype=x_kv.dtype)
    attn_last = None

    for it in range(block.iters):
        s_norm = block.norm_slots(slots)
        q = block._mh_project(s_norm, block.to_q) * block.scale

        mask_expanded = token_mask.unsqueeze(1)
        k_masked = k * mask_expanded
        v_masked = v * mask_expanded

        attn_logits = torch.matmul(k_masked, q.transpose(-2, -1))
        tau = block.log_tau.exp().clamp(block.tau_min, block.tau_max)
        attn_ = torch.softmax(attn_logits / tau, dim=-1)

        attn_norm = attn_ + block.eps
        attn_norm = attn_norm / (attn_norm.sum(dim=-2, keepdim=True) + block.eps)

        updates = torch.matmul(attn_norm.transpose(-2, -1), v_masked)
        updates = block._mh_merge(updates)
        slots = block.norm_update(slots + updates)
        slots = slots + block.mlp(block.norm_mlp(slots))

        attn_last = attn_.mean(dim=1).transpose(-2, -1)   # (B, Q, N)

        if it < block.iters - 1:
            max_claim = attn_.mean(dim=1).max(dim=-1).values.unsqueeze(-1)
            filter_idx = min(it, len(block.filters) - 1)
            keep_prob = block.filters[filter_idx](x_kv, max_claim)
            token_mask = token_mask * keep_prob

    return slots, attn_last, token_mask.squeeze(-1)   # (B, N)


@torch.no_grad()
def descriptor_support_map(model, img_tensor):
    """
    Aggregate slot attention weighted by per-slot descriptor norm, token
    feature strength, AND the iterative token mask (which suppresses
    boundary/background tokens the model itself filtered out).

    Returns: (Hf, Wf) numpy heatmap in [0, 1].
    """
    bb = model.backbone
    agg = model.aggregator

    feats = bb(img_tensor)
    conv_out = agg.proj_c(feats)
    _, D, Hf, Wf = conv_out.shape
    token_strength = conv_out.norm(dim=1).squeeze(0).cpu().numpy()   # (Hf, Wf)

    x = conv_out.flatten(2).permute(0, 2, 1)
    x = agg.norm_input(x)

    pos = (agg._make_pos_grid(1, Hf, Wf, device=x.device, dtype=x.dtype)
           if agg.use_pos_embed else None)
    block = agg.blocks
    if block.use_pos_embed and pos is not None:
        x_kv = x + block.pos_mlp(pos)
    else:
        x_kv = x

    slots, attn_last, token_mask = _replay_psdblock(block, x_kv)
    slots = agg.slot_interact(slots)
    desc_per_slot = agg.fc(slots).squeeze(0)
    slot_desc_norm = desc_per_slot.norm(dim=-1).cpu().numpy()         # (Q,)

    Q = attn_last.size(1)
    attn_qhw = attn_last.squeeze(0).reshape(Q, Hf, Wf).cpu().numpy()
    mask_2d = token_mask.squeeze(0).reshape(Hf, Wf).cpu().numpy()     # (Hf, Wf)

    weighted = (attn_qhw
                * slot_desc_norm[:, None, None]
                * token_strength[None, :, :]
                * mask_2d[None, :, :])
    heat = weighted.sum(axis=0)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    return heat


# --------------------------------------------------------------------------- #
# Rendering helpers (lifted from visualize_single_iter.py, simplified)
# --------------------------------------------------------------------------- #
def _smooth(arr2d, sigma=0.7):
    if sigma <= 0:
        return arr2d
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1 = torch.exp(-(x ** 2) / (2 * sigma ** 2)); k1 = k1 / k1.sum()
    k2d = (k1[:, None] * k1[None, :])[None, None]
    t = torch.tensor(arr2d, dtype=torch.float32)[None, None]
    t = F.pad(t, (radius,) * 4, mode="reflect")
    t = F.conv2d(t, k2d)
    return t[0, 0].numpy()


def _upsample(arr2d, H_im, W_im, mode="bicubic"):
    return F.interpolate(
        torch.tensor(arr2d, dtype=torch.float32)[None, None],
        size=(H_im, W_im), mode=mode, align_corners=False,
    )[0, 0].numpy()


def _denorm_to_display(img_tensor):
    """img_tensor: (C, H, W) normalized -> (H, W, 3) in [0,1]."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.clip(img * std + mean, 0, 1)


def _overlay(ax, img_disp, heat, title=None, border_color=None,
             alpha_max=0.85, gamma=0.6, pct_lo=10, pct_hi=99):
    """Overlay heat with percentile contrast stretch + gamma boost."""
    H_im, W_im = img_disp.shape[:2]
    h_smooth = _smooth(heat, sigma=0.7)
    # Percentile stretch — make weak peaks visible without saturating
    lo = float(np.percentile(h_smooth, pct_lo))
    hi = float(np.percentile(h_smooth, pct_hi))
    h_stretch = np.clip((h_smooth - lo) / (hi - lo + 1e-8), 0, 1)
    # Gamma < 1 lifts mid-tones
    h_boost = np.power(h_stretch, gamma)
    h_up = np.clip(_upsample(h_boost, H_im, W_im), 0, 1)
    ax.imshow(img_disp)
    rgba = plt.get_cmap("jet")(h_up)
    rgba[..., 3] = h_up * alpha_max
    ax.imshow(rgba, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(3)
        sp.set_edgecolor(border_color or "black")
    if title:
        ax.set_title(title, fontsize=10)


# --------------------------------------------------------------------------- #
# Main viz routine
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_path", required=True)
    p.add_argument("--out_dir", default="vis_retrieval")
    p.add_argument("--img_size", type=int, default=252)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--direction", default="d2s", choices=["d2s", "s2d"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build model + dataset ------------------------------------------------
    print(f"[1/4] Loading model from {args.ckpt}")
    model = build_model(args.ckpt, device)

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size), interpolation=3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = University1652ValDataset(args.test_path, direction=args.direction,
                                  transform=transform)
    print(f"      direction={args.direction}  Q={ds.num_queries}  G={ds.num_references}")

    # 2. Extract descriptors --------------------------------------------------
    print("[2/4] Extracting descriptors")
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)
    feats = []
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(device, non_blocking=True)
            desc, _, _, _ = model(img)
            feats.append(desc.float().cpu())
    feats = torch.cat(feats, dim=0)
    g_feats = F.normalize(feats[:ds.num_references], dim=-1)
    q_feats = F.normalize(feats[ds.num_references:], dim=-1)

    # 3. Pick 5 queries (one per distinct building) and rank gallery ----------
    print(f"[3/4] Selecting {args.n_samples} query buildings")
    query_buildings = list(dict.fromkeys(ds.query_ids))   # unique, ordered
    chosen_bids = random.sample(query_buildings, args.n_samples)
    chosen_q_idx = [ds.query_ids.index(b) for b in chosen_bids]

    rows = []
    for q_idx in chosen_q_idx:
        sim = q_feats[q_idx] @ g_feats.t()                 # (G,)
        top = sim.topk(args.top_k).indices.tolist()
        rows.append({"q_idx": q_idx, "top": top,
                     "qid": ds.query_ids[q_idx]})

    # 4. Render figure --------------------------------------------------------
    print(f"[4/4] Rendering {args.n_samples}x{args.top_k+1} figure")
    nrows = args.n_samples
    ncols = 1 + args.top_k
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.4 * ncols, 2.5 * nrows),
        gridspec_kw={"hspace": 0.18, "wspace": 0.05},
    )
    if nrows == 1:
        axes = axes[None, :]

    for r, row in enumerate(rows):
        # ---- Query ----
        q_path = ds.query_paths[row["q_idx"]]
        q_tensor = transform(read_image(str(q_path))).unsqueeze(0).to(device)
        q_disp = _denorm_to_display(q_tensor[0])
        q_heat = descriptor_support_map(model, q_tensor)
        _overlay(axes[r, 0], q_disp, q_heat,
                 title=f"Query  (bid={row['qid']})" if r == 0 else f"bid={row['qid']}",
                 border_color="blue")

        # ---- Top-K retrieved gallery ----
        for k in range(args.top_k):
            g_idx = row["top"][k]
            g_path = ds.gallery_paths[g_idx]
            g_bid = ds.gallery_ids[g_idx]
            correct = (g_bid == row["qid"])
            border = "lime" if correct else "red"

            g_tensor = transform(read_image(str(g_path))).unsqueeze(0).to(device)
            g_disp = _denorm_to_display(g_tensor[0])
            g_heat = descriptor_support_map(model, g_tensor)

            mark = "✓" if correct else "✗"
            ttl = (f"Top-{k+1}  bid={g_bid}  {mark}"
                   if r == 0 else f"bid={g_bid}  {mark}")
            _overlay(axes[r, 1 + k], g_disp, g_heat,
                     title=ttl, border_color=border)

    fig.suptitle(
        f"U1652  {args.direction}  retrieval visualization "
        f"(combined slot attention; blue=query, green=correct, red=wrong)",
        fontsize=12, y=0.995,
    )
    save_path = out_dir / f"retrieval_{args.direction}_n{args.n_samples}_k{args.top_k}.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()

"""
Visualize last-iteration slot attention + cumulative filter (token mask)
for retrieval pairs (query + top-1 retrieved).

Layout:
    Row i (one building, 5 rows total):
        [Original Q] [Q slot (last iter)] [Q filter (cumulative)]
        [Original R] [R slot (last iter)] [R filter (cumulative)]
    → 5 buildings × 2 sub-rows × 3 cols  (or 10 rows × 3 cols)

Usage:
    python visualize_slot_filter.py \
        --ckpt logs/.../epoch[xx]_d2sR@1[...].ckpt \
        --test_path data/University-Release/test
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
@torch.no_grad()
def replay_capture(model, img_tensor):
    """
    Replay BoQ + PSDBlock and capture for the LAST iteration:
      - combined slot heatmap (sum over slots of attn × ‖desc‖ × ‖tok‖ × mask)
      - cumulative token_mask  (final, after all filtering iters)
    Plus token_strength for scaling.

    Returns dict with: combined_slot, token_mask_2d, Hf, Wf.
    """
    bb = model.backbone
    agg = model.aggregator
    block = agg.blocks

    feats = bb(img_tensor)
    conv_out = agg.proj_c(feats)
    _, D, Hf, Wf = conv_out.shape
    token_strength = conv_out.norm(dim=1).squeeze(0).cpu().numpy()       # (Hf, Wf)

    x = conv_out.flatten(2).permute(0, 2, 1)
    x = agg.norm_input(x)
    pos = (agg._make_pos_grid(1, Hf, Wf, device=x.device, dtype=x.dtype)
           if agg.use_pos_embed else None)
    if block.use_pos_embed and pos is not None:
        x_kv = x + block.pos_mlp(pos)
    else:
        x_kv = x

    x_norm = block.norm_inputs(x_kv)
    k = block._mh_project(x_norm, block.to_k)
    v = block._mh_project(x_norm, block.to_v)

    Q = block.num_queries
    B, N, _ = x_kv.shape
    slots = block.queries.repeat(B, 1, 1)
    token_mask = torch.ones(B, N, 1, device=x_kv.device, dtype=x_kv.dtype)
    last_attn = None

    for it in range(block.iters):
        s_norm = block.norm_slots(slots)
        q = block._mh_project(s_norm, block.to_q) * block.scale
        m_exp = token_mask.unsqueeze(1)
        attn_logits = torch.matmul(k * m_exp, q.transpose(-2, -1))
        tau = block.log_tau.exp().clamp(block.tau_min, block.tau_max)
        attn_ = torch.softmax(attn_logits / tau, dim=-1)
        attn_norm = attn_ + block.eps
        attn_norm = attn_norm / (attn_norm.sum(dim=-2, keepdim=True) + block.eps)

        updates = torch.matmul(attn_norm.transpose(-2, -1), v * m_exp)
        updates = block._mh_merge(updates)
        slots = block.norm_update(slots + updates)
        slots = slots + block.mlp(block.norm_mlp(slots))

        last_attn = attn_.mean(dim=1).transpose(-2, -1)   # (B, Q, N)

        if it < block.iters - 1:
            max_claim = attn_.mean(dim=1).max(dim=-1).values.unsqueeze(-1)
            filter_idx = min(it, len(block.filters) - 1)
            keep_prob = block.filters[filter_idx](x_kv, max_claim)
            token_mask = token_mask * keep_prob

    # Final descriptor support combination
    slots = agg.slot_interact(slots)
    slot_desc_norm = agg.fc(slots).squeeze(0).norm(dim=-1).cpu().numpy()    # (Q,)
    attn_qhw = last_attn.squeeze(0).reshape(Q, Hf, Wf).cpu().numpy()
    mask_2d = token_mask.squeeze(0).squeeze(-1).reshape(Hf, Wf).cpu().numpy()

    weighted = (attn_qhw
                * slot_desc_norm[:, None, None]
                * token_strength[None, :, :]
                * mask_2d[None, :, :])
    combined = weighted.sum(axis=0)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

    return {
        "combined_slot": combined,
        "token_mask": mask_2d,           # values in [0,1], 1=keep, ~0=drop
        "Hf": Hf, "Wf": Wf,
    }


# --------------------------------------------------------------------------- #
def _smooth(arr2d, sigma=0.7):
    if sigma <= 0:
        return arr2d
    r = max(1, int(3 * sigma))
    xs = torch.arange(-r, r + 1, dtype=torch.float32)
    k1 = torch.exp(-(xs ** 2) / (2 * sigma ** 2)); k1 = k1 / k1.sum()
    k2d = (k1[:, None] * k1[None, :])[None, None]
    t = torch.tensor(arr2d, dtype=torch.float32)[None, None]
    t = F.pad(t, (r,) * 4, mode="reflect")
    return F.conv2d(t, k2d)[0, 0].numpy()


def _upsample(arr, H_im, W_im, mode="bicubic"):
    return F.interpolate(
        torch.tensor(arr, dtype=torch.float32)[None, None],
        size=(H_im, W_im), mode=mode, align_corners=False,
    )[0, 0].numpy()


def _denorm(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    return np.clip(img * std + mean, 0, 1)


def _slot_overlay(ax, img_disp, heat, alpha_max=0.85, gamma=0.6):
    H_im, W_im = img_disp.shape[:2]
    h = _smooth(heat, 0.7)
    lo, hi = float(np.percentile(h, 10)), float(np.percentile(h, 99))
    h = np.clip((h - lo) / (hi - lo + 1e-8), 0, 1) ** gamma
    h_up = np.clip(_upsample(h, H_im, W_im), 0, 1)
    ax.imshow(img_disp)
    rgba = plt.get_cmap("jet")(h_up); rgba[..., 3] = h_up * alpha_max
    ax.imshow(rgba, interpolation="bilinear")


def _filter_overlay(ax, img_disp, mask, alpha=0.75):
    """Show cumulative keep_prob: green = kept, red = dropped."""
    H_im, W_im = img_disp.shape[:2]
    drop = np.clip(1.0 - mask, 0, 1)            # 1 = fully dropped
    drop = _smooth(drop, 0.7)
    drop_up = np.clip(_upsample(drop, H_im, W_im), 0, 1)
    ax.imshow(img_disp, alpha=0.55)
    ax.imshow(drop_up, cmap="RdYlGn_r", alpha=alpha,
              vmin=0, vmax=1, interpolation="bilinear")


def _strip(ax, border=None, lw=3):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(lw)
        sp.set_edgecolor(border or "black")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_path", required=True)
    p.add_argument("--out_dir", default="vis_retrieval")
    p.add_argument("--img_size", type=int, default=252)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--direction", default="d2s", choices=["d2s", "s2d"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"[3/4] Selecting {args.n_samples} buildings")
    qbs = list(dict.fromkeys(ds.query_ids))
    chosen_bids = random.sample(qbs, args.n_samples)
    samples = []
    for bid in chosen_bids:
        q_idx = ds.query_ids.index(bid)
        sim = q_feats[q_idx] @ g_feats.t()
        g_idx = int(sim.argmax().item())
        samples.append({"qid": bid,
                        "q_path": ds.query_paths[q_idx],
                        "g_path": ds.gallery_paths[g_idx],
                        "g_bid": ds.gallery_ids[g_idx]})

    print(f"[4/4] Rendering {args.n_samples}x6 figure (Q | Q-slot | Q-filter | R | R-slot | R-filter)")
    nrows = args.n_samples
    ncols = 6
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.4 * ncols, 2.5 * nrows),
                             gridspec_kw={"hspace": 0.18, "wspace": 0.05})
    if nrows == 1:
        axes = axes[None, :]

    col_titles = ["Query", "Q  slot (last iter)", "Q  filter (cumulative)",
                  "Top-1", "R  slot (last iter)", "R  filter (cumulative)"]

    for r, s in enumerate(samples):
        # --- Query side ---
        q_t = transform(read_image(str(s["q_path"]))).unsqueeze(0).to(device)
        q_d = _denorm(q_t[0])
        q_state = replay_capture(model, q_t)

        axes[r, 0].imshow(q_d); _strip(axes[r, 0], "blue")
        _slot_overlay(axes[r, 1], q_d, q_state["combined_slot"])
        _strip(axes[r, 1], "blue")
        _filter_overlay(axes[r, 2], q_d, q_state["token_mask"])
        _strip(axes[r, 2], "blue")
        axes[r, 0].set_ylabel(f"bid={s['qid']}", fontsize=10, fontweight="bold")

        # --- Retrieved gallery side ---
        g_t = transform(read_image(str(s["g_path"]))).unsqueeze(0).to(device)
        g_d = _denorm(g_t[0])
        g_state = replay_capture(model, g_t)

        correct = (s["g_bid"] == s["qid"])
        bd = "lime" if correct else "red"
        mark = "✓" if correct else "✗"
        axes[r, 3].imshow(g_d); _strip(axes[r, 3], bd)
        axes[r, 3].set_xlabel(f"bid={s['g_bid']} {mark}", fontsize=9)
        _slot_overlay(axes[r, 4], g_d, g_state["combined_slot"])
        _strip(axes[r, 4], bd)
        _filter_overlay(axes[r, 5], g_d, g_state["token_mask"])
        _strip(axes[r, 5], bd)

        if r == 0:
            for c, t in enumerate(col_titles):
                axes[0, c].set_title(t, fontsize=10)

    fig.suptitle(
        f"U1652 {args.direction}  —  last-iter slot heatmap + cumulative filter "
        f"(blue=query, green=correct, red=wrong)",
        fontsize=12, y=0.995,
    )
    out = out_dir / f"slot_filter_{args.direction}_n{args.n_samples}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

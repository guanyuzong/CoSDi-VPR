"""
Standalone SUES-200 evaluation: R@1/5/10 + mAP, both directions, all 4 heights.

Usage:
    python eval_sues.py --ckpt path/to/best.ckpt \
        --test_path data/SUES-200-512x512-V2/SUES-200-512x512
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from src.backbones import DinoV2
from src.cosdi import CoSDi
from src.model import CoSDiModel
from src.dataloaders.SUES200Val import SUES200ValDataset


@torch.no_grad()
def extract(model, loader, device):
    feats = []
    for img, _ in loader:
        img = img.to(device, non_blocking=True)
        desc, _, _, _ = model(img)
        feats.append(desc.float().cpu())
    return torch.cat(feats, dim=0)


def compute_metrics(sim: torch.Tensor, query_ids, gallery_ids, k_values=(1, 5, 10)):
    Nq, Ng = sim.shape
    indices = sim.argsort(dim=1, descending=True)
    g_ids = torch.tensor([hash(g) for g in gallery_ids])
    q_ids = torch.tensor([hash(q) for q in query_ids])
    matches = (g_ids[indices] == q_ids[:, None])

    out = {}
    for k in k_values:
        out[f"R@{k}"] = matches[:, :k].any(dim=1).float().mean().item()

    pos_mask = matches.float()
    num_pos = pos_mask.sum(dim=1).clamp(min=1)
    cum_hits = pos_mask.cumsum(dim=1)
    ranks = torch.arange(1, Ng + 1, dtype=torch.float32)
    precisions = cum_hits / ranks
    ap = (precisions * pos_mask).sum(dim=1) / num_pos
    out["AP"] = ap.mean().item()
    return out


def evaluate(model, test_path, height, direction, img_size, batch_size, num_workers, device,
             num_train, num_total):
    transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = SUES200ValDataset(test_path, height=height, direction=direction,
                           transform=transform, num_train=num_train, num_total=num_total)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    feats = extract(model, loader, device)
    g_feats = F.normalize(feats[:ds.num_references], dim=-1)
    q_feats = F.normalize(feats[ds.num_references:], dim=-1)

    chunk = 512
    sims = []
    for i in range(0, q_feats.size(0), chunk):
        sims.append(q_feats[i:i + chunk] @ g_feats.t())
    sim = torch.cat(sims, dim=0)
    return compute_metrics(sim, ds.query_ids, ds.gallery_ids)


def build_model(ckpt_path, device, use_pos_embed=False,
                channel_proj=512, num_queries=20, num_layers=1, output_dim=6144,
                unfreeze_n_blocks=3):
    backbone = DinoV2(backbone_name="dinov2_vitb14", unfreeze_n_blocks=unfreeze_n_blocks)
    aggregator = CoSDi(
        in_channels=backbone.out_channels,
        proj_channels=channel_proj, num_queries=num_queries,
        num_layers=num_layers, row_dim=output_dim // channel_proj,
        use_pos_embed=use_pos_embed,
    )
    model = CoSDiModel(backbone, aggregator)
    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_path", required=True,
                    help="Folder containing satellite-view/ and drone_view_512/")
    ap.add_argument("--img_size", type=int, default=252)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--use_pos_embed", action="store_true")
    ap.add_argument("--num_train", type=int, default=120)
    ap.add_argument("--num_total", type=int, default=200)
    ap.add_argument("--heights", type=int, nargs="+", default=[150, 200, 250, 300])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.ckpt, device, use_pos_embed=args.use_pos_embed)

    results = {}
    for h in args.heights:
        for d in ("d2s", "s2d"):
            print(f"\n>>> height={h}m  {d}")
            results[(h, d)] = evaluate(
                model, args.test_path, h, d,
                args.img_size, args.batch_size, args.num_workers, device,
                args.num_train, args.num_total,
            )

    print("\n" + "=" * 72)
    print("SUES-200 Results")
    print("=" * 72)
    print(f"{'Height':<8}{'Dir':<6}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'AP':>8}")
    print("-" * 72)
    for (h, d), m in results.items():
        print(f"{h:<8}{d:<6}{m['R@1']*100:>8.2f}{m['R@5']*100:>8.2f}"
              f"{m['R@10']*100:>8.2f}{m['AP']*100:>8.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

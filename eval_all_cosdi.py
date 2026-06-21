import argparse, csv, glob, os, sys
import torch
from torchvision.transforms import v2 as T

COSDI_DIR = os.path.dirname(os.path.abspath(__file__))
if COSDI_DIR not in sys.path:
    sys.path.insert(0, COSDI_DIR)

from src.cosdi import CoSDi
from src.backbones import DinoV2
import test as cosdi_test   # CoSDi's test.py: evaluate_dataset, get_val_img_size, dataset classes

DATA = "/home/arc/data/Zong/Bag-of-Queries-main-/Bag-of-Queries-main/data"  # must contain images
_EMB2NAME = {384: "dinov2_vits14", 768: "dinov2_vitb14",
             1024: "dinov2_vitl14", 1536: "dinov2_vitg14"}


def build_model(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    in_ch = int(sd["aggregator.proj_c.weight"].shape[1])
    pc = int(sd["aggregator.proj_c.weight"].shape[0])
    nq = int(sd["aggregator.blocks.queries"].shape[1])
    name = _EMB2NAME.get(in_ch, "dinov2_vitb14")
    bb = DinoV2(backbone_name=name, unfreeze_n_blocks=2)
    agg = CoSDi(in_channels=in_ch, proj_channels=pc, num_queries=nq)
    bb.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("backbone.")}, strict=False)
    agg.load_state_dict({k[11:]: v for k, v in sd.items() if k.startswith("aggregator.")}, strict=False)
    print(f"[CoSDi] backbone={name} in_ch={in_ch} proj={pc} num_queries={nq}")
    return bb.to(device).eval(), agg.to(device).eval(), name


def main():
    cks = glob.glob(os.path.join(COSDI_DIR, "*.ckpt"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=(cks[0] if cks else None))
    ap.add_argument("--data_root", default=DATA)
    ap.add_argument("--out", default="cosdi_results.csv")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()

    dev = torch.device("cuda")
    bb, agg, name = build_model(a.ckpt, dev)
    sz = cosdi_test.get_val_img_size(name)            # (322,322)
    tf = T.Compose([                                   # == test.py val_transform (BICUBIC)
        T.Resize(sz, interpolation=3),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    R = a.data_root
    C = cosdi_test  # dataset classes live in test.py's namespace
    specs = [
        ("Tokyo24/7",       lambda: C.Tokyo247Dataset(dataset_path=f"{R}/tokyo247", transform=tf)),
        ("MSLS-val",        lambda: C.MapillarySLSDataset(dataset_path=f"{R}/msls-val", transform=tf)),
        ("Pitts30k",        lambda: C.PittsburghDatasetTest(dataset_path=f"{R}/Pittsburgh", split="pitts30k-test", transform=tf)),
        ("Nordland(+/-1)",  lambda: C.NordlandDataset(dataset_path=f"{R}/Nordland", transform=tf)),
        ("Nordland(+/-10)", lambda: C.NordlandDataset(dataset_path=f"{R}/Nordland_full", transform=tf)),
        ("SPED",            lambda: C.SPEDDataset(dataset_path=f"{R}/SPED", transform=tf)),
        ("AmsterTime",      lambda: C.AmsterTimeDataset(dataset_path=f"{R}/amstertime", transform=tf)),
        ("Baidu",           lambda: C.BaiduDataset(dataset_path=f"{R}/baidu", transform=tf)),
        ("SVOX-night",      lambda: C.SVOXDataset(dataset_path=f"{R}/svox", condition="night", transform=tf)),
        ("SVOX-overcast",   lambda: C.SVOXDataset(dataset_path=f"{R}/svox", condition="overcast", transform=tf)),
        ("SVOX-rain",       lambda: C.SVOXDataset(dataset_path=f"{R}/svox", condition="rain", transform=tf)),
        ("SVOX-snow",       lambda: C.SVOXDataset(dataset_path=f"{R}/svox", condition="snow", transform=tf)),
        ("SVOX-sun",        lambda: C.SVOXDataset(dataset_path=f"{R}/svox", condition="sun", transform=tf)),
    ]
    rows = []
    for label, make in specs:
        print(f"\n==> {label}")
        try:
            ds = make()
            rec = cosdi_test.evaluate_dataset(ds, bb, agg, dev, a.batch_size, a.num_workers,
                                              show_progress=True, save_slot_attn=False, pca_dim=0)
            r1, r5, r10 = rec.get(1, 0) * 100, rec.get(5, 0) * 100, rec.get(10, 0) * 100
            print(f"   {label}: R@1={r1:.1f}  R@5={r5:.1f}  R@10={r10:.1f}")
            rows.append([label, ds.num_references, ds.num_queries, f"{r1:.1f}", f"{r5:.1f}", f"{r10:.1f}"])
        except Exception as e:
            print(f"   [warn] {label}: {e}")
            rows.append([label, "", "", "ERR", "ERR", "ERR"])

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "num_db", "num_q", "R@1", "R@5", "R@10"]); w.writerows(rows)
    print(f"\n[done] saved {a.out}")
    for r in rows: print("  ", r)


if __name__ == "__main__":
    main()

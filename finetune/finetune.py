"""Fine-tune best_model.pth on Phoenix ADE+SAM pseudo-labels.

Matches upstream exactly: DeepLabV3-ResNet50, 8-class head (idx 0 background +
7 classes), Resize((900,900)) + ToTensor, NO normalization, argmax dim=1.

Trains the existing best_model.pth on the refined pseudo-label masks (our 7
classes -> model indices 1..7; background 0 kept as-is and ignored in loss).
Holds out a test split, reports per-class IoU before vs after, saves the
fine-tuned checkpoint.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

OUR = ["Building", "Sky", "Vegetation", "Road", "Waterbody", "Vehicle", "Terrain"]
COLORS = np.array([[255,255,0],[0,0,255],[0,255,0],[170,170,170],[96,25,134],[253,167,206],[255,0,0]], dtype=np.uint8)
RES = 900


def color_to_idx(mask_rgb: np.ndarray) -> np.ndarray:
    """Map a colorized refined mask (our 7 colors) -> model index 1..7 (0=bg)."""
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.int64)  # 0 = background/ignore
    for c in range(7):
        m = np.all(mask_rgb == COLORS[c], axis=-1)
        out[m] = c + 1  # model index: 1..7
    return out


class DS:
    def __init__(self, frames, masks, transform):
        self.frames, self.masks, self.t = frames, masks, transform
    def __len__(self): return len(self.frames)
    def __getitem__(self, i):
        import torch
        img = Image.open(self.frames[i]).convert("RGB")
        x = self.t(img)
        mrgb = np.asarray(Image.open(self.masks[i]).convert("RGB").resize((RES, RES), Image.NEAREST))
        y = torch.from_numpy(color_to_idx(mrgb))
        return x, y


def iou_report(model, loader, dev):
    import torch
    inter = np.zeros(8); union = np.zeros(8)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev); pred = model(x)["out"].argmax(1).cpu().numpy()
            y = y.numpy()
            for c in range(8):
                p = pred == c; g = y == c
                inter[c] += np.logical_and(p, g).sum()
                union[c] += np.logical_or(p, g).sum()
    iou = inter / np.maximum(union, 1)
    return {(["bg"] + OUR)[c]: round(float(iou[c]), 3) for c in range(8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True, help="dir with <stem>__refined.png")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    import torch, torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = transforms.Compose([transforms.Resize((RES, RES)), transforms.ToTensor()])  # NO normalize

    # pair frames with refined masks
    frames, masks = [], []
    for fp in sorted(args.frames.glob("*.jpg")):
        mp = args.masks / f"{fp.stem}__refined.png"
        if mp.exists():
            frames.append(fp); masks.append(mp)
    if not frames:
        print("no frame/mask pairs found"); return 1
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(frames))
    ntest = max(1, int(len(frames) * args.test_frac))
    test_i, train_i = idx[:ntest], idx[ntest:]
    tr = DS([frames[i] for i in train_i], [masks[i] for i in train_i], t)
    te = DS([frames[i] for i in test_i], [masks[i] for i in test_i], t)
    trl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=2)
    tel = DataLoader(te, batch_size=args.bs, shuffle=False, num_workers=2)
    print(f"train={len(tr)} test={len(te)}", flush=True)

    # build model + load checkpoint (8-class head)
    # Match upstream: reshape ONLY the main classifier head to 8 classes. Leave
    # aux_classifier at its 21-channel default (the checkpoint kept it at 21; it
    # is only used in training and ignored at inference via ['out']).
    model = models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None)
    model.classifier[4] = nn.Conv2d(256, 8, kernel_size=1)
    sd = torch.load(args.weights, map_location="cpu")
    sd = sd.get("state_dict", sd.get("model", sd))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    # disable the aux head so its 21-class output never enters our 8-class loss
    model.aux_classifier = None
    model.to(dev)

    before = iou_report(model, tel, dev)
    print("IoU BEFORE:", json.dumps(before), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # ignore background (0) in the loss so we train only on labeled surfaces
    lossf = nn.CrossEntropyLoss(ignore_index=0)
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for x, y in trl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)["out"]
            loss = lossf(out, y)
            loss.backward(); opt.step()
            tot += float(loss)
        print(f"epoch {ep+1}/{args.epochs} loss={tot/len(trl):.4f}", flush=True)

    after = iou_report(model, tel, dev)
    print("IoU AFTER:", json.dumps(after), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    (args.out.parent / "finetune_report.json").write_text(json.dumps(
        {"before": before, "after": after, "n_train": len(tr), "n_test": len(te),
         "epochs": args.epochs, "lr": args.lr}, indent=2))
    print(f"OK: saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

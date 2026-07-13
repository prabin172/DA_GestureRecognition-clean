#!/usr/bin/env python3
from __future__ import annotations
import argparse, random, re, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.kinematic_encoder import KinematicEncoder
from src.models.kinematic_decoder import KinematicDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A_RE = re.compile(r"A(\d{3})")

class NTUFileDataset(Dataset):
    def __init__(self, files, root, max_frames=120):
        self.files = [Path(f) for f in files]
        self.root = Path(root)
        self.max_frames = max_frames
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        f = self.files[i]
        arr = np.load(self.root / f.name).astype(np.float32)
        arr = arr.reshape(arr.shape[0], -1)
        out = np.zeros((self.max_frames, 68), np.float32)
        mask = np.zeros((self.max_frames,), np.float32)
        T = min(arr.shape[0], self.max_frames)
        out[:T] = arr[:T]; mask[:T] = 1.0
        m = A_RE.search(f.stem)
        y = int(m.group(1)) - 1
        return torch.from_numpy(out), torch.from_numpy(mask), torch.tensor(y, dtype=torch.long)

class GestureHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, z): return self.fc(self.norm(z))

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def masked_mean(x, mask):
    w = mask.float().unsqueeze(-1)
    denom = mask.float().sum(dim=1, keepdim=True).clamp(1.0)
    return (x*w).sum(1)/denom

def geodesic_mae_loss(recon, target, mae_mask):
    rq = F.normalize(recon.view(-1,17,4), p=2, dim=-1)
    tq = target.view(-1,17,4)
    dot = torch.abs(torch.sum(rq*tq, dim=-1)).clamp(0,1)
    dist = 1-dot
    keep = mae_mask.view(-1)==1
    return dist[keep].mean()

def stratified_files(root, seed, train_frac=0.8, adapt_frac_of_holdout=0.25, limit_per_class=0):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for f in sorted(Path(root).glob("*.npy")):
        m = A_RE.search(f.stem)
        if m:
            groups[int(m.group(1))-1].append(f)
    train=[]; adapt=[]; test=[]
    for y, fs in groups.items():
        fs = list(fs)
        rng.shuffle(fs)
        if limit_per_class and len(fs) > limit_per_class:
            fs = fs[:limit_per_class]
        n_train = max(1, int(len(fs)*train_frac))
        train_g = fs[:n_train]
        hold = fs[n_train:]
        n_adapt = max(1, int(len(hold)*adapt_frac_of_holdout)) if len(hold) else 0
        train += train_g
        adapt += hold[:n_adapt]
        test += hold[n_adapt:]
    return train, adapt, test

def train_supervised(ds, epochs, batch, workers):
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    enc=KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    head=GestureHead(120).to(DEVICE)
    opt=optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=2e-4, weight_decay=0.05)
    crit=nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        enc.train(); head.train()
        correct=total=0
        for x,mask,y in tqdm(dl, desc=f"NTU->NTU SUP pretrain {ep}/{epochs}"):
            x=x.to(DEVICE); mask=mask.to(DEVICE); y=y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits=head(masked_mean(enc(x,mask=mask),mask))
            loss=crit(logits,y)
            loss.backward(); opt.step()
            correct += int((logits.argmax(1)==y).sum().item()); total += y.numel()
        print(f"SUP epoch {ep} acc={100*correct/max(total,1):.2f}")
    return enc

def supcon_loss(z, y, temperature=0.1):
    """Supervised contrastive loss (Khosla et al. 2020), in-batch positives = same label."""
    z = F.normalize(z, dim=-1)
    B = z.shape[0]
    sim = z @ z.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    self_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    exp_sim = torch.exp(sim) * self_mask
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & self_mask
    pos_count = pos_mask.sum(1).clamp(min=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_count
    has_pos = pos_mask.sum(1) > 0
    loss = -mean_log_prob_pos
    return loss[has_pos].mean() if has_pos.any() else loss.mean()

def train_supcon(ds, epochs, batch, workers):
    """Mirrors train_supervised but with SupCon loss instead of CE (no classification head)."""
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    enc=KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    opt=optim.AdamW(enc.parameters(), lr=2e-4, weight_decay=0.05)
    for ep in range(1, epochs+1):
        enc.train()
        running=0.0; nb=0
        for x,mask,y in tqdm(dl, desc=f"NTU->NTU SupCon pretrain {ep}/{epochs}"):
            x=x.to(DEVICE); mask=mask.to(DEVICE); y=y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            z=masked_mean(enc(x,mask=mask),mask)
            loss=supcon_loss(z,y)
            loss.backward(); opt.step()
            running += loss.item(); nb += 1
        print(f"SupCon epoch {ep} loss={running/max(nb,1):.4f}")
    return enc

def train_mae(ds, epochs, batch, workers, sup=False):
    dl=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=workers,pin_memory=True)
    enc=KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
    dec=KinematicDecoder(embed_dim=512, feature_dim=68).to(DEVICE)
    head=GestureHead(120).to(DEVICE) if sup else None
    params=list(enc.parameters())+list(dec.parameters())+([] if head is None else list(head.parameters()))
    opt=optim.AdamW(params, lr=2e-4, weight_decay=0.05)
    ce=nn.CrossEntropyLoss()
    mask_ratio=.70
    for ep in range(1, epochs+1):
        enc.train(); dec.train()
        if head: head.train()
        for x,mask,y in tqdm(dl, desc=f"NTU->NTU {'SupMAE' if sup else 'MAE'} pretrain {ep}/{epochs}"):
            x=x.to(DEVICE); mask=mask.to(DEVICE); y=y.to(DEVICE)
            B,T,D=x.shape
            noise=torch.rand(B,T,device=DEVICE)+(1-mask)*10
            ids=torch.argsort(noise,dim=1)[:,:int(T*(1-mask_ratio))]
            masked=torch.zeros_like(x)
            bi=torch.arange(B,device=DEVICE).unsqueeze(-1)
            masked[bi,ids]=x[bi,ids]
            mae_mask=torch.ones(B,T,device=DEVICE); mae_mask[bi,ids]=0; mae_mask=mae_mask*mask
            opt.zero_grad(set_to_none=True)
            zt=enc(masked,mask=mask)
            recon=dec(zt,mask=mask)
            loss=geodesic_mae_loss(recon,x,mae_mask)
            if sup:
                logits=head(masked_mean(enc(x,mask=mask),mask))
                loss = loss + ce(logits,y)
            loss.backward(); opt.step()
    return enc

def train_head_eval(enc, adapt_ds, test_ds, epochs, batch, workers, method):
    for p in enc.parameters(): p.requires_grad_(False)
    enc.eval()
    head=GestureHead(120).to(DEVICE)
    opt=optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    crit=nn.CrossEntropyLoss()
    adl=DataLoader(adapt_ds,batch_size=batch,shuffle=True,num_workers=workers)
    tdl=DataLoader(test_ds,batch_size=batch,shuffle=False,num_workers=workers)
    curve=[]
    for ep in range(1,epochs+1):
        head.train()
        for x,mask,y in adl:
            x=x.to(DEVICE); mask=mask.to(DEVICE); y=y.to(DEVICE)
            with torch.no_grad(): z=masked_mean(enc(x,mask=mask),mask)
            opt.zero_grad(set_to_none=True)
            loss=crit(head(z),y); loss.backward(); opt.step()
        acc=eval_acc(enc,head,tdl)
        curve.append({"method":method,"epoch":ep,"acc":acc})
        print(f"HEAD | {method} epoch={ep} acc={acc:.2f}")
    return curve[-1]["acc"], curve

@torch.no_grad()
def eval_acc(enc,head,dl):
    enc.eval(); head.eval(); c=t=0
    for x,mask,y in dl:
        x=x.to(DEVICE); mask=mask.to(DEVICE); y=y.to(DEVICE)
        pred=head(masked_mean(enc(x,mask=mask),mask)).argmax(1)
        c += int((pred==y).sum().item()); t += y.numel()
    return 100*c/max(t,1)

def make_plots(summary, curves, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd.DataFrame(summary).to_csv(out_dir/"summary.csv", index=False)
    cdf=pd.DataFrame(curves); cdf.to_csv(out_dir/"epoch_curves.csv", index=False)
    plt.figure(figsize=(7,4.5))
    for m,g in cdf.groupby("method"):
        plt.plot(g["epoch"], g["acc"], label=m)
    plt.xlabel("Linear-head epoch"); plt.ylabel("NTU held-out test accuracy (%)")
    plt.title("NTU->NTU objective sanity")
    plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"accuracy_vs_epoch.png", dpi=180); plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ntu-root", type=Path, default=Path("Data_Processed/ntu_quats"))
    ap.add_argument("--out-dir", type=Path, default=Path("trained_models/NTU-to-NTU-objective-sanity"))
    ap.add_argument("--pretrain-epochs", type=int, default=10)
    ap.add_argument("--head-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--methods", default="scratch,supervised,mae,supmae")
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()
    set_seed(args.seed)
    root=args.ntu_root if args.ntu_root.is_absolute() else PROJECT_ROOT/args.ntu_root
    out=args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT/args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    train,adapt,test=stratified_files(root,args.seed,limit_per_class=args.limit_per_class)
    print(f"NTU split train={len(train)} adapt={len(adapt)} test={len(test)} root={root}")
    train_ds=NTUFileDataset(train,root); adapt_ds=NTUFileDataset(adapt,root); test_ds=NTUFileDataset(test,root)
    summary=[]; curves=[]
    for method in [m.strip().lower() for m in args.methods.split(",") if m.strip()]:
        if method=="scratch":
            enc=KinematicEncoder(feature_dim=68, embed_dim=512).to(DEVICE)
        elif method=="supervised":
            enc=train_supervised(train_ds,args.pretrain_epochs,args.batch_size,args.num_workers)
        elif method=="mae":
            enc=train_mae(train_ds,args.pretrain_epochs,args.batch_size,args.num_workers,sup=False)
        elif method=="supmae":
            enc=train_mae(train_ds,args.pretrain_epochs,args.batch_size,args.num_workers,sup=True)
        elif method=="supcon":
            enc=train_supcon(train_ds,args.pretrain_epochs,args.batch_size,args.num_workers)
        else:
            raise ValueError(method)
        acc, curve=train_head_eval(enc,adapt_ds,test_ds,args.head_epochs,args.batch_size,args.num_workers,method)
        summary.append({"method":method,"final_acc":acc,"train_n":len(train),"adapt_n":len(adapt),"test_n":len(test)})
        curves += curve
    make_plots(summary,curves,out)
    print(f"Done. Summary: {out/'summary.csv'}")

if __name__=="__main__":
    main()

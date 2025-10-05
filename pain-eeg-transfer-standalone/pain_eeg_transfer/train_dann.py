
import os, json, numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .models.dann import DANN
import random

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class ArrDS(Dataset):
    def __init__(self, X, y, d):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.d = torch.from_numpy(d).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i], self.d[i]

def train_epoch(model, loader, opt, ce, ce_d, lam, device):
    model.train(); tot=0
    for X,y,d in loader:
        X,y,d = X.to(device), y.to(device), d.to(device)
        opt.zero_grad()
        yhat, dhat = model(X, lam)
        loss = ce(yhat,y) + ce_d(dhat,d)
        loss.backward(); opt.step()
        tot += loss.item()*len(y)
    return tot/len(loader.dataset)

def evaluate(model, loader, device):
    model.eval(); Y=[]; P=[]
    with torch.no_grad():
        for X,y,_ in loader:
            X = X.to(device)
            logits,_ = model(X, lam=0.0)
            prob = torch.softmax(logits,1)[:,1].cpu().numpy()
            Y.append(y.numpy()); P.append(prob)
    y = np.concatenate(Y); p = np.concatenate(P)
    ba = balanced_accuracy_score(y,(p>=0.5).astype(int))
    try: auc = roc_auc_score(y,p)
    except: auc = float('nan')
    return {"balanced_acc": float(ba), "auroc": float(auc)}

def run_training(args):
    set_seed(args.seed)
    data = np.load(args.features, allow_pickle=True)
    X, y, sub, dom_txt = data["X"], data["y"], data["subject"], data["domain"]
    dom = LabelEncoder().fit_transform(dom_txt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    results = {}

    if args.scheme == "loso":
        metrics=[]
        for s in np.unique(sub):
            tr = sub!=s; te = sub==s
            tr_loader = DataLoader(ArrDS(X[tr],y[tr],dom[tr]), batch_size=args.batch_size, shuffle=True)
            te_loader = DataLoader(ArrDS(X[te],y[te],dom[te]), batch_size=256)
            model = DANN(X.shape[1], hidden=args.hidden, n_classes=len(np.unique(y)), n_domains=len(np.unique(dom)), lam=args.lam).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            ce = nn.CrossEntropyLoss(); ce_d = nn.CrossEntropyLoss()
            for ep in range(args.epochs):
                lam = args.lam * (ep/args.epochs)
                train_epoch(model, tr_loader, opt, ce, ce_d, lam, device)
            m = evaluate(model, te_loader, device); m["subject"]=str(s); metrics.append(m)
        results["folds"]=metrics
        results["mean_balanced_acc"]=float(np.nanmean([m["balanced_acc"] for m in metrics]))
        results["mean_auroc"]=float(np.nanmean([m["auroc"] for m in metrics]))

    elif args.scheme == "lodo":
        if args.test_domain is None: raise ValueError("--test_domain required")
        te = dom_txt==args.test_domain; tr = ~te
        tr_loader = DataLoader(ArrDS(X[tr],y[tr],dom[tr]), batch_size=args.batch_size, shuffle=True)
        te_loader = DataLoader(ArrDS(X[te],y[te],dom[te]), batch_size=256)
        model = DANN(X.shape[1], hidden=args.hidden, n_classes=len(np.unique(y)), n_domains=len(np.unique(dom)), lam=args.lam).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        ce = nn.CrossEntropyLoss(); ce_d = nn.CrossEntropyLoss()
        for ep in range(args.epochs):
            lam = args.lam * (ep/args.epochs)
            train_epoch(model, tr_loader, opt, ce, ce_d, lam, device)
        results["lodo"]=evaluate(model, te_loader, device)

    (out/"metrics.json").write_text(json.dumps(results, indent=2))
    print(f"[OK] wrote {out/'metrics.json'}")


import torch, torch.nn as nn

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam=lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.neg()*ctx.lam, None

class DANN(nn.Module):
    def __init__(self, in_dim, hidden=128, n_classes=2, n_domains=2, lam=0.1):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
        )
        self.cls = nn.Linear(hidden, n_classes)
        self.dom = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, n_domains))
        self.lam = lam
    def forward(self, x, lam=None):
        f = self.feat(x)
        y = self.cls(f)
        lam = self.lam if lam is None else lam
        d = self.dom(GradReverse.apply(f, lam))
        return y, d

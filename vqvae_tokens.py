"""Entrena un VQ-VAE pequeño en CIFAR-10 → tokens visuales APRENDIDOS (codebook K=512, grid 8×8=64 por imagen).
Sustituye los tokens kmeans crudos (fuentes correlacionadas/bajo-rango que colisionan bajo CDMA) por un vocab
visual entrenado, para testear si la baja multiplexabilidad de la imagen era artefacto de tokenización.
Guarda vqvae_codes.pt (N,64) int. Reporta perplexity del codebook (uso real, sanity)."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

dev = torch.device("cuda")
torch.manual_seed(0)
K, DZ, EPOCHS, BS, LR, BETA = 512, 64, 12, 256, 3e-4, 0.25
DECAY, EPS = 0.99, 1e-5     # EMA codebook (van den Oord) — evita colapso


class VQ(nn.Module):
    """VQ-VAE con codebook EMA + reinit de códigos muertos (anti-colapso)."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),       # 32→16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),     # 16→8
            nn.Conv2d(128, DZ, 1))                      # 8×8×DZ
        self.register_buffer("cb", torch.randn(K, DZ))           # codebook (no-param, EMA)
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_avg", self.cb.clone())
        self.dec = nn.Sequential(
            nn.Conv2d(DZ, 128, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())

    def quantize(self, ze):                              # ze:(b,DZ,8,8)
        b, c, h, w = ze.shape
        flat = ze.permute(0, 2, 3, 1).reshape(-1, c)    # (b*64,DZ)
        d = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.cb.t() + self.cb.pow(2).sum(1))
        idx = d.argmin(1)                               # (b*64,)
        zq = self.cb[idx].view(b, h, w, c).permute(0, 3, 1, 2)
        if self.training:                               # update EMA del codebook (FUERA de autograd)
            with torch.no_grad():
                fd = flat.detach()
                oh = F.one_hot(idx, K).type_as(fd)      # (b*64,K)
                self.cluster_size.mul_(DECAY).add_(oh.sum(0), alpha=1 - DECAY)
                self.embed_avg.mul_(DECAY).add_(oh.t() @ fd, alpha=1 - DECAY)
                n = self.cluster_size.sum()
                cs = (self.cluster_size + EPS) / (n + K * EPS) * n
                self.cb.copy_(self.embed_avg / cs.unsqueeze(1))
                # reinit códigos muertos a vectores del batch (anti-colapso)
                dead = self.cluster_size < 1.0
                if dead.any():
                    r = fd[torch.randint(0, fd.shape[0], (int(dead.sum()),), device=fd.device)]
                    self.cb[dead] = r; self.embed_avg[dead] = r; self.cluster_size[dead] = 1.0
        return zq, idx.view(b, h * w)

    def forward(self, x):
        ze = self.enc(x)
        zq, idx = self.quantize(ze)
        loss_vq = BETA * F.mse_loss(ze, zq.detach())    # solo commitment (codebook por EMA)
        zq_st = ze + (zq - ze).detach()                 # straight-through
        recon = self.dec(zq_st)
        return recon, loss_vq, idx


def main():
    ds = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.ToTensor())
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])     # (N,3,32,32)
    N = imgs.shape[0]
    m = VQ().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    for ep in range(EPOCHS):
        perm = torch.randperm(N)
        tot_r = tot_v = 0.0
        for i in range(0, N, BS):
            xb = imgs[perm[i:i+BS]].to(dev)
            recon, lvq, _ = m(xb)
            lr_ = F.mse_loss(recon, xb)
            loss = lr_ + lvq
            opt.zero_grad(); loss.backward(); opt.step()
            tot_r += lr_.item() * xb.shape[0]; tot_v += lvq.item() * xb.shape[0]
        print(f"  ep {ep:2d} recon={tot_r/N:.4f} vq={tot_v/N:.4f}", flush=True)
    # encode todo
    m.eval()
    codes = torch.empty(N, 64, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, N, BS):
            _, _, idx = m(imgs[i:i+BS].to(dev))
            codes[i:i+idx.shape[0]] = idx.cpu()
    torch.save(codes, "vqvae_codes.pt")
    # perplexity del codebook (uso): exp(H) sobre la distribución de códigos
    counts = torch.bincount(codes.flatten(), minlength=K).float()
    p = counts / counts.sum()
    ppl = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
    used = int((counts > 0).sum())
    print(f"\ncodes (N,64)={tuple(codes.shape)}  codebook usado={used}/{K}  perplexity={ppl:.1f}/{K}")
    print("→ vqvae_codes.pt")


if __name__ == "__main__":
    main()

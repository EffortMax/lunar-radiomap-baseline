"""
阶段 1：pathloss 图的自编码器（AE）重建实验。

目标：
  把 256x256 的 pathloss 图压缩成潜特征 z（默认 4 x 32 x 32，压缩约 16 倍），
  再解码回 pathloss，检验潜特征能否保留足够的结构（用 valid px RMSE，单位 dB 评估）。

要点：
  - 只在有效像素（mask==1，光线追踪得到的值）上计算损失与指标。
  - 输入：归一化 pl（无效像素置 0）+ 掩码通道，让网络知道哪些像素无效。
  - 归一化范围默认使用 metadata.json 中全局 pathloss_range_db（[20.0, 228.0]，跨度 208 dB），
    与官方基线 lunar_dataset.py / train.py / train_wandb.py 保持完全一致
    （可通过 --norm-range per_band 切换为 pathloss_range_db_per_band[band]）。
  - 不使用高程图、Tx 位置，不做几何条件（属于阶段 2/3）。
  - 只写新文件，不修改 train.py；不删除任何文件。
  - W&B（第 2 版补丁）：默认离线记录（--wandb-mode offline），本地 run 目录在 <out>/wandb 下，
    训练结束后用 wandb sync 上传；--wandb-mode online 为在线记录；--wandb-mode disabled 完全关闭。
    key 不写入本文件，通过环境变量 WANDB_API_KEY（见 wandb_accounts.template.ps1）或已有登录提供。

输出（--out 目录）：
  ae_<run>_best.pt      验证集 RMSE 最好的权重（只含 encoder/decoder）
  ae_<run>_history.csv  每个 epoch 的训练损失与验证 RMSE
  ae_<run>_summary.json 最终结果与配置

用法（示例，先小规模冒烟）：
  python ae_recon.py --root <数据目录> --band 415 --epochs 1 --batch-size 16 --limit-batches 20 --out runs_ae --run-name smoke

工作量估算（运行前请用 --limit-batches 测一次实际耗时）：
  训练集 25,720 个样本；batch 16 时约 1,608 步/epoch；验证集 2,330 个样本。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class PLReconDataset(Dataset):
    """返回 (输入 [2,H,W], 目标 [1,H,W], 掩码 [1,H,W])。"""

    def __init__(self, root, split, band, lo, hi):
        self.root = root
        self.split = split
        self.lo, self.hi = float(lo), float(hi)
        self.pl = np.load(os.path.join(root, f"{split}_pl{band}.npy"), mmap_mode="r")
        self.mk = np.load(os.path.join(root, f"{split}_mask{band}.npy"), mmap_mode="r")
        self.n = self.pl.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        pl = np.asarray(self.pl[i]).astype(np.float32)
        valid = np.unpackbits(np.asarray(self.mk[i])).reshape(256, 256).astype(np.float32)
        y = np.clip((pl - self.lo) / (self.hi - self.lo), 0.0, 1.0) * valid
        x = np.stack([y, valid], axis=0)
        return torch.from_numpy(x), torch.from_numpy(y[None]), torch.from_numpy(valid[None])


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
    )


class AE(nn.Module):
    """256 -> 128 -> 64 -> 32 的卷积 AE，潜特征 latent_ch x 32 x 32。"""

    def __init__(self, base=32, latent_ch=4):
        super().__init__()
        self.enc = nn.Sequential(
            block(2, base), nn.Conv2d(base, base, 4, 2, 1),           # 128
            block(base, base * 2), nn.Conv2d(base * 2, base * 2, 4, 2, 1),  # 64
            block(base * 2, base * 4), nn.Conv2d(base * 4, base * 4, 4, 2, 1),  # 32
            block(base * 4, base * 4), nn.Conv2d(base * 4, latent_ch, 1),   # latent
        )
        self.dec = nn.Sequential(
            nn.Conv2d(latent_ch, base * 4, 1), block(base * 4, base * 4),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(base * 4, base * 2, 3, padding=1),  # 64
            block(base * 2, base * 2),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(base * 2, base, 3, padding=1),      # 128
            block(base, base),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(base, base, 3, padding=1),          # 256
            block(base, base), nn.Conv2d(base, 1, 1),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def masked_mse(pred, y, m):
    num = ((pred - y) ** 2 * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--band", default="415", choices=["415", "58"])
    ap.add_argument("--norm-range", default="global", choices=["global", "per_band"],
                    help="默认 global 使用 metadata.json 的 pathloss_range_db ([20.0, 228.0])，与基线 lunar_dataset.py 一致；"
                         "per_band 使用 pathloss_range_db_per_band[band]")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--latent-ch", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-batches", type=int, default=0, help="冒烟测试：每个 epoch 只跑前 n 个 batch")
    ap.add_argument("--out", default="runs_ae")
    ap.add_argument("--run-name", default="ae415")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"],
                    help="默认离线记录；训练后用 wandb sync 上传")
    ap.add_argument("--wandb-project", default="lunar-radiomap-gc1")
    ap.add_argument("--wandb-entity", default="", help="团队名；留空则使用环境变量 WANDB_ENTITY 或个人账户")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.root, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if a.norm_range == "global":
        lo, hi = meta["pathloss_range_db"]
    else:
        lo, hi = meta["pathloss_range_db_per_band"][a.band]

    tr = PLReconDataset(a.root, "train", a.band, lo, hi)
    va = PLReconDataset(a.root, "val", a.band, lo, hi)
    dl_tr = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, drop_last=True)
    dl_va = DataLoader(va, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)

    dev = torch.device(a.device)
    model = AE(a.base, a.latent_ch).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs))
    scaler = torch.amp.GradScaler(enabled=(dev.type == "cuda"))

    print(f"band={a.band} norm_range={a.norm_range} dB range=[{lo},{hi}] train={len(tr)} val={len(va)} "
          f"params={n_params:,} device={dev}")

    wb_run = None
    if a.wandb_mode != "disabled":
        import wandb  # 需要：python -m pip install wandb（已在本机 .venv 中安装）
        wb_dir = os.path.join(a.out, "wandb")
        os.makedirs(wb_dir, exist_ok=True)
        wb_run = wandb.init(
            project=a.wandb_project,
            entity=(a.wandb_entity or None),
            name=a.run_name,
            config={**vars(a), "params": n_params, "device": str(dev), "dB_lo": lo, "dB_hi": hi},
            mode=a.wandb_mode,
            dir=wb_dir,
            tags=["stage1", "ae", f"band{a.band}"],
        )
        print(f"wandb: mode={a.wandb_mode} project={a.wandb_project} run={a.run_name} dir={wb_dir}")

    hist_path = os.path.join(a.out, f"ae_{a.run_name}_history.csv")
    best_path = os.path.join(a.out, f"ae_{a.run_name}_best.pt")
    sum_path = os.path.join(a.out, f"ae_{a.run_name}_summary.json")
    best = float("inf")
    rows = []

    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        tot, cnt = 0.0, 0
        for i, (x, y, m) in enumerate(dl_tr):
            if a.limit_batches and i >= a.limit_batches:
                break
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            with torch.autocast(device_type=dev.type, enabled=(dev.type == "cuda")):
                pred, _ = model(x)
                loss = masked_mse(pred.float(), y, m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            cnt += 1
        sched.step()
        train_loss = tot / max(cnt, 1)

        model.eval()
        se, n_valid = 0.0, 0.0
        with torch.no_grad():
            for i, (x, y, m) in enumerate(dl_va):
                if a.limit_batches and i >= a.limit_batches:
                    break
                x, y, m = x.to(dev), y.to(dev), m.to(dev)
                pred, _ = model(x)
                d = (pred.float().clamp(0, 1) - y) * (hi - lo)   # 反归一化到 dB 差值
                se += float(((d ** 2) * m).sum())
                n_valid += float(m.sum())
        val_rmse_db = float(np.sqrt(se / max(n_valid, 1.0)))
        dt = time.time() - t0
        row = {"epoch": ep, "train_loss_norm": train_loss, "val_rmse_db": val_rmse_db,
               "lr": opt.param_groups[0]["lr"], "seconds": round(dt, 1)}
        rows.append(row)
        print(f"epoch {ep}: train_loss {train_loss:.6f}  val_rmse {val_rmse_db:.3f} dB  ({dt:.0f}s)")
        if wb_run is not None:
            wb_run.log({"epoch": ep, "train/loss": train_loss, "val/rmse_db": val_rmse_db,
                        "lr": row["lr"], "seconds": dt}, step=ep)

        if val_rmse_db < best:
            best = val_rmse_db
            torch.save({"model": model.state_dict(), "args": vars(a), "lo": lo, "hi": hi,
                        "norm_range": a.norm_range, "epoch": ep, "val_rmse_db": val_rmse_db}, best_path)
            print(f"  new best {val_rmse_db:.3f} dB -> {best_path}")

    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_rmse_db": best, "params": n_params, "config": vars(a),
                   "dB_lo": lo, "dB_hi": hi, "norm_range": a.norm_range,
                   "latent_shape": [a.latent_ch, 32, 32], "compression_ratio": 256 * 256 / (a.latent_ch * 32 * 32)},
                  f, ensure_ascii=False, indent=2)
    if wb_run is not None:
        wb_run.summary["best_val_rmse_db"] = best
        wb_run.summary["params"] = n_params
        wb_run.finish()
    print(f"done. best val RMSE {best:.3f} dB (valid px). history -> {hist_path}")


if __name__ == "__main__":
    main()

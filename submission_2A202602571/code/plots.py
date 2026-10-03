"""plots.py — ảnh từng thí nghiệm (figures/<exp_id>.png) và ảnh chồng theo nhóm (figures/compare_<nhóm>.png)."""
from __future__ import annotations

import matplotlib.pyplot as plt


def _cfg_text(cfg: dict) -> str:
    hidden = "-".join(str(h) for h in cfg["hidden"])
    parts = [f"loss={cfg['loss']}", f"opt={cfg['optimizer']}", f"lr={cfg['lr']:g}", f"wd={cfg['weight_decay']:g}",
             f"batch={cfg['batch']}", f"hidden={hidden}", f"dropout={cfg['dropout']:g}",
             f"clip={cfg['clip_norm'] if cfg['clip_norm'] is not None else 'none'}",
             f"{cfg['precision']}", f"init={cfg['init']}", f"seed={cfg['seed']}"]
    if cfg.get("scheduler"):
        parts.append(f"sched={cfg['scheduler']}")
    return ", ".join(parts)


def plot_run(result: dict, path: str) -> None:
    """Một thí nghiệm -> một PNG 3 ô: (1) train/val loss, (2) val acc + macro-F1, (3) grad_norm trước clip."""
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    ep = h["epoch"]
    if ep:
        ax = axes[0]
        ax.plot(ep, h["train_loss"], marker="o", ms=3, label="train loss (eval mode)")
        ax.plot(ep, h["val_loss"], marker="s", ms=3, label="val loss")
        ax.set_ylabel(f"loss ({cfg['loss'].upper()})"); ax.set_title("Loss theo epoch")
        ax = axes[1]
        ax.plot(ep, h["val_acc"], marker="o", ms=3, label="val accuracy")
        ax.plot(ep, h["val_macro_f1"], marker="s", ms=3, label="val macro-F1")
        ax.axhline(0.4876, color="gray", ls=":", lw=1, label="đoán lớp đa số (acc 0,4876)")
        ax.set_ylabel("metric"); ax.set_title("Val accuracy / macro-F1")
        ax = axes[2]
        ax.plot(ep, h["grad_norm"], marker="o", ms=3, label="trung bình mỗi epoch")
        if "grad_norm_max" in h:
            ax.plot(ep, h["grad_norm_max"], ls="--", lw=1, label="lớn nhất trong epoch")
        if cfg["clip_norm"] is not None:
            ax.axhline(cfg["clip_norm"], color="red", ls=":", lw=1.2, label=f"ngưỡng clip c={cfg['clip_norm']:g}")
        ax.set_yscale("log"); ax.set_ylabel("‖g‖₂ toàn cục (trước clip)"); ax.set_title("grad_norm")
        for ax in axes:
            if s.get("best_epoch") is not None:
                ax.axvline(s["best_epoch"], color="k", ls="--", lw=0.8, alpha=0.5, label=f"best epoch={s['best_epoch']}")
            ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    else:
        reason = result.get("notes") or s.get("diverged_at") or "không có epoch nào hoàn tất"
        for ax in axes:
            ax.text(0.5, 0.5, "Không có lịch sử để vẽ\n" + str(reason), ha="center", va="center",
                    wrap=True, transform=ax.transAxes, fontsize=9)
            ax.set_xlabel("epoch")
    status = "  [DIVERGED]" if s.get("diverged") else ""
    fig.suptitle(f"{cfg['exp_id']}{status} — {_cfg_text(cfg)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "") -> None:
    """Vẽ chồng một hoặc nhiều chỉ số của nhiều thí nghiệm, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    metric: tên một khoá của history ("val_loss", "val_macro_f1", "grad_norm", ...) hoặc danh sách các khoá
            (mỗi khoá một ô). Lần chạy không có lịch sử (phân kỳ từ epoch 1) được ghi vào chú thích.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for r in results:
            h, eid = r["history"], r["cfg"]["exp_id"]
            if h["epoch"]:
                ax.plot(h["epoch"], h[m], marker="o", ms=2.5, lw=1.3,
                        label=eid + (" (diverged)" if r["summary"].get("diverged") else ""))
            else:
                ax.plot([], [], label=eid + " (không có epoch nào)")
        if m.startswith("grad_norm"):
            ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel(m); ax.set_title(m); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(title or "So sánh", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

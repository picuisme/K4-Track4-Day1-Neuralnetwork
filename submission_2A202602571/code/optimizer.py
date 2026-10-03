"""optimizer.py — chọn bộ tối ưu, lịch tốc độ học và cắt gradient (gom một chỗ để mọi thí nghiệm công bằng).

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    weight_decay của SGD/Adam là L2 cộng vào gradient (với Adam nó bị chia cho sqrt(v_hat));
    weight_decay của AdamW trừ thẳng vào trọng số, không đi qua m, v.
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận {name!r}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, warmup_steps: int = 0, **kwargs):
    """Bộ lập lịch tốc độ học, gọi .step() sau MỖI bước cập nhật.

    None      -> None (lr hằng, dùng cho baseline và mọi thí nghiệm một-yếu-tố)
    "cosine"  -> khởi động tuyến tính `warmup_steps` bước rồi giảm cosine về 0 ở bước cuối
    """
    if name is None or name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"scheduler không hỗ trợ: {name!r}")

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    clip_grad_norm_ trả về chuẩn toàn cục đo trước khi nhân hệ số min(1, c/||g||);
    với max_norm=inf hệ số luôn là 1 nên chỉ đo mà không cắt.
    Khi dùng FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    c = float("inf") if max_norm is None else float(max_norm)
    total_norm = torch.nn.utils.clip_grad_norm_(params, c)
    return float(total_norm)

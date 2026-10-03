"""model.py — MLP cho bài toán 7 lớp, shape cố định (README mục 3; GUIDE "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}

INITS = ("zeros", "normal", "xavier", "he", "default")


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron (p của nn.Dropout = q trong slide); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        layers, n_in = [], in_features
        for h in hidden:
            # Dropout(p=0) là phép đồng nhất nên luôn giữ trong chuỗi: cấu trúc module như nhau ở mọi thí nghiệm
            layers += [nn.Linear(n_in, h, bias=True), nn.ReLU(), nn.Dropout(dropout)]
            n_in = h
        layers.append(nn.Linear(n_in, num_classes, bias=True))   # logit thô, không softmax, không dropout
        self.net = nn.Sequential(*layers)
        self.hidden, self.dropout, self.init = tuple(hidden), dropout, init
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7)."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias = 0, trừ "default").

        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_  -> Var = 2/(n_in+n_out)  (KHÔNG phải 1/n_in của slide)
        "he"      : nn.init.kaiming_normal_(nonlinearity="relu") -> Var = 2/n_in
        "default" : giữ mặc định của nn.Linear (kaiming_uniform a=sqrt(5) ~ U(-1/sqrt(n_in), 1/sqrt(n_in)),
                    Var = 1/(3 n_in); bias cũng ngẫu nhiên) — KHÔNG phải He
    """
    if init not in INITS:
        raise ValueError(f"init phải thuộc {INITS}, nhận {init!r}")
    if init == "default":
        return
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init == "zeros":
                nn.init.zeros_(m.weight)
            elif init == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init == "he":
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt theo lớp ở bước 0 (một lô val).

    Quy ước đo: với lớp ẩn, đo SAU ReLU (đầu vào thật sự của lớp kế tiếp); với lớp cuối, đo trên logit
    (sau Linear). Trả về [std_ẩn_1, ..., std_ẩn_k, std_logit].
    """
    model.eval()
    h, stats = x, []
    layers = list(model.net)
    for i, layer in enumerate(layers):
        h = layer(h)
        if isinstance(layer, nn.ReLU) or i == len(layers) - 1:
            stats.append(h.float().std().item())
    return stats

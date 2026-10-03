"""data.py — nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz. Trả về X_train_full, y_train_full, X_eval, y_eval, eval_row_id."""
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_tr, y_tr = tr["X"], tr["y"]
    X_ev, y_ev, row_id = ev["X"], ev["y"], ev["row_id"]
    for X, y in ((X_tr, y_tr), (X_ev, y_ev)):
        assert X.dtype == np.float32 and X.ndim == 2 and X.shape[1] == N_FEATURES, X.shape
        assert y.dtype == np.int64 and y.shape == (len(X),)
        assert y.min() >= 0 and y.max() <= N_CLASSES - 1, "nhãn phải nằm trong 0..6"
    assert len(row_id) == len(X_ev) and len(np.unique(row_id)) == len(row_id)
    return X_tr, y_tr, X_ev, y_ev, row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn. Trả về X_tr, y_tr, X_val, y_val.

    Mọi thí nghiệm dùng cùng seed=42 và val_fraction=0.2 nên cùng một tập val.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean và std của 10 cột số, tính CHỈ trên phần train còn lại sau khi tách val.

    Nếu tính trên val/eval (hay toàn bộ dữ liệu), thống kê của dữ liệu "chưa được thấy" lọt vào
    tiền xử lý: rò rỉ thông tin, điểm val/eval không còn là ước lượng trung thực.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)   # float64 để trung bình của 370k số lớn không mất chính xác
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)          # cột hằng: chia 1 thay vì chia 0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Bản sao của X với 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    out = X.copy()
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / std
    return out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader)."""
    X_full, y_full, X_ev, y_ev, row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)                      # chỉ từ X_tr
    X_tr, X_val, X_ev = (apply_standardizer(a, mean, std) for a in (X_tr, X_val, X_ev))

    majority = int(np.bincount(y_tr, minlength=N_CLASSES).argmax())   # lớp đa số xác định trên train
    majority_val_acc = float((y_val == majority).mean())
    if verbose:
        print(f"train (sau tách val): {X_tr.shape}   val: {X_val.shape}   eval: {X_ev.shape}")
        print("tỉ lệ lớp (%)  train:", np.round(100 * np.bincount(y_tr, minlength=7) / len(y_tr), 2))
        print("               val  :", np.round(100 * np.bincount(y_val, minlength=7) / len(y_val), 2))
        print(f"luôn đoán lớp đa số (lớp {majority}) -> accuracy trên val = {majority_val_acc:.4f}")

    t = lambda a, dt: torch.tensor(a, dtype=dt, device=device)
    return dict(
        X_tr=t(X_tr, torch.float32), y_tr=t(y_tr, torch.int64),
        X_val=t(X_val, torch.float32), y_val=t(y_val, torch.int64),
        X_eval=t(X_ev, torch.float32), y_eval=t(y_ev, torch.int64),
        eval_row_id=row_id, mean=mean, std=std,
        majority_class=majority, majority_val_acc=majority_val_acc, device=str(device),
    )


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Lô cuối có thể nhỏ hơn batch_size và VẪN được dùng (không bỏ mẫu nào); loss lấy trung bình
    trong lô nên lô nhỏ chỉ làm bước cập nhật đó nhiễu hơn một chút.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]

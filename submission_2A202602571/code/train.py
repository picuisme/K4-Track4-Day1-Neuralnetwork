"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val trong notebook rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val (Part 2 của notebook), không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
    scheduler=None,            # None = lr hằng; "cosine" chỉ dùng ở cấu hình kết hợp cuối (ghi vào notes)
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán. Cùng công thức với scripts/evaluate.py.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, k: int = N_CLASSES) -> np.ndarray:
    """Ma trận nhầm lẫn k x k tính ngay trên device bằng bincount (hàng = thật, cột = dự đoán)."""
    return torch.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k).cpu().numpy()


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Nhãn dự đoán int64 (N,) = argmax của logits, ở chế độ eval(), FP32."""
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)])


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy trên logit thô và nhãn int64.
       "mse" : MSE giữa logit và one-hot của y, KHÔNG có hệ số 1/2; trung bình trên cả 7 phần tử của mỗi mẫu
               rồi trung bình trên lô (đúng như nn.MSELoss mặc định: chia cho B*7).
    reduction="sum" trả tổng theo mẫu (dùng trong evaluate để chia N ở cuối).
    """
    logits = logits.float()     # dưới autocast logit có thể là fp16/bf16; tính loss ở fp32 cho ổn định
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, num_classes=logits.shape[1]).float()
        per_sample = ((logits - target) ** 2).mean(dim=1)
        return per_sample.mean() if reduction == "mean" else per_sample.sum()
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """dict(loss, acc, macro_f1, confusion) ở chế độ eval() (dropout tắt), no_grad, FP32."""
    model.eval()
    total, preds = 0.0, []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        total += compute_loss(logits, y[i:i + batch_size], loss_name, reduction="sum").item()
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds)
    cm = confusion_matrix(y, pred)
    return dict(loss=total / len(X), acc=float(np.trace(cm) / cm.sum()),
                macro_f1=macro_f1_from_confusion(cm), confusion=cm)


def _sync(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về {"cfg", "history", "summary", "best_state"}.

    Quy ước đo:
      - train_loss: toàn bộ X_tr, ở eval() (dropout tắt) -> cùng thang với val_loss.
      - grad_norm : chuẩn L2 toàn cục TRƯỚC khi clip; mỗi epoch lưu trung bình và giá trị lớn nhất.
      - epoch_time_s: chỉ thời gian vòng lặp cập nhật (không tính phần đánh giá cuối epoch), đã đồng bộ GPU.
      - peak_mem_MB : torch.cuda.max_memory_allocated; gồm cả tensor dữ liệu đang nằm trên GPU.
      - best_epoch  : epoch có val_loss thấp nhất; val_acc / val_macro_f1 trong summary lấy tại epoch đó.
    X_eval không bao giờ được dùng trong hàm này.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] chưa được đặt")
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    dev = device.type
    hidden, precision, loss_name = tuple(cfg["hidden"]), cfg["precision"], cfg["loss"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"precision không hợp lệ: {precision!r}")
    if precision == "fp16" and dev != "cuda":
        raise RuntimeError("fp16 + GradScaler cần GPU CUDA; thiết bị hiện tại là " + dev)
    if precision == "bf16" and dev == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU này không hỗ trợ bf16")

    # ---- 0. seed, model, optimizer
    set_seed(cfg["seed"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    n_params = count_params(model)
    if hidden in EXPECTED_PARAMS:
        assert n_params == EXPECTED_PARAMS[hidden], (n_params, EXPECTED_PARAMS[hidden])
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), total_steps=steps_per_epoch * cfg["epochs"],
                                warmup_steps=steps_per_epoch if cfg.get("scheduler") == "cosine" else 0)
    use_amp = precision != "fp32"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    gen = torch.Generator(device=device)          # thứ tự xáo lô phụ thuộc seed của cấu hình
    gen.manual_seed(cfg["seed"])
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- 1. loss bước 0 (trước mọi cập nhật), trên val
    step0_loss = evaluate(model, X_val, y_val, loss_name)["loss"]

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                            "grad_norm", "grad_norm_max", "epoch_time_s")}
    best_val, best_epoch, best_state = float("inf"), None, None
    diverged, diverged_at, all_gn, n_clipped, n_skipped = False, None, [], 0, 0
    clip = cfg["clip_norm"]

    # ---- 2. vòng huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        gns = []
        _sync(dev); t0 = time.perf_counter()
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], gen):
            with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):   # chỉ bọc forward + loss
                loss = compute_loss(model(xb), yb, loss_name)
            if not torch.isfinite(loss):
                diverged, diverged_at = True, f"epoch {epoch}, bước {len(gns) + 1}: loss huấn luyện = {loss.item()}"
                break
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)        # đưa gradient về thang thật TRƯỚC khi đo / clip
                gn = clip_gradients(model.parameters(), clip)
                scaler.step(optimizer)            # tự bỏ qua bước nếu gradient có inf/NaN
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if math.isfinite(gn):
                gns.append(gn)
                n_clipped += int(clip is not None and gn > clip)
            else:
                n_skipped += 1
        _sync(dev); epoch_time = time.perf_counter() - t0
        all_gn += gns
        if diverged:
            break

        # ---- cuối epoch: mọi phép đo ở eval()
        tr = evaluate(model, X_tr, y_tr, loss_name)
        va = evaluate(model, X_val, y_val, loss_name)
        if not (math.isfinite(tr["loss"]) and math.isfinite(va["loss"])):
            diverged, diverged_at = True, f"epoch {epoch}: loss đánh giá không hữu hạn"
            break
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr["loss"]); hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"]); hist["val_macro_f1"].append(va["macro_f1"])
        hist["grad_norm"].append(float(np.mean(gns)) if gns else float("nan"))
        hist["grad_norm_max"].append(float(np.max(gns)) if gns else float("nan"))
        hist["epoch_time_s"].append(epoch_time)
        if va["loss"] < best_val:
            best_val, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}   # bản sao, không tham chiếu
        if verbose:
            print(f"  [{cfg['exp_id']}] ep {epoch:2d}  train {tr['loss']:.4f}  val {va['loss']:.4f}  "
                  f"acc {va['acc']:.4f}  f1 {va['macro_f1']:.4f}  gn {hist['grad_norm'][-1]:.3f}  {epoch_time:.1f}s")

    # ---- 3. tóm tắt (tên khoá trùng tên cột của experiments.xlsx)
    i = None if best_epoch is None else hist["epoch"].index(best_epoch)
    g = np.asarray(all_gn) if all_gn else None
    summary = dict(
        step0_loss=step0_loss,
        best_val_loss=None if i is None else hist["val_loss"][i],
        best_epoch=best_epoch,
        final_train_loss=hist["train_loss"][-1] if hist["epoch"] else None,
        final_val_loss=hist["val_loss"][-1] if hist["epoch"] else None,
        val_acc=None if i is None else hist["val_acc"][i],
        val_macro_f1=None if i is None else hist["val_macro_f1"][i],
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])) if hist["epoch"] else None,
        peak_mem_MB=torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else None,
        diverged=diverged,
        # phần bổ sung (không phải cột của bảng) để lý giải kết quả
        diverged_at=diverged_at, n_params=n_params, total_steps=len(all_gn) + n_skipped,
        steps_per_epoch=steps_per_epoch, epochs_done=len(hist["epoch"]),
        grad_norm_p50=None if g is None else float(np.percentile(g, 50)),
        grad_norm_p90=None if g is None else float(np.percentile(g, 90)),
        grad_norm_max=None if g is None else float(g.max()),
        clip_fraction=None if (clip is None or g is None) else n_clipped / len(g),
        skipped_steps=n_skipped, device=dev,
    )
    return dict(cfg=cfg, history=hist, summary=summary, best_state=best_state)


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi CSV `row_id,pred` cho scripts/evaluate.py (đủ mọi dòng eval, mỗi row_id đúng một lần)."""
    row_id, preds = np.asarray(row_id).astype(np.int64), np.asarray(preds).astype(np.int64)
    assert row_id.shape == preds.shape and len(np.unique(row_id)) == len(row_id)
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1
    with open(path, "w", newline="\n") as f:
        f.write("row_id,pred\n")
        f.writelines(f"{r},{p}\n" for r, p in zip(row_id.tolist(), preds.tolist()))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> str:
    """Nạp best_state (epoch có val loss thấp nhất), dự đoán TOÀN BỘ eval ở eval()/FP32, ghi CSV.

    Chỉ gọi cho baseline và cấu hình cuối cùng, sau khi mọi lựa chọn đã chốt bằng val.
    Việc chấm điểm do scripts/evaluate.py làm (gọi trong notebook). Trả về pred_path.
    """
    if result.get("best_state") is None:
        raise RuntimeError("không có best_state (lần chạy phân kỳ?)")
    device = data["X_eval"].device
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    return pred_path

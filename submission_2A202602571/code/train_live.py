"""train_live.py — huấn luyện một cấu hình và IN train/val loss sau từng epoch ra terminal (để theo dõi / chụp màn hình).

Chạy trong thư mục submission_<MSSV>/code/:

    python train_live.py                    # baseline base-s1: M-base, SGD+momentum, lr 0.3, 20 epoch, seed 1
    python train_live.py --config final     # cấu hình cuối final-s1: M-wide, Adam lr 0.003, 40 epoch, cosine
    python train_live.py --epochs 5 --seed 2 --csv train_log.csv

Dùng lại đúng run_experiment của train.py nên số in ra cùng định nghĩa với notebook:
train loss đo ở eval() trên toàn bộ train, val loss / accuracy / macro-F1 trên val, grad_norm trước khi clip.
Script KHÔNG ghi vào figures/ hay results/ và không đụng tới tập eval; chỉ ghi CSV khi có --csv.
"""
from __future__ import annotations

import argparse
import csv
import math

import torch

from data import prepare_data
from train import DEFAULT_CFG, run_experiment

CONFIGS = {
    # lr của baseline và cấu hình cuối là các giá trị đã chọn bằng val trong lab.ipynb
    "baseline": dict(exp_id="base-s1", lr=0.3),
    "final": dict(exp_id="final-s1", group="final", optimizer="adam", lr=0.003, weight_decay=0.0,
                  hidden=(512, 256), epochs=40, scheduler="cosine"),
}


def main():
    ap = argparse.ArgumentParser(description="In train/val loss theo epoch trong lúc huấn luyện")
    ap.add_argument("--config", choices=CONFIGS, default="baseline")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None, help="mặc định: số epoch của cấu hình")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default="../../data/processed", help="thư mục chứa train.npz, eval.npz")
    ap.add_argument("--csv", default=None, help="nếu đặt, ghi lịch sử theo epoch ra file CSV này")
    args = ap.parse_args()

    cfg = {**DEFAULT_CFG, **CONFIGS[args.config], "seed": args.seed}
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    print(f"torch {torch.__version__} | device: {args.device}"
          + (f" | GPU: {torch.cuda.get_device_name(0)}" if args.device == "cuda" else ""))
    data = prepare_data(args.device, val_fraction=0.2, seed=42, processed_dir=args.data)
    print(f"\ncấu hình [{args.config}]: loss={cfg['loss']}, optimizer={cfg['optimizer']}, lr={cfg['lr']}, "
          f"batch={cfg['batch']}, epochs={cfg['epochs']}, hidden={tuple(cfg['hidden'])}, "
          f"scheduler={cfg['scheduler']}, seed={cfg['seed']}")
    print(f"mốc: ln 7 = {math.log(7):.4f}; accuracy đoán lớp đa số trên val = {data['majority_val_acc']:.4f}")
    print("mỗi dòng: epoch | train loss (eval mode) | val loss | val accuracy | val macro-F1 | grad_norm TB | thời gian\n")

    result = run_experiment(cfg, data, verbose=True)      # verbose=True in một dòng sau mỗi epoch

    s, h = result["summary"], result["history"]
    print(f"\nloss bước 0 (val, trước cập nhật đầu tiên): {s['step0_loss']:.4f}")
    if s["best_epoch"] is None:
        print("lần chạy phân kỳ:", s["diverged_at"])
    else:
        print(f"epoch tốt nhất theo val loss: {s['best_epoch']}  | val loss {s['best_val_loss']:.4f}  "
              f"| val acc {s['val_acc']:.4f}  | val macro-F1 {s['val_macro_f1']:.4f}")
        print(f"train loss cuối: {s['final_train_loss']:.4f}  | val loss cuối: {s['final_val_loss']:.4f}  "
              f"| {s['time_per_epoch_s']:.2f} s/epoch")
    if args.csv:
        keys = ["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "epoch_time_s"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys)
            w.writerows(zip(*(h[k] for k in keys)))
        print("đã ghi", args.csv)


if __name__ == "__main__":
    main()

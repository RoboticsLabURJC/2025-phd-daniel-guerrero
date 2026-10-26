#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np


def balance_labels_csv(
    csv_path: str,
    out_path: str = None,
    steer_col: str = "expert_steer",
    seed: int = 42,
    target_n: int = 35000,
    bin_edges=(0.02, 0.05, 0.20, 0.50, 1.00),
    keep_frac=None,
):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    steer_col = steer_col.strip().lower()
    if steer_col not in df.columns:
        raise KeyError(f"No existe la columna '{steer_col}' en el CSV. Columnas: {list(df.columns)}")

    # Recomendado para 47k -> 35k
    if keep_frac is None:
        keep_frac = {
            "straight": 0.50,  # |steer| <= 0.02
            "near":     0.70,  # 0.02 < |steer| <= 0.05
            "soft":     1.00,  # 0.05 < |steer| <= 0.20
            "medium":   1.00,  # 0.20 < |steer| <= 0.50
            "hard":     1.00,  # 0.50 < |steer| <= 1.00
        }

    abs_steer = df[steer_col].astype("float32").abs().to_numpy()
    e0, e1, e2, e3, e4 = bin_edges
    labels = ["straight", "near", "soft", "medium", "hard"]

    bins = pd.cut(
        abs_steer,
        bins=[-1e-9, e0, e1, e2, e3, e4],
        labels=labels,
        include_lowest=True,
    )

    df2 = df.copy()
    df2["_bin"] = bins.astype(str)

    # 1) Downsample por fracción
    parts = []
    for k in labels:
        sub = df2[df2["_bin"] == k]
        if len(sub) == 0:
            continue
        frac = float(keep_frac.get(k, 1.0))
        if frac < 1.0:
            sub = sub.sample(frac=frac, random_state=seed)
        parts.append(sub)

    balanced = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    # 2) Si quedó arriba de target, recorta proporcional por bin hasta target
    if target_n is not None and len(balanced) > target_n:
        grouped = balanced.groupby("_bin", group_keys=False)
        total = len(balanced)
        out_parts = []
        for k, g in grouped:
            take = max(1, int(round(target_n * (len(g) / total))))
            take = min(take, len(g))
            out_parts.append(g.sample(n=take, random_state=seed))
        balanced = pd.concat(out_parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

        if len(balanced) > target_n:
            balanced = balanced.sample(n=target_n, random_state=seed).reset_index(drop=True)

    if out_path is None:
        base, ext = os.path.splitext(csv_path)
        out_path = f"{base}_balanced_{len(balanced)}{ext}"

    balanced.drop(columns=["_bin"], errors="ignore").to_csv(out_path, index=False)

    print(f"Original:   {len(df)}")
    print(f"Balanceado: {len(balanced)}")
    print(f"Guardado en: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Balanceo de labels.csv por bins de |expert_steer|")
    p.add_argument("--csv", required=True, help="Ruta a labels.csv")
    p.add_argument("--out", default=None, help="Ruta de salida (default: al lado del original)")
    p.add_argument("--steer_col", default="expert_steer")
    p.add_argument("--target_n", type=int, default=35000)
    p.add_argument("--seed", type=int, default=42)

    # permite ajustar fracciones desde CLI si quieres
    p.add_argument("--keep_straight", type=float, default=0.50)
    p.add_argument("--keep_near", type=float, default=0.70)

    args = p.parse_args()

    keep_frac = {
        "straight": args.keep_straight,
        "near":     args.keep_near,
        "soft":     1.0,
        "medium":   1.0,
        "hard":     1.0,
    }

    balance_labels_csv(
        csv_path=args.csv,
        out_path=args.out,
        steer_col=args.steer_col,
        seed=args.seed,
        target_n=args.target_n,
        keep_frac=keep_frac,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def make_out_path(in_csv: str, suffix: str) -> str:
    base, ext = os.path.splitext(in_csv)
    return f"{base}{suffix}{ext}"

def main():
    p = argparse.ArgumentParser(description="Balancea labels.csv (CARLA) por |expert_steer| y guarda *_balanced.csv al lado.")
    p.add_argument("--in_csv", required=True, help="Ruta a labels.csv (entrada)")
    p.add_argument("--out_csv", default=None, help="Ruta salida. Si no se da, crea *_balanced.csv junto al original.")
    p.add_argument("--steer_col", default="expert_steer", help="Columna de steer (default: expert_steer)")
    p.add_argument("--t0", type=float, default=0.02, help="Umbral bin straight (default 0.02)")
    p.add_argument("--t1", type=float, default=0.05, help="Umbral bin near_straight (default 0.05)")
    p.add_argument("--t2", type=float, default=0.2,  help="Umbral bin soft (default 0.2)")
    p.add_argument("--t3", type=float, default=0.5,  help="Umbral bin medium (default 0.5)")
    p.add_argument("--keep_straight", type=float, default=0.10, help="Fracción a conservar en straight (default 0.10)")
    p.add_argument("--keep_near", type=float, default=0.20, help="Fracción a conservar en near_straight (default 0.20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = p.parse_args()

    in_csv = args.in_csv
    out_csv = args.out_csv or make_out_path(in_csv, "_balanced")

    df = pd.read_csv(in_csv)
    df.columns = df.columns.str.strip().str.lower()

    steer_col = args.steer_col.strip().lower()
    if steer_col not in df.columns:
        raise KeyError(f"No existe la columna '{steer_col}'. Columnas disponibles: {list(df.columns)}")

    abs_steer = df[steer_col].abs()

    # bins: [0..t0], (t0..t1], (t1..t2], (t2..t3], (t3..1]
    bins = pd.cut(
        abs_steer,
        bins=[-1e-9, args.t0, args.t1, args.t2, args.t3, 1.0],
        labels=["straight", "near_straight", "soft", "medium", "hard"]
    )

    keep_frac = {
        "straight": args.keep_straight,
        "near_straight": args.keep_near,
        "soft": 1.0,
        "medium": 1.0,
        "hard": 1.0
    }

    # Resumen original
    print("=== ORIGINAL ===")
    print("Rows:", len(df))
    print(bins.value_counts().sort_index())

    # Muestreo por bin
    parts = []
    for k, frac in keep_frac.items():
        df_k = df[bins == k]
        if len(df_k) == 0:
            continue
        if frac < 1.0:
            df_k = df_k.sample(frac=frac, random_state=args.seed)
        parts.append(df_k)

    df_balanced = (
        pd.concat(parts, ignore_index=True)
          .sample(frac=1, random_state=args.seed)
          .reset_index(drop=True)
    )

    # Resumen balanceado
    abs_b = df_balanced[steer_col].abs()
    bins_b = pd.cut(
        abs_b,
        bins=[-1e-9, args.t0, args.t1, args.t2, args.t3, 1.0],
        labels=["straight", "near_straight", "soft", "medium", "hard"]
    )

    print("\n=== BALANCED ===")
    print("Rows:", len(df_balanced))
    print(bins_b.value_counts().sort_index())

    # Guardar
    df_balanced.to_csv(out_csv, index=False)
    print("\n✅ Guardado:", out_csv)

if __name__ == "__main__":
    main()

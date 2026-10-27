
import argparse

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





def set_nature_style():

    plt.rcParams.update({

        "font.family": "sans-serif",

        "font.sans-serif": ["Arial", "DejaVu Sans"],

        "font.size": 8,

        "axes.labelsize": 8,

        "xtick.labelsize": 8,

        "ytick.labelsize": 8,

        "axes.linewidth": 0.8,

        "xtick.major.width": 0.8,

        "ytick.major.width": 0.8,

        "xtick.major.size": 3,

        "ytick.major.size": 3,

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

        "savefig.dpi": 600,

    })





def save_all(fig, out_base):

    out_base = Path(out_base)

    fig.savefig(str(out_base) + ".png", dpi=600, bbox_inches="tight")

    fig.savefig(str(out_base) + ".tiff", dpi=600, bbox_inches="tight")

    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")

    print("Saved:", str(out_base) + ".png")

    print("Saved:", str(out_base) + ".tiff")

    print("Saved:", str(out_base) + ".pdf")





def read_one_run(run_dir, aggregation):

    run_dir = Path(run_dir)

    summary_path = run_dir / "aggregation_sweep_patient_channel" / "aggregation_sweep_summary.csv"



    if not summary_path.exists():

        raise FileNotFoundError(f"Missing: {summary_path}")



    df = pd.read_csv(summary_path)



    if "aggregation" not in df.columns:

        raise RuntimeError(f"'aggregation' column not found in {summary_path}")



    row = df[df["aggregation"].astype(str) == aggregation].copy()



    if row.empty:

        raise RuntimeError(f"Aggregation '{aggregation}' not found in {summary_path}")



    row = row.iloc[0]



    out = {

        "run_dir": str(run_dir),

        "aggregation": aggregation,

        "Accuracy": float(row["test_accuracy"]),

        "Balanced accuracy": float(row["test_balanced_accuracy"]),

        "F1 score": float(row["test_f1_score"]),

        "AUROC": float(row["test_auroc"]),

        "AUPRC": float(row["test_auprc"]),

        "Kappa": float(row["test_kappa"]),

        "MCC": float(row["test_mcc"]),

    }



    return out





def plot_bar_with_seed_points(df, metrics, out_base, y_min=None, y_max=1.0):

    set_nature_style()



    values = [df[m].values.astype(float) for m in metrics]

    means = [np.mean(v) for v in values]

    stds = [np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in values]



    x = np.arange(len(metrics))



    fig_width = max(2.2, 0.55 * len(metrics))

    fig_height = 2.45



    fig, ax = plt.subplots(figsize=(fig_width, fig_height))



    bar_color = "#4B3358"



    ax.bar(

        x,

        means,

        width=0.78,

        color=bar_color,

        edgecolor=bar_color,

        linewidth=0.8,

        zorder=2,

    )



    # Mean ± SD error bars

    if len(df) > 1:

        ax.errorbar(

            x,

            means,

            yerr=stds,

            fmt="none",

            ecolor="black",

            elinewidth=0.8,

            capsize=2,

            capthick=0.8,

            zorder=4,

        )



    # Individual seed/run dots

    rng = np.random.default_rng(42)



    for i, m in enumerate(metrics):

        vals = df[m].values.astype(float)

        jitter = rng.uniform(-0.13, 0.13, size=len(vals))



        ax.scatter(

            np.full(len(vals), x[i]) + jitter,

            vals,

            s=13,

            facecolors="none",

            edgecolors="#9A9A9A",

            linewidths=0.75,

            zorder=5,

        )



    ax.set_xticks(x)

    ax.set_xticklabels(metrics, rotation=45, ha="right", rotation_mode="anchor")



    all_vals = np.concatenate(values)



    if y_min is None:

        # automatic lower limit, but not below 0

        y_min = max(0.0, np.floor((np.nanmin(all_vals) - 0.05) * 10) / 10)



    ax.set_ylim(y_min, y_max)



    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)



    ax.yaxis.grid(True, linestyle="-", linewidth=0.35, alpha=0.25, zorder=1)

    ax.set_axisbelow(True)



    ax.set_ylabel("Performance")

    ax.set_title("")



    plt.tight_layout()

    save_all(fig, out_base)

    plt.close(fig)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dirs", nargs="+", required=True, help="Repeated seed/run directories")

    parser.add_argument("--aggregation", default="median",

                        choices=["mean", "median", "max", "p90", "p95", "top10_mean", "top20_mean", "top30_mean"])

    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--metrics", nargs="+", default=["Accuracy", "F1 score", "AUROC", "Kappa"])

    parser.add_argument("--y_min", type=float, default=None)

    parser.add_argument("--y_max", type=float, default=1.0)

    args = parser.parse_args()



    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    rows = []

    for run_dir in args.run_dirs:

        rows.append(read_one_run(run_dir, args.aggregation))



    df = pd.DataFrame(rows)



    csv_path = out_dir / f"repeated_runs_{args.aggregation}_metrics.csv"

    df.to_csv(csv_path, index=False)



    print("\nRepeated-run values:")

    print(df[["run_dir", "aggregation"] + args.metrics].to_string(index=False))



    print("\nMean ± SD:")

    for m in args.metrics:

        vals = df[m].values.astype(float)

        mean = np.mean(vals)

        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0

        print(f"{m}: {mean:.4f} ± {sd:.4f}")



    out_base = out_dir / f"nature_barplot_repeated_runs_{args.aggregation}"

    plot_bar_with_seed_points(

        df=df,

        metrics=args.metrics,

        out_base=out_base,

        y_min=args.y_min,

        y_max=args.y_max,

    )



    print("\nSaved CSV:", csv_path)





if __name__ == "__main__":

    main()


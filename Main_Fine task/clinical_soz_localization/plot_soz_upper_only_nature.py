
import argparse

from pathlib import Path

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





def set_nature_style():

    plt.rcParams.update({

        "font.family": "sans-serif",

        "font.sans-serif": ["Arial", "DejaVu Sans"],

        "font.size": 8,

        "axes.titlesize": 9,

        "axes.labelsize": 9,

        "xtick.labelsize": 7,

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

    fig.savefig(str(out_base) + ".pdf", dpi=600, bbox_inches="tight")

    print("Saved:", str(out_base) + ".png")

    print("Saved:", str(out_base) + ".tiff")

    print("Saved:", str(out_base) + ".pdf")





def is_ictal(x):

    s = str(x).lower()

    return ("task-ictal" in s) or ("ictal" in s and "interictal" not in s)





def run_no(x):

    s = str(x)

    m = re.search(r"run[-_]?(\d+)", s, flags=re.IGNORECASE)

    if m:

        return int(m.group(1))

    return 999999





def short_seizure_labels(seizure_ids):

    seizure_ids = sorted(seizure_ids, key=lambda z: (run_no(z), str(z)))

    return {sid: f"Seizure {i+1}" for i, sid in enumerate(seizure_ids)}





def aggregate_values(x, method):

    x = np.asarray(x, dtype=float)

    x = x[np.isfinite(x)]

    if len(x) == 0:

        return np.nan



    if method == "mean":

        return float(np.mean(x))

    if method == "median":

        return float(np.median(x))

    if method == "max":

        return float(np.max(x))

    if method == "p90":

        return float(np.percentile(x, 90))

    if method == "p95":

        return float(np.percentile(x, 95))

    if method.startswith("top"):

        pct = float(method.replace("top", "").replace("_mean", ""))

        k = max(1, int(np.ceil(len(x) * pct / 100.0)))

        return float(np.mean(np.sort(x)[-k:]))



    raise ValueError(f"Unknown aggregation method: {method}")





def prepare_matrix(pred_csv, patient_name, agg_method="median", prob_col="prob_soz", seizure_col="edf_name", top_n=25):

    df = pd.read_csv(pred_csv)



    required = ["patient_name", seizure_col, "channel_name", prob_col]

    missing = [c for c in required if c not in df.columns]

    if missing:

        raise RuntimeError(f"Missing columns in prediction CSV: {missing}")



    df = df[df["patient_name"].astype(str) == str(patient_name)].copy()

    if df.empty:

        raise RuntimeError(f"No data found for patient {patient_name}")



    df = df[df[seizure_col].apply(is_ictal)].copy()

    if df.empty:

        raise RuntimeError(f"No ictal rows found for patient {patient_name}")



    df = df.rename(columns={seizure_col: "seizure_id", prob_col: "prob"})



    # Aggregate within each seizure × channel

    rows = []

    for (seizure_id, channel), g in df.groupby(["seizure_id", "channel_name"]):

        rows.append({

            "seizure_id": seizure_id,

            "channel_name": channel,

            "score": aggregate_values(g["prob"].values, agg_method),

            "label": int(g["label"].max()) if "label" in g.columns else 0,

        })

    agg_df = pd.DataFrame(rows)



    # Rank channels by clinical label first, then score

    channel_rank = (

        agg_df.groupby("channel_name", as_index=False)

        .agg(

            mean_score=("score", "mean"),

            clinical_count=("label", "sum")

        )

    )



    channel_rank = channel_rank.sort_values(

        ["clinical_count", "mean_score", "channel_name"],

        ascending=[False, False, True]

    )



    if top_n is not None and top_n > 0:

        channel_rank = channel_rank.head(top_n).copy()



    channels = channel_rank["channel_name"].tolist()



    seizures = sorted(agg_df["seizure_id"].astype(str).unique(), key=lambda z: (run_no(z), z))

    seizure_map = short_seizure_labels(seizures)

    seizure_labels = [seizure_map[s] for s in seizures]



    s_to_i = {s: i for i, s in enumerate(seizures)}

    c_to_j = {c: j for j, c in enumerate(channels)}



    mat = np.full((len(seizures), len(channels)), np.nan, dtype=float)



    for _, r in agg_df.iterrows():

        s = str(r["seizure_id"])

        c = str(r["channel_name"])

        if s in s_to_i and c in c_to_j:

            mat[s_to_i[s], c_to_j[c]] = float(r["score"])



    return mat, channels, seizure_labels





def plot_upper_only(mat, channels, seizure_labels, out_base):

    set_nature_style()



    n_channels = len(channels)

    width = max(8.5, min(14.0, 0.42 * n_channels))

    height = 2.8



    fig, ax = plt.subplots(figsize=(width, height))



    im = ax.imshow(

        mat,

        aspect="auto",

        interpolation="nearest",

        cmap="viridis",

        vmin=0,

        vmax=1

    )



    # X axis

    ax.set_xlabel("SEEG channel")

    ax.set_xticks(np.arange(len(channels)))

    ax.set_xticklabels(channels, rotation=55, ha="right", rotation_mode="anchor")



    # Y axis: keep only seizure names, no axis label

    ax.set_yticks(np.arange(len(seizure_labels)))

    ax.set_yticklabels(seizure_labels)

    ax.set_ylabel("")



    # No title

    ax.set_title("")



    # Clean spines

    ax.spines["top"].set_visible(True)

    ax.spines["right"].set_visible(True)

    ax.spines["left"].set_visible(True)

    ax.spines["bottom"].set_visible(True)



    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    cbar.set_label("SOZ probability")

    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])



    plt.tight_layout()

    save_all(fig, out_base)

    plt.close(fig)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_csv", required=True)

    parser.add_argument("--patient_name", required=True)

    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--agg_method", default="median",

                        choices=["mean", "median", "max", "p90", "p95", "top10_mean", "top20_mean", "top30_mean"])

    parser.add_argument("--prob_col", default="prob_soz")

    parser.add_argument("--seizure_col_pred", default="edf_name")

    parser.add_argument("--top_n", type=int, default=25)

    args = parser.parse_args()



    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    mat, channels, seizure_labels = prepare_matrix(

        pred_csv=args.pred_csv,

        patient_name=args.patient_name,

        agg_method=args.agg_method,

        prob_col=args.prob_col,

        seizure_col=args.seizure_col_pred,

        top_n=args.top_n,

    )



    out_base = out_dir / f"soz_upper_only_{args.agg_method}_{args.patient_name}"

    plot_upper_only(mat, channels, seizure_labels, out_base)





if __name__ == "__main__":

    main()


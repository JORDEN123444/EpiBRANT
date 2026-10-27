
import argparse

from pathlib import Path

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import gridspec





def set_nature_style():

    plt.rcParams.update({

        "font.family": "sans-serif",

        "font.sans-serif": ["Arial", "DejaVu Sans"],

        "font.size": 8,

        "axes.titlesize": 10,

        "axes.labelsize": 9,

        "xtick.labelsize": 7,

        "ytick.labelsize": 8,

        "axes.linewidth": 0.8,

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

        "savefig.dpi": 600,

    })





def save_all(fig, base):

    base = Path(base)

    fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight")

    fig.savefig(str(base) + ".pdf", dpi=600, bbox_inches="tight")

    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")

    print("Saved:", str(base) + ".png")

    print("Saved:", str(base) + ".pdf")

    print("Saved:", str(base) + ".tiff")





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





def prepare_ictal_data(pred_csv, patient_name, agg_method, prob_col="prob_soz", seizure_col="edf_name"):

    pred = pd.read_csv(pred_csv)



    required = ["patient_name", seizure_col, "channel_name", prob_col, "label"]

    missing = [c for c in required if c not in pred.columns]

    if missing:

        raise RuntimeError(f"Missing columns in prediction CSV: {missing}")



    d = pred[pred["patient_name"].astype(str) == str(patient_name)].copy()

    if d.empty:

        raise RuntimeError(f"No rows found for patient {patient_name}")



    d = d[d[seizure_col].apply(is_ictal)].copy()

    if d.empty:

        raise RuntimeError(f"No ictal recordings found for patient {patient_name}")



    d = d.rename(columns={seizure_col: "seizure_id", prob_col: "prob"})



    # Revised upper heatmap:

    # For each seizure and channel, aggregate segment probabilities using top10_mean/p90/etc.

    rows = []

    for (seizure_id, channel), g in d.groupby(["seizure_id", "channel_name"]):

        rows.append({

            "seizure_id": seizure_id,

            "channel_name": channel,

            "score": aggregate_values(g["prob"].values, agg_method),

            "label": int(g["label"].max()),

            "soz": int(g["soz"].max()) if "soz" in g.columns else int(g["label"].max()),

            "resection": int(g["resection"].max()) if "resection" in g.columns else 0,

            "anatomical": str(g["anatomical"].iloc[0]) if "anatomical" in g.columns else "",

            "n_segments": int(len(g)),

        })



    ictal_ch = pd.DataFrame(rows)



    # Lower panel:

    # Clinical SOZ-label occurrence count = number of ictal recordings where the channel has label 1.

    clinical = (

        ictal_ch[ictal_ch["label"].astype(int) == 1]

        .groupby("channel_name", as_index=False)["seizure_id"]

        .nunique()

        .rename(columns={"seizure_id": "clinical_soz_count"})

    )



    return d, ictal_ch, clinical





def build_matrices(ictal_ch, clinical_count, top_n=25, sort_mode="clinical_then_score"):

    seizures = sorted(ictal_ch["seizure_id"].astype(str).unique(), key=lambda z: (run_no(z), z))

    seizure_label_map = short_seizure_labels(seizures)



    mean_score = (

        ictal_ch.groupby("channel_name", as_index=False)["score"]

        .mean()

        .rename(columns={"score": "mean_revised_score"})

    )



    all_channels = pd.DataFrame({"channel_name": sorted(ictal_ch["channel_name"].astype(str).unique())})



    ranking = all_channels.merge(mean_score, on="channel_name", how="left")

    ranking = ranking.merge(clinical_count, on="channel_name", how="left")

    ranking["clinical_soz_count"] = ranking["clinical_soz_count"].fillna(0)

    ranking["mean_revised_score"] = ranking["mean_revised_score"].fillna(0)



    if sort_mode == "clinical_then_score":

        ranking = ranking.sort_values(

            ["clinical_soz_count", "mean_revised_score", "channel_name"],

            ascending=[False, False, True]

        )

    elif sort_mode == "score_then_clinical":

        ranking = ranking.sort_values(

            ["mean_revised_score", "clinical_soz_count", "channel_name"],

            ascending=[False, False, True]

        )

    else:

        ranking = ranking.sort_values("channel_name")



    if top_n and top_n > 0:

        ranking = ranking.head(top_n).copy()



    channels = ranking["channel_name"].tolist()



    mat = np.full((len(seizures), len(channels)), np.nan, dtype=float)

    s_to_i = {s: i for i, s in enumerate(seizures)}

    c_to_j = {c: j for j, c in enumerate(channels)}



    for _, r in ictal_ch.iterrows():

        sid = str(r["seizure_id"])

        ch = str(r["channel_name"])

        if sid in s_to_i and ch in c_to_j:

            mat[s_to_i[sid], c_to_j[ch]] = float(r["score"])



    count_row = np.zeros((1, len(channels)), dtype=float)

    count_map = dict(zip(ranking["channel_name"], ranking["clinical_soz_count"]))

    for j, ch in enumerate(channels):

        count_row[0, j] = float(count_map.get(ch, 0))



    seizure_labels = [seizure_label_map[s] for s in seizures]



    return mat, count_row, channels, seizure_labels, ranking





def plot_nature_heatmap(mat, count_row, channels, seizure_labels, out_base, agg_method):

    set_nature_style()



    n_channels = len(channels)

    width = max(8.8, min(15.5, 0.40 * n_channels))

    height = 6.4



    fig = plt.figure(figsize=(width, height), constrained_layout=True)



    gs = gridspec.GridSpec(

        2, 2,

        figure=fig,

        width_ratios=[22, 0.65],

        height_ratios=[3.25, 1.15],

        wspace=0.04,

        hspace=0.20

    )



    ax1 = fig.add_subplot(gs[0, 0])

    cax1 = fig.add_subplot(gs[0, 1])



    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    cax2 = fig.add_subplot(gs[1, 1])



    cmap = "viridis"



    # Upper heatmap

    im1 = ax1.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=cmap)

    cb1 = fig.colorbar(im1, cax=cax1)

    cb1.set_label("Revised SOZ probability", rotation=90, labelpad=8)

    cb1.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])



    ax1.set_title(f"Figure 7A. Ictal-only predicted channel-level SOZ probability across seizures ({agg_method})",

                  fontweight="bold", pad=7)

    ax1.set_ylabel("Ictal seizure")

    ax1.set_yticks(np.arange(len(seizure_labels)))

    ax1.set_yticklabels(seizure_labels)

    ax1.set_xticks(np.arange(n_channels))

    ax1.set_xticklabels(channels, rotation=55, ha="right", rotation_mode="anchor")

    ax1.tick_params(axis="x", labelbottom=True)



    # Lower heatmap

    vmax_count = max(1, int(np.nanmax(count_row)))

    im2 = ax2.imshow(count_row, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax_count, cmap=cmap)

    cb2 = fig.colorbar(im2, cax=cax2)

    cb2.set_label("SOZ-label count", rotation=90, labelpad=8)

    cb2.set_ticks(np.arange(0, vmax_count + 1, 1))



    ax2.set_title("Figure 7B. Clinical SOZ-label occurrence count",

                  fontweight="bold", pad=7)

    ax2.set_ylabel("Clinical\nSOZ count")

    ax2.set_yticks([0])

    ax2.set_yticklabels(["Count"])

    ax2.set_xlabel("SEEG channel")

    ax2.set_xticks(np.arange(n_channels))

    ax2.set_xticklabels(channels, rotation=55, ha="right", rotation_mode="anchor")



    for j in range(n_channels):

        val = int(round(count_row[0, j]))

        color = "black" if val >= 0.6 * vmax_count else "white"

        ax2.text(j, 0, str(val), ha="center", va="center", fontsize=7, color=color)



    # Panel labels

    ax1.text(-0.11, 1.08, "a", transform=ax1.transAxes, fontsize=14, fontweight="bold")

    ax2.text(-0.11, 1.18, "b", transform=ax2.transAxes, fontsize=14, fontweight="bold")



    save_all(fig, out_base)

    plt.close(fig)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_csv", required=True)

    parser.add_argument("--patient_name", required=True)

    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--agg_method", default="top10_mean",

                        choices=["mean", "median", "max", "p90", "p95", "top10_mean", "top20_mean", "top30_mean"])

    parser.add_argument("--top_n", type=int, default=25)

    parser.add_argument("--prob_col", default="prob_soz")

    parser.add_argument("--seizure_col_pred", default="edf_name")

    parser.add_argument("--sort_mode", default="clinical_then_score",

                        choices=["clinical_then_score", "score_then_clinical", "channel_name"])

    args = parser.parse_args()



    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    _, ictal_ch, clinical_count = prepare_ictal_data(

        pred_csv=args.pred_csv,

        patient_name=args.patient_name,

        agg_method=args.agg_method,

        prob_col=args.prob_col,

        seizure_col=args.seizure_col_pred,

    )



    mat, count_row, channels, seizure_labels, ranking = build_matrices(

        ictal_ch=ictal_ch,

        clinical_count=clinical_count,

        top_n=args.top_n,

        sort_mode=args.sort_mode,

    )



    ranking_path = out_dir / f"fig7AB_revised_{args.agg_method}_{args.patient_name}_ranking.csv"

    ranking.to_csv(ranking_path, index=False)



    out_base = out_dir / f"fig7AB_revised_{args.agg_method}_clinical_soz_count_{args.patient_name}"



    plot_nature_heatmap(

        mat=mat,

        count_row=count_row,

        channels=channels,

        seizure_labels=seizure_labels,

        out_base=out_base,

        agg_method=args.agg_method,

    )



    print("Saved ranking:", ranking_path)

    print("Done.")





if __name__ == "__main__":

    main()


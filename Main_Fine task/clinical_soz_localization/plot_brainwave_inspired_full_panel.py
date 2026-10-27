
import argparse

from pathlib import Path

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import gridspec

from sklearn.metrics import (

    confusion_matrix,

    roc_auc_score,

    average_precision_score,

    precision_recall_curve,

)





def set_style():

    plt.rcParams.update({

        "font.family": "sans-serif",

        "font.sans-serif": ["Arial", "DejaVu Sans"],

        "font.size": 7,

        "axes.titlesize": 8,

        "axes.labelsize": 8,

        "xtick.labelsize": 6,

        "ytick.labelsize": 6,

        "axes.linewidth": 0.7,

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

        "savefig.dpi": 600,

    })





def is_ictal(x):

    s = str(x).lower()

    return ("task-ictal" in s) or ("ictal" in s and "interictal" not in s)





def run_no(x):

    s = str(x)

    m = re.search(r"run[-_]?(\d+)", s, flags=re.IGNORECASE)

    if m:

        return int(m.group(1))

    return 999999





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





def load_threshold(run_dir, agg_method):

    summary_path = Path(run_dir) / "aggregation_sweep_patient_channel" / "aggregation_sweep_summary.csv"

    if not summary_path.exists():

        return 0.5



    df = pd.read_csv(summary_path)

    row = df[df["aggregation"].astype(str) == str(agg_method)]

    if row.empty:

        return 0.5



    for col in ["test_threshold", "val_threshold", "threshold"]:

        if col in row.columns:

            return float(row.iloc[0][col])



    return 0.5





def patient_performance_points(channel_scores_csv, threshold):

    ch = pd.read_csv(channel_scores_csv)



    rows = []

    for patient, g in ch.groupby("patient_name"):

        y = g["label"].astype(int).values

        p = g["prob_soz"].astype(float).values

        pred = (p >= threshold).astype(int)



        if len(np.unique(y)) < 2:

            continue



        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()



        sensitivity = tp / max(1, tp + fn)

        specificity = tn / max(1, tn + fp)



        try:

            auroc = roc_auc_score(y, p)

        except Exception:

            auroc = np.nan



        try:

            ap = average_precision_score(y, p)

        except Exception:

            ap = np.nan



        rows.append({

            "patient_name": patient,

            "sensitivity": sensitivity,

            "specificity": specificity,

            "auroc": auroc,

            "ap": ap,

        })



    return pd.DataFrame(rows)





def prepare_patient_heatmaps(pred_csv, patient_name, agg_method, top_n, prob_col, seizure_col, sort_mode):

    pred = pd.read_csv(pred_csv)



    required = ["patient_name", seizure_col, "channel_name", prob_col, "label"]

    missing = [c for c in required if c not in pred.columns]

    if missing:

        raise RuntimeError(f"Missing columns in prediction CSV: {missing}")



    d = pred[pred["patient_name"].astype(str) == str(patient_name)].copy()

    if d.empty:

        raise RuntimeError(f"No rows found for patient: {patient_name}")



    d = d[d[seizure_col].apply(is_ictal)].copy()

    if d.empty:

        raise RuntimeError(f"No ictal rows found for patient: {patient_name}")



    d = d.rename(columns={seizure_col: "seizure_id", prob_col: "prob"})



    rows = []

    for (seizure_id, channel), g in d.groupby(["seizure_id", "channel_name"]):

        rows.append({

            "seizure_id": str(seizure_id),

            "channel_name": str(channel),

            "score": aggregate_values(g["prob"].values, agg_method),

            "label": int(g["label"].max()),

            "n_segments": int(len(g)),

        })



    ictal_ch = pd.DataFrame(rows)



    clinical_count = (

        ictal_ch[ictal_ch["label"].astype(int) == 1]

        .groupby("channel_name", as_index=False)["seizure_id"]

        .nunique()

        .rename(columns={"seizure_id": "clinical_soz_count"})

    )



    mean_score = (

        ictal_ch.groupby("channel_name", as_index=False)["score"]

        .mean()

        .rename(columns={"score": "mean_score"})

    )



    all_channels = pd.DataFrame({"channel_name": sorted(ictal_ch["channel_name"].unique())})

    ranking = all_channels.merge(mean_score, on="channel_name", how="left")

    ranking = ranking.merge(clinical_count, on="channel_name", how="left")

    ranking["clinical_soz_count"] = ranking["clinical_soz_count"].fillna(0)

    ranking["mean_score"] = ranking["mean_score"].fillna(0)



    if sort_mode == "clinical_then_score":

        ranking = ranking.sort_values(

            ["clinical_soz_count", "mean_score", "channel_name"],

            ascending=[False, False, True]

        )

    elif sort_mode == "score_then_clinical":

        ranking = ranking.sort_values(

            ["mean_score", "clinical_soz_count", "channel_name"],

            ascending=[False, False, True]

        )

    else:

        ranking = ranking.sort_values("channel_name")



    if top_n > 0:

        ranking = ranking.head(top_n).copy()



    channels = ranking["channel_name"].tolist()

    seizures = sorted(ictal_ch["seizure_id"].unique(), key=lambda z: (run_no(z), z))

    seizure_labels = [f"Seizure {i+1}" for i in range(len(seizures))]



    mat = np.full((len(channels), len(seizures)), np.nan)

    c_to_i = {c: i for i, c in enumerate(channels)}

    s_to_j = {s: j for j, s in enumerate(seizures)}



    for _, r in ictal_ch.iterrows():

        ch = r["channel_name"]

        sz = r["seizure_id"]

        if ch in c_to_i and sz in s_to_j:

            mat[c_to_i[ch], s_to_j[sz]] = float(r["score"])



    count_vec = np.zeros(len(channels))

    count_map = dict(zip(ranking["channel_name"], ranking["clinical_soz_count"]))

    for i, ch in enumerate(channels):

        count_vec[i] = float(count_map.get(ch, 0))



    return mat, count_vec, channels, seizure_labels, ranking





def save_all(fig, out_base):

    out_base = Path(out_base)

    fig.savefig(str(out_base) + ".png", dpi=600, bbox_inches="tight")

    fig.savefig(str(out_base) + ".pdf", dpi=600, bbox_inches="tight")

    fig.savefig(str(out_base) + ".tiff", dpi=600, bbox_inches="tight")

    print("Saved:", str(out_base) + ".png")

    print("Saved:", str(out_base) + ".pdf")

    print("Saved:", str(out_base) + ".tiff")





def plot_full_panel(perf, mat, count_vec, channels, seizure_labels, out_base, agg_method):

    set_style()



    n_channels = len(channels)

    fig = plt.figure(figsize=(12.5, 5.8), constrained_layout=True)



    outer = gridspec.GridSpec(

        1, 3,

        figure=fig,

        width_ratios=[1.15, 2.35, 1.35],

        wspace=0.22

    )



    # Panel a: performance scatter

    gs_a = gridspec.GridSpecFromSubplotSpec(

        2, 1,

        subplot_spec=outer[0],

        hspace=0.35

    )



    ax_a1 = fig.add_subplot(gs_a[0, 0])

    ax_a2 = fig.add_subplot(gs_a[1, 0])



    if len(perf) > 0:

        ax_a1.scatter(perf["sensitivity"], perf["specificity"], s=28, edgecolor="black", linewidth=0.3)

        ax_a2.scatter(perf["auroc"], perf["ap"], s=28, edgecolor="black", linewidth=0.3)



        for _, r in perf.iterrows():

            label = str(r["patient_name"]).replace("sub-", "")

            ax_a1.text(r["sensitivity"] + 0.005, r["specificity"] + 0.005, label, fontsize=5)

            ax_a2.text(r["auroc"] + 0.005, r["ap"] + 0.005, label, fontsize=5)



    ax_a1.set_xlabel("Sensitivity")

    ax_a1.set_ylabel("Specificity")

    ax_a1.set_xlim(0, 1.02)

    ax_a1.set_ylim(0, 1.02)

    ax_a1.grid(True, alpha=0.25, linewidth=0.4)



    ax_a2.set_xlabel("AUROC")

    ax_a2.set_ylabel("AP")

    ax_a2.set_xlim(0, 1.02)

    ax_a2.set_ylim(0, 1.02)

    ax_a2.grid(True, alpha=0.25, linewidth=0.4)



    ax_a1.text(-0.28, 1.15, "a", transform=ax_a1.transAxes, fontsize=12, fontweight="bold")



    # Panel b: channel-level predicted probability heatmap

    gs_b = gridspec.GridSpecFromSubplotSpec(

        1, 2,

        subplot_spec=outer[1],

        width_ratios=[22, 0.9],

        wspace=0.05

    )



    ax_b = fig.add_subplot(gs_b[0, 0])

    cax_b = fig.add_subplot(gs_b[0, 1])



    im_b = ax_b.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")

    cb_b = fig.colorbar(im_b, cax=cax_b)

    cb_b.set_label("Predicted SOZ probability")



    ax_b.set_title(f"Channel-level predicted SOZ probability ({agg_method})", pad=6)

    ax_b.set_xlabel("Multiple ictal seizures")

    ax_b.set_ylabel("Channels")

    ax_b.set_xticks(np.arange(len(seizure_labels)))

    ax_b.set_xticklabels(seizure_labels, rotation=35, ha="right")

    ax_b.set_yticks(np.arange(n_channels))

    ax_b.set_yticklabels(channels, fontsize=5)



    ax_b.text(-0.12, 1.04, "b", transform=ax_b.transAxes, fontsize=12, fontweight="bold")



    # Panel c: clinical SOZ-label count heatmap

    gs_c = gridspec.GridSpecFromSubplotSpec(

        1, 2,

        subplot_spec=outer[2],

        width_ratios=[8, 0.8],

        wspace=0.08

    )



    ax_c = fig.add_subplot(gs_c[0, 0])

    cax_c = fig.add_subplot(gs_c[0, 1])



    count_mat = count_vec.reshape(-1, 1)

    vmax_count = max(1, int(np.nanmax(count_vec)))



    im_c = ax_c.imshow(count_mat, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax_count, cmap="viridis")

    cb_c = fig.colorbar(im_c, cax=cax_c)

    cb_c.set_label("SOZ-label count")

    cb_c.set_ticks(np.arange(0, vmax_count + 1, 1))



    ax_c.set_title("Clinical SOZ-label\noccurrence count", pad=6)

    ax_c.set_xticks([0])

    ax_c.set_xticklabels(["Count"])

    ax_c.set_yticks(np.arange(n_channels))

    ax_c.set_yticklabels([])

    ax_c.tick_params(axis="y", length=0)



    for i, v in enumerate(count_vec):

        color = "black" if v >= 0.6 * vmax_count else "white"

        ax_c.text(0, i, str(int(v)), ha="center", va="center", fontsize=6, color=color)



    ax_c.text(-0.55, 1.04, "c", transform=ax_c.transAxes, fontsize=12, fontweight="bold")



    fig.suptitle(

        "BrainWave-inspired SEEG SOZ localization visualization",

        fontsize=11,

        fontweight="bold",

        y=1.03

    )



    save_all(fig, out_base)

    plt.close(fig)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", required=True)

    parser.add_argument("--patient_name", required=True)

    parser.add_argument("--agg_method", default="median",

                        choices=["mean", "median", "max", "p90", "p95", "top10_mean", "top20_mean", "top30_mean"])

    parser.add_argument("--top_n", type=int, default=25)

    parser.add_argument("--prob_col", default="prob_soz")

    parser.add_argument("--seizure_col_pred", default="edf_name")

    parser.add_argument("--sort_mode", default="clinical_then_score",

                        choices=["clinical_then_score", "score_then_clinical", "channel_name"])

    args = parser.parse_args()



    run_dir = Path(args.run_dir)

    out_dir = run_dir / "figures_soz_localization"

    out_dir.mkdir(parents=True, exist_ok=True)



    channel_scores_csv = run_dir / "aggregation_sweep_patient_channel" / f"test_channel_scores_{args.agg_method}.csv"

    if not channel_scores_csv.exists():

        raise RuntimeError(f"Missing channel score file: {channel_scores_csv}")



    threshold = load_threshold(run_dir, args.agg_method)

    perf = patient_performance_points(channel_scores_csv, threshold)



    pred_csv = run_dir / "test_segment_predictions.csv"



    mat, count_vec, channels, seizure_labels, ranking = prepare_patient_heatmaps(

        pred_csv=pred_csv,

        patient_name=args.patient_name,

        agg_method=args.agg_method,

        top_n=args.top_n,

        prob_col=args.prob_col,

        seizure_col=args.seizure_col_pred,

        sort_mode=args.sort_mode,

    )



    ranking_path = out_dir / f"brainwave_inspired_{args.agg_method}_{args.patient_name}_ranking.csv"

    ranking.to_csv(ranking_path, index=False)



    out_base = out_dir / f"brainwave_inspired_{args.agg_method}_{args.patient_name}"



    plot_full_panel(

        perf=perf,

        mat=mat,

        count_vec=count_vec,

        channels=channels,

        seizure_labels=seizure_labels,

        out_base=out_base,

        agg_method=args.agg_method,

    )



    print("Saved ranking:", ranking_path)

    print("Done.")





if __name__ == "__main__":

    main()


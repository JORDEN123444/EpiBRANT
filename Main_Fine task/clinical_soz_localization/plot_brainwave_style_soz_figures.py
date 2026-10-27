
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





def save_figure(fig, out_base):

    out_base = Path(out_base)

    fig.savefig(str(out_base) + ".png", dpi=600, bbox_inches="tight")

    fig.savefig(str(out_base) + ".pdf", dpi=600, bbox_inches="tight")

    fig.savefig(str(out_base) + ".tiff", dpi=600, bbox_inches="tight")

    print("Saved:", str(out_base) + ".png")

    print("Saved:", str(out_base) + ".pdf")

    print("Saved:", str(out_base) + ".tiff")





def is_ictal_name(x):

    s = str(x).lower()

    return ("task-ictal" in s) or ("ictal" in s and "interictal" not in s)





def run_number_from_name(x):

    s = str(x)

    m = re.search(r"run[-_]?(\d+)", s, flags=re.IGNORECASE)

    if m:

        return int(m.group(1))

    return 999999





def short_seizure_labels(seizure_ids):

    seizure_ids = sorted(seizure_ids, key=lambda z: (run_number_from_name(z), str(z)))

    return {sid: f"Seizure {i+1}" for i, sid in enumerate(seizure_ids)}





def prepare_ictal_predictions(pred_csv, patient_name, prob_col="prob_soz", seizure_col="edf_name"):

    pred = pd.read_csv(pred_csv)



    required = ["patient_name", seizure_col, "channel_name", prob_col]

    missing = [c for c in required if c not in pred.columns]

    if missing:

        raise RuntimeError(f"Missing columns in prediction CSV: {missing}")



    pred = pred[pred["patient_name"].astype(str) == str(patient_name)].copy()

    if pred.empty:

        raise RuntimeError(f"No prediction rows found for patient: {patient_name}")



    pred = pred[pred[seizure_col].apply(is_ictal_name)].copy()

    if pred.empty:

        raise RuntimeError(f"No ictal rows found for patient: {patient_name}. Check edf_name values.")



    pred = pred.rename(columns={seizure_col: "seizure_id", prob_col: "prob"})



    pred_agg = (

        pred.groupby(["patient_name", "seizure_id", "channel_name"], as_index=False)["prob"]

        .mean()

        .rename(columns={"prob": "mean_prob"})

    )



    return pred, pred_agg





def make_clinical_soz_count(pred_ictal, patient_name):

    if "label" not in pred_ictal.columns:

        raise RuntimeError("Prediction CSV must contain 'label' column for clinical SOZ-label count figure.")



    d = pred_ictal[pred_ictal["patient_name"].astype(str) == str(patient_name)].copy()

    d = d[d["label"].astype(int) == 1].copy()



    if d.empty:

        return pd.DataFrame(columns=["patient_name", "channel_name", "clinical_soz_count"])



    # Count in how many ictal recordings each channel is clinically labelled SOZ.

    # Drop duplicates because each channel has multiple segments.

    d = d[["patient_name", "seizure_id", "channel_name"]].drop_duplicates()



    cnt = (

        d.groupby(["patient_name", "channel_name"], as_index=False)["seizure_id"]

        .nunique()

        .rename(columns={"seizure_id": "clinical_soz_count"})

    )

    return cnt





def load_true_onset_count(onset_csv, patient_name, seizure_col_onset="seizure_id"):

    onset = pd.read_csv(onset_csv)



    required = ["patient_name", seizure_col_onset, "channel_name"]

    missing = [c for c in required if c not in onset.columns]

    if missing:

        raise RuntimeError(f"Missing columns in onset CSV: {missing}")



    onset = onset[onset["patient_name"].astype(str) == str(patient_name)].copy()

    if onset.empty:

        raise RuntimeError(f"No onset rows found for patient: {patient_name}")



    onset = onset.rename(columns={seizure_col_onset: "seizure_id"})



    if "is_onset" not in onset.columns:

        onset["is_onset"] = 1



    onset = onset[onset["is_onset"].astype(int) == 1].copy()

    onset = onset[onset["seizure_id"].apply(is_ictal_name)].copy()



    cnt = (

        onset[["patient_name", "seizure_id", "channel_name"]]

        .drop_duplicates()

        .groupby(["patient_name", "channel_name"], as_index=False)["seizure_id"]

        .nunique()

        .rename(columns={"seizure_id": "onset_frequency"})

    )

    return cnt





def build_matrices(pred_agg, count_df, count_col, top_n=25):

    seizures = sorted(pred_agg["seizure_id"].astype(str).unique(), key=lambda z: (run_number_from_name(z), z))

    label_map = short_seizure_labels(seizures)



    mean_prob = (

        pred_agg.groupby("channel_name", as_index=False)["mean_prob"]

        .mean()

        .rename(columns={"mean_prob": "mean_prob_all"})

    )



    all_channels = pd.DataFrame({"channel_name": sorted(pred_agg["channel_name"].astype(str).unique())})



    ranking = all_channels.merge(mean_prob, on="channel_name", how="left")

    ranking = ranking.merge(count_df[["channel_name", count_col]], on="channel_name", how="left")

    ranking[count_col] = ranking[count_col].fillna(0)

    ranking["mean_prob_all"] = ranking["mean_prob_all"].fillna(0)



    ranking = ranking.sort_values(

        [count_col, "mean_prob_all", "channel_name"],

        ascending=[False, False, True]

    ).reset_index(drop=True)



    if top_n and top_n > 0:

        ranking = ranking.head(top_n).copy()



    channels = ranking["channel_name"].tolist()



    prob_mat = np.full((len(seizures), len(channels)), np.nan, dtype=float)

    seizure_to_i = {s: i for i, s in enumerate(seizures)}

    channel_to_j = {c: j for j, c in enumerate(channels)}



    for _, r in pred_agg.iterrows():

        sid = str(r["seizure_id"])

        ch = str(r["channel_name"])

        if sid in seizure_to_i and ch in channel_to_j:

            prob_mat[seizure_to_i[sid], channel_to_j[ch]] = float(r["mean_prob"])



    count_row = np.zeros((1, len(channels)), dtype=float)

    count_map = dict(zip(ranking["channel_name"], ranking[count_col]))

    for j, ch in enumerate(channels):

        count_row[0, j] = float(count_map.get(ch, 0))



    seizure_labels = [label_map[s] for s in seizures]

    return prob_mat, count_row, channels, seizure_labels, ranking





def plot_two_heatmaps(

    prob_mat,

    count_row,

    channels,

    seizure_labels,

    out_base,

    top_title,

    bottom_title,

    bottom_ylabel,

    bottom_cbar_label,

    figure7_labels=False,

    count_max=None,

):

    set_nature_style()



    n_channels = len(channels)

    width = max(8.5, min(15.0, 0.38 * n_channels))

    height = 6.2



    fig = plt.figure(figsize=(width, height), constrained_layout=True)

    gs = gridspec.GridSpec(

        2, 2,

        figure=fig,

        width_ratios=[20, 0.55],

        height_ratios=[3.2, 1.15],

        wspace=0.04,

        hspace=0.18

    )



    ax1 = fig.add_subplot(gs[0, 0])

    cax1 = fig.add_subplot(gs[0, 1])

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    cax2 = fig.add_subplot(gs[1, 1])



    cmap = "viridis"



    im1 = ax1.imshow(prob_mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=cmap)

    cb1 = fig.colorbar(im1, cax=cax1)

    cb1.set_label("Probability", rotation=90, labelpad=8)

    cb1.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])



    ax1.set_title(top_title, fontweight="bold", pad=7)

    ax1.set_ylabel("Seizure")

    ax1.set_yticks(np.arange(len(seizure_labels)))

    ax1.set_yticklabels(seizure_labels)

    ax1.set_xticks(np.arange(n_channels))

    ax1.set_xticklabels(channels, rotation=55, ha="right", rotation_mode="anchor")

    ax1.tick_params(axis="x", labelbottom=True)



    vmax_count = count_max if count_max is not None else max(1, int(np.nanmax(count_row)))

    im2 = ax2.imshow(count_row, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax_count, cmap=cmap)

    cb2 = fig.colorbar(im2, cax=cax2)

    cb2.set_label(bottom_cbar_label, rotation=90, labelpad=8)

    cb2.set_ticks(np.arange(0, vmax_count + 1, 1))



    ax2.set_title(bottom_title, fontweight="bold", pad=7)

    ax2.set_ylabel(bottom_ylabel)

    ax2.set_yticks([0])

    ax2.set_yticklabels(["Count"])

    ax2.set_xlabel("Channel")

    ax2.set_xticks(np.arange(n_channels))

    ax2.set_xticklabels(channels, rotation=55, ha="right", rotation_mode="anchor")



    # Annotate counts

    for j in range(n_channels):

        val = int(round(count_row[0, j]))

        txt_color = "black" if val >= 0.6 * vmax_count else "white"

        if vmax_count <= 1 and val == 0:

            txt_color = "white"

        ax2.text(j, 0, str(val), ha="center", va="center", fontsize=7, color=txt_color)



    # Panel labels

    if figure7_labels:

        box = dict(boxstyle="square,pad=0.25", facecolor="#103b63", edgecolor="#103b63")

        ax1.text(-0.13, 1.08, "Figure 7A", transform=ax1.transAxes, fontsize=10,

                 fontweight="bold", color="white", va="center", ha="left", bbox=box)

        ax2.text(-0.13, 1.17, "Figure 7B", transform=ax2.transAxes, fontsize=10,

                 fontweight="bold", color="white", va="center", ha="left", bbox=box)

    else:

        ax1.text(-0.11, 1.06, "a", transform=ax1.transAxes, fontsize=14,

                 fontweight="bold", va="center", ha="left")

        ax2.text(-0.11, 1.15, "b", transform=ax2.transAxes, fontsize=14,

                 fontweight="bold", va="center", ha="left")



    save_figure(fig, out_base)

    plt.close(fig)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_csv", required=True)

    parser.add_argument("--patient_name", required=True)

    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--onset_csv", default=None, help="True seizure-wise onset annotations CSV")

    parser.add_argument("--top_n", type=int, default=25)

    parser.add_argument("--prob_col", default="prob_soz")

    parser.add_argument("--seizure_col_pred", default="edf_name")

    parser.add_argument("--seizure_col_onset", default="seizure_id")

    args = parser.parse_args()



    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    pred_ictal, pred_agg = prepare_ictal_predictions(

        pred_csv=args.pred_csv,

        patient_name=args.patient_name,

        prob_col=args.prob_col,

        seizure_col=args.seizure_col_pred,

    )



    # -------------------------

    # Figure 1: BrainWave-style onset-frequency figure

    # -------------------------

    if args.onset_csv is not None:

        onset_count = load_true_onset_count(

            onset_csv=args.onset_csv,

            patient_name=args.patient_name,

            seizure_col_onset=args.seizure_col_onset,

        )



        prob_mat, count_row, channels, seizure_labels, ranking = build_matrices(

            pred_agg=pred_agg,

            count_df=onset_count,

            count_col="onset_frequency",

            top_n=args.top_n,

        )



        ranking.to_csv(out_dir / f"brainwave_style_onset_frequency_{args.patient_name}_ranking.csv", index=False)



        plot_two_heatmaps(

            prob_mat=prob_mat,

            count_row=count_row,

            channels=channels,

            seizure_labels=seizure_labels,

            out_base=out_dir / f"fig7_brainwave_style_onset_frequency_{args.patient_name}",

            top_title="Predicted channel-level SOZ probability across seizures",

            bottom_title="Onset channel frequency",

            bottom_ylabel="Onset\ncount",

            bottom_cbar_label="Times serving as onset site",

            figure7_labels=False,

            count_max=max(1, int(np.nanmax(count_row))),

        )



    else:

        print("No --onset_csv provided. Skipping BrainWave-style true onset-frequency figure.")



    # -------------------------

    # Figure 2: Figure 7A/7B clinical SOZ-label occurrence count

    # -------------------------

    clinical_count = make_clinical_soz_count(pred_ictal, args.patient_name)



    prob_mat, count_row, channels, seizure_labels, ranking = build_matrices(

        pred_agg=pred_agg,

        count_df=clinical_count.rename(columns={"clinical_soz_count": "clinical_count"}),

        count_col="clinical_count",

        top_n=args.top_n,

    )



    ranking.to_csv(out_dir / f"figure7AB_clinical_soz_count_{args.patient_name}_ranking.csv", index=False)



    plot_two_heatmaps(

        prob_mat=prob_mat,

        count_row=count_row,

        channels=channels,

        seizure_labels=seizure_labels,

        out_base=out_dir / f"fig7AB_clinical_soz_label_count_{args.patient_name}",

        top_title="Predicted channel-level SOZ probability across seizures",

        bottom_title="Clinical SOZ-label occurrence count",

        bottom_ylabel="SOZ-label\ncount",

        bottom_cbar_label="SOZ-label count",

        figure7_labels=True,

        count_max=max(1, int(np.nanmax(count_row))),

    )



    print("\nDone.")

    print("Output directory:", out_dir)





if __name__ == "__main__":

    main()


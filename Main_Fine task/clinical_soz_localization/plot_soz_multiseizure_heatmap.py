
import argparse

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





def save_all(fig, base):

    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")

    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")

    fig.savefig(str(base) + ".pdf", dpi=600, bbox_inches="tight")

    print("Saved:", str(base) + ".png")





def style():

    plt.rcParams.update({

        "font.family": "sans-serif",

        "font.size": 8,

        "axes.labelsize": 8,

        "xtick.labelsize": 7,

        "ytick.labelsize": 7,

        "axes.linewidth": 0.8,

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

    })





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_csv", required=True,

                        help="test_segment_predictions.csv or similar")

    parser.add_argument("--onset_csv", required=True,

                        help="CSV with seizure onset annotations")

    parser.add_argument("--patient_name", required=True)

    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--top_n", type=int, default=30)

    parser.add_argument("--seizure_col_pred", default="edf_name")

    parser.add_argument("--seizure_col_onset", default="seizure_id")

    parser.add_argument("--prob_col", default="prob_soz")

    args = parser.parse_args()



    style()



    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    pred = pd.read_csv(args.pred_csv)

    onset = pd.read_csv(args.onset_csv)



    pred = pred[pred["patient_name"].astype(str) == str(args.patient_name)].copy()

    onset = onset[onset["patient_name"].astype(str) == str(args.patient_name)].copy()



    if pred.empty:

        raise RuntimeError(f"No prediction rows found for patient {args.patient_name}")

    if onset.empty:

        raise RuntimeError(f"No onset rows found for patient {args.patient_name}")



    # rename seizure columns to common name

    pred = pred.rename(columns={args.seizure_col_pred: "seizure_id"})

    onset = onset.rename(columns={args.seizure_col_onset: "seizure_id"})



    # If onset CSV has no explicit is_onset column, create one

    if "is_onset" not in onset.columns:

        onset["is_onset"] = 1



    # Aggregate predicted probability per seizure-channel

    pred_agg = (

        pred.groupby(["patient_name", "seizure_id", "channel_name"], as_index=False)[args.prob_col]

        .mean()

        .rename(columns={args.prob_col: "mean_prob"})

    )



    # Aggregate onset count per channel

    onset_count = (

        onset.groupby(["patient_name", "channel_name"], as_index=False)["is_onset"]

        .sum()

        .rename(columns={"is_onset": "onset_count"})

    )



    # Mean probability across all seizures for sorting

    mean_prob_ch = (

        pred_agg.groupby(["patient_name", "channel_name"], as_index=False)["mean_prob"]

        .mean()

        .rename(columns={"mean_prob": "mean_prob_all"})

    )



    ranking = onset_count.merge(mean_prob_ch, on=["patient_name", "channel_name"], how="outer").fillna(0.0)

    ranking = ranking.sort_values(["onset_count", "mean_prob_all", "channel_name"], ascending=[False, False, True])



    channels = ranking["channel_name"].tolist()

    if args.top_n is not None and args.top_n > 0:

        channels = channels[:args.top_n]



    pred_agg = pred_agg[pred_agg["channel_name"].isin(channels)].copy()

    onset_count = onset_count[onset_count["channel_name"].isin(channels)].copy()



    seizures = sorted(pred_agg["seizure_id"].astype(str).unique().tolist())



    # Upper heatmap: seizures x channels

    prob_mat = np.full((len(seizures), len(channels)), np.nan, dtype=float)

    seizure_to_i = {s: i for i, s in enumerate(seizures)}

    channel_to_j = {c: j for j, c in enumerate(channels)}



    for _, r in pred_agg.iterrows():

        i = seizure_to_i[str(r["seizure_id"])]

        j = channel_to_j[r["channel_name"]]

        prob_mat[i, j] = float(r["mean_prob"])



    # Lower heatmap: 1 x channels (count of onset occurrences)

    onset_row = np.zeros((1, len(channels)), dtype=float)

    onset_map = {r["channel_name"]: float(r["onset_count"]) for _, r in onset_count.iterrows()}

    for j, ch in enumerate(channels):

        onset_row[0, j] = onset_map.get(ch, 0.0)



    # Normalize onset row for display if needed

    onset_max = max(1.0, onset_row.max())



    # Figure

    fig = plt.figure(figsize=(max(7.5, 0.28 * len(channels)), 5.8))

    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4.5, 1.0], hspace=0.18)



    # Upper panel

    ax1 = fig.add_subplot(gs[0, 0])

    im1 = ax1.imshow(prob_mat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)

    ax1.set_ylabel("Seizure")

    ax1.set_yticks(np.arange(len(seizures)))

    ax1.set_yticklabels(seizures)

    ax1.set_xticks(np.arange(len(channels)))

    ax1.set_xticklabels(channels, rotation=60, ha="right")

    ax1.set_title("Predicted channel-level SOZ probability across seizures")



    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.025, pad=0.01)

    cbar1.set_label("Probability")



    # Lower panel

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    im2 = ax2.imshow(onset_row, aspect="auto", interpolation="nearest", vmin=0.0, vmax=onset_max)

    ax2.set_ylabel("Onset\ncount")

    ax2.set_yticks([0])

    ax2.set_yticklabels(["Count"])

    ax2.set_xticks(np.arange(len(channels)))

    ax2.set_xticklabels(channels, rotation=60, ha="right")



    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.025, pad=0.01)

    cbar2.set_label("Times serving as onset site")



    # annotate lower panel values

    for j in range(len(channels)):

        ax2.text(j, 0, str(int(onset_row[0, j])), ha="center", va="center", fontsize=7)



    fig.tight_layout()

    base = out_dir / f"fig7_multiseizure_heatmap_{args.patient_name}"

    save_all(fig, base)

    plt.close(fig)



    # also save ranked table

    out_tbl = pd.DataFrame({

        "channel_name": channels,

        "mean_prob_all": [mean_prob_ch.set_index("channel_name").reindex(ch)["mean_prob_all"] if False else 0 for ch in channels]

    })



    ranking_out = ranking[ranking["channel_name"].isin(channels)].copy()

    ranking_out.to_csv(out_dir / f"fig7_multiseizure_heatmap_{args.patient_name}_ranking.csv", index=False)

    print("Saved:", out_dir / f"fig7_multiseizure_heatmap_{args.patient_name}_ranking.csv")





if __name__ == "__main__":

    main()



import argparse

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, confusion_matrix





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

        "legend.fontsize": 7,

        "axes.linewidth": 0.8,

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

    })





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", required=True)

    parser.add_argument("--split", default="test")

    parser.add_argument("--patient_name", default=None)

    parser.add_argument("--top_n", type=int, default=25)

    args = parser.parse_args()



    style()



    run_dir = Path(args.run_dir)

    fig_dir = run_dir / "figures_soz_localization"

    fig_dir.mkdir(parents=True, exist_ok=True)



    seg = pd.read_csv(run_dir / f"{args.split}_segment_predictions.csv")

    ch = pd.read_csv(run_dir / f"{args.split}_channel_localization_scores.csv")

    ch_m = pd.read_csv(run_dir / f"{args.split}_channel_metrics.csv").iloc[0]

    topk_path = run_dir / f"{args.split}_topk_hit_rate.csv"



    th = float(ch_m["threshold"])



    # ROC

    y = ch["label"].values.astype(int)

    p = ch["prob_soz"].values.astype(float)



    fpr, tpr, _ = roc_curve(y, p)

    roc_auc = auc(fpr, tpr)



    fig, ax = plt.subplots(figsize=(4.6, 4.0))

    ax.plot(fpr, tpr, linewidth=1.8, label=f"Channel AUROC={roc_auc:.3f}")

    ax.plot([0,1], [0,1], linestyle="--", linewidth=0.8)

    ax.set_xlabel("False positive rate")

    ax.set_ylabel("True positive rate")

    ax.set_xlim(0,1)

    ax.set_ylim(0,1.02)

    ax.legend(frameon=False)

    ax.set_title(f"SOZ localization ROC ({args.split})")

    fig.tight_layout()

    save_all(fig, fig_dir / f"fig1_channel_roc_{args.split}")

    plt.close(fig)



    # PR

    prec, rec, _ = precision_recall_curve(y, p)

    ap = average_precision_score(y, p)



    fig, ax = plt.subplots(figsize=(4.6, 4.0))

    ax.plot(rec, prec, linewidth=1.8, label=f"Channel AUPRC={ap:.3f}")

    ax.set_xlabel("Recall")

    ax.set_ylabel("Precision")

    ax.set_xlim(0,1)

    ax.set_ylim(0,1.02)

    ax.legend(frameon=False)

    ax.set_title(f"SOZ localization PR ({args.split})")

    fig.tight_layout()

    save_all(fig, fig_dir / f"fig2_channel_pr_{args.split}")

    plt.close(fig)



    # Confusion matrix

    pred = (p >= th).astype(int)

    cm = confusion_matrix(y, pred, labels=[0,1])



    fig, ax = plt.subplots(figsize=(3.8, 3.4))

    ax.imshow(cm)

    ax.set_xticks([0,1])

    ax.set_yticks([0,1])

    ax.set_xticklabels(["Normal", "SOZ"])

    ax.set_yticklabels(["Normal", "SOZ"])

    ax.set_xlabel("Predicted")

    ax.set_ylabel("True")

    ax.set_title(f"Channel confusion matrix\nthreshold={th:.2f}")

    for i in range(2):

        for j in range(2):

            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=10)

    fig.tight_layout()

    save_all(fig, fig_dir / f"fig3_channel_confusion_{args.split}")

    plt.close(fig)



    # Score distribution

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    ax.violinplot([p[y==0], p[y==1]], positions=[0,1], showmedians=True)

    ax.boxplot([p[y==0], p[y==1]], positions=[0,1], widths=0.2, showfliers=False)

    ax.axhline(th, linestyle="--", linewidth=1.0)

    ax.set_xticks([0,1])

    ax.set_xticklabels(["Normal", "SOZ"])

    ax.set_ylabel("Predicted SOZ probability")

    ax.set_ylim(-0.02, 1.02)

    ax.set_title(f"Channel score distribution ({args.split})")

    fig.tight_layout()

    save_all(fig, fig_dir / f"fig4_channel_score_distribution_{args.split}")

    plt.close(fig)



    # Ranked localization plot for one patient

    if args.patient_name is None:

        patient = ch["patient_name"].iloc[0]

    else:

        patient = args.patient_name



    d = ch[ch["patient_name"].astype(str) == str(patient)].copy()

    d = d.sort_values("prob_soz", ascending=False).head(args.top_n)



    if len(d) > 0:

        labels = d["channel_name"].astype(str).values

        vals = d["prob_soz"].values

        true = d["label"].values.astype(int)



        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        x = np.arange(len(d))

        ax.bar(x, vals)

        ax.scatter(x[true==1], vals[true==1] + 0.03, marker="*", s=55)

        ax.axhline(th, linestyle="--", linewidth=1.0)

        ax.set_xticks(x)

        ax.set_xticklabels(labels, rotation=60, ha="right")

        ax.set_ylabel("Mean channel SOZ probability")

        ax.set_ylim(0, 1.08)

        ax.set_title(f"Ranked SOZ candidate channels: {patient}")

        fig.tight_layout()

        save_all(fig, fig_dir / f"fig5_ranked_channels_{args.split}_{patient}")

        plt.close(fig)



    # Top-k hit rate

    if topk_path.exists():

        topk = pd.read_csv(topk_path)

        fig, ax = plt.subplots(figsize=(4.2, 3.4))

        ax.plot(topk["top_k"], topk["hit_rate"], marker="o", linewidth=1.5)

        ax.set_xlabel("Top-k channels")

        ax.set_ylabel("SOZ hit rate")

        ax.set_ylim(0, 1.05)

        ax.set_title(f"Top-k SOZ localization hit rate ({args.split})")

        fig.tight_layout()

        save_all(fig, fig_dir / f"fig6_topk_hit_rate_{args.split}")

        plt.close(fig)



    print("Figures saved in:", fig_dir)





if __name__ == "__main__":

    main()


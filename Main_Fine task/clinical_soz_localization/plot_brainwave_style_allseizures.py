
import argparse, os, math

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import gridspec

from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch



def find_prob_col(df):

    for c in ["prob_soz", "probability", "prob", "y_prob", "score"]:

        if c in df.columns:

            return c

    raise RuntimeError("No probability column found. Columns: " + str(df.columns.tolist()))



def clean_ch(x):

    x = str(x).strip().upper()

    x = x.replace("EEG ", "").replace("POL ", "").replace("-REF", "").replace("REF", "")

    x = x.replace(" ", "")

    return x



def natural_key(ch):

    import re

    m = re.match(r"^([A-Z']+)(\d+)$", str(ch))

    if m:

        return (m.group(1), int(m.group(2)))

    return (str(ch), 9999)



def load_predictions(run_dir, split):

    run_dir = Path(run_dir)

    files = []

    if split == "all":

        for s in ["train", "val", "test"]:

            p = run_dir / f"{s}_segment_predictions.csv"

            if p.exists():

                files.append((s, p))

    else:

        p = run_dir / f"{split}_segment_predictions.csv"

        if not p.exists():

            raise FileNotFoundError(p)

        files.append((split, p))

    if not files:

        raise RuntimeError("No prediction CSV files found. Run evaluate_soz.py for train/val first.")

    dfs = []

    for s, p in files:

        d = pd.read_csv(p)

        d["prediction_split_file"] = s

        dfs.append(d)

    return pd.concat(dfs, ignore_index=True)



def load_manifest(run_dir, manifest_csv, split):

    run_manifest = Path(run_dir) / "manifest_with_internal_split.csv"

    if run_manifest.exists():

        man = pd.read_csv(run_manifest)

    else:

        man = pd.read_csv(manifest_csv)

        if "split_internal" not in man.columns and "split" in man.columns:

            man["split_internal"] = man["split"]

    if "clean_channel_name" not in man.columns:

        man["clean_channel_name"] = man["channel_name"].map(clean_ch)

    if split != "all":

        if "split_internal" in man.columns:

            man = man[man["split_internal"].astype(str).str.lower() == split.lower()].copy()

        elif "split" in man.columns:

            man = man[man["split"].astype(str).str.lower() == split.lower()].copy()

    else:

        if "split_internal" in man.columns:

            man = man[man["split_internal"].astype(str).str.lower().isin(["train", "val", "test"])].copy()

    return man.reset_index(drop=True)



def attach_metadata(pred, man, split):

    pred = pred.reset_index(drop=True).copy()

    if split == "all" and "prediction_split_file" in pred.columns and "split_internal" in man.columns:

        parts = []

        for s, d in pred.groupby("prediction_split_file", sort=False):

            m = man[man["split_internal"].astype(str).str.lower() == str(s).lower()].reset_index(drop=True)

            d = d.reset_index(drop=True)

            if len(d) != len(m):

                print(f"WARNING row mismatch for {s}: pred={len(d)}, manifest={len(m)}. Using min length.")

            n = min(len(d), len(m))

            d = d.iloc[:n].copy()

            m = m.iloc[:n].copy()

            for c in ["patient_id","patient_name","seizure_id","raw_file","channel_name","clean_channel_name","target_type","target_rel_start_sec","target_rel_end_sec","label","split_internal"]:

                if c not in d.columns and c in m.columns:

                    d[c] = m[c].values

            parts.append(d)

        pred = pd.concat(parts, ignore_index=True)

    else:

        n = min(len(pred), len(man))

        pred = pred.iloc[:n].copy()

        man = man.iloc[:n].copy()

        for c in ["patient_id","patient_name","seizure_id","raw_file","channel_name","clean_channel_name","target_type","target_rel_start_sec","target_rel_end_sec","label","split_internal"]:

            if c not in pred.columns and c in man.columns:

                pred[c] = man[c].values

    if "clean_channel_name" not in pred.columns:

        pred["clean_channel_name"] = pred["channel_name"].map(clean_ch)

    return pred



def grid_from_values(channels, values, n_cols=7):

    n_rows = int(math.ceil(len(channels) / n_cols))

    mat = np.full((n_rows, n_cols), np.nan)

    labels = [["" for _ in range(n_cols)] for __ in range(n_rows)]

    for i, ch in enumerate(channels):

        r, c = divmod(i, n_cols)

        mat[r, c] = values.get(ch, np.nan)

        labels[r][c] = ch

    return mat, labels



def draw_grid(ax, mat, labels, title, vmin, vmax, cmap, cbar_label, show_int=False, red_set=None):

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_title(title, fontweight="bold", fontsize=12)

    ax.set_xticks([])

    ax.set_yticks([])

    red_set = red_set or set()

    for r in range(mat.shape[0]):

        for c in range(mat.shape[1]):

            ch = labels[r][c]

            if not ch:

                continue

            val = mat[r, c]

            txt = f"{ch}\n{int(val)}" if show_int else f"{ch}\n{val:.2f}"

            color = "white" if val < (vmax * 0.55) else "black"

            ax.text(c, r, txt, ha="center", va="center", fontsize=6.5, color=color)

            if ch in red_set:

                ax.add_patch(Rectangle((c-0.5, r-0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.0))

    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01)

    cb.set_label(cbar_label)

    return im



def draw_brain_panel(ax, top_channels):

    ax.axis("off")

    ax.set_title("Candidate SOZ /\nepileptic discharge area", fontweight="bold", fontsize=12)

    brain = Ellipse((0.52, 0.56), 0.78, 0.48, facecolor="lightgray", edgecolor="gray", alpha=0.35)

    cereb = Ellipse((0.72, 0.38), 0.26, 0.18, facecolor="lightgray", edgecolor="gray", alpha=0.25)

    ax.add_patch(brain); ax.add_patch(cereb)

    rng = np.random.default_rng(7)

    xs = 0.47 + 0.08 * rng.normal(size=min(16, len(top_channels)))

    ys = 0.56 + 0.07 * rng.normal(size=min(16, len(top_channels)))

    colors = ["red"]*3 + ["orange"]*5 + ["gold"]*4 + ["green"]*4

    ax.scatter(xs, ys, s=55, c=colors[:len(xs)], edgecolors="k", linewidths=0.25, zorder=5)

    ax.annotate("epileptic discharge\ncandidate zone", xy=(0.50,0.56), xytext=(0.10,0.85),

                arrowprops=dict(arrowstyle="->", color="red", linestyle="--", lw=1.5),

                fontsize=9, ha="left")

    ax.text(0.05, 0.08, "Red/orange = higher priority\nYellow/green = lower priority", fontsize=8)



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--run_dir", required=True)

    ap.add_argument("--manifest_csv", required=True)

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--split", default="all", choices=["train","val","test","all"])

    ap.add_argument("--threshold", type=float, default=0.015)

    ap.add_argument("--top_n", type=int, default=49)

    ap.add_argument("--top_k", type=int, default=10)

    ap.add_argument("--patient_name", default="Patient")

    args = ap.parse_args()



    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(args.run_dir, args.split)

    man = load_manifest(args.run_dir, args.manifest_csv, args.split)

    pred = attach_metadata(pred, man, args.split)

    prob_col = find_prob_col(pred)



    early = pred[(pred["target_rel_start_sec"] >= 0) & (pred["target_rel_end_sec"] <= 30)].copy()

    if early.empty:

        early = pred[pred["target_type"].astype(str).eq("early_ictal_onset")].copy()

    if early.empty:

        raise RuntimeError("No early ictal rows found.")

    early["clean_channel_name"] = early["clean_channel_name"].map(clean_ch)



    mean_prob = early.groupby("clean_channel_name")[prob_col].mean().rename("mean_prob")

    occ = early.assign(discharge=(early[prob_col] >= args.threshold).astype(int)).groupby("clean_channel_name")["discharge"].mean().rename("occurrence")



    onset_counts = {}

    for sid, d in early.groupby("seizure_id"):

        score = d.groupby("clean_channel_name")[prob_col].mean().sort_values(ascending=False)

        for ch in score.head(args.top_k).index:

            onset_counts[ch] = onset_counts.get(ch, 0) + 1

    freq = pd.Series(onset_counts, name="frequency")



    clinical = early.groupby("clean_channel_name")["label"].max() if "label" in early.columns else pd.Series(dtype=float)



    rank = pd.concat([mean_prob, occ, freq, clinical.rename("clinical_label")], axis=1).fillna(0)

    rank = rank.sort_values(["frequency","occurrence","mean_prob"], ascending=False)

    channels = rank.head(args.top_n).index.tolist()

    red_set = set(rank[rank["clinical_label"] > 0].index.tolist())



    # Panel A: all early bins for all seizures, no S1/S2/S3 labels.

    blocks = []

    for sid, d in early.groupby("seizure_id", sort=False):

        piv = d.pivot_table(index="clean_channel_name", columns="target_rel_start_sec", values=prob_col, aggfunc="mean")

        piv = piv.reindex(index=channels).reindex(sorted(piv.columns), axis=1).fillna(0)

        blocks.append(piv.values)

    matA = np.concatenate(blocks, axis=1)



    occ_values = {ch: float(rank.loc[ch, "occurrence"]) for ch in channels}

    freq_values = {ch: float(rank.loc[ch, "frequency"]) for ch in channels}

    matB, labB = grid_from_values(channels, occ_values, n_cols=7)

    matC, labC = grid_from_values(channels, freq_values, n_cols=7)



    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"ps.fonttype":42})

    fig = plt.figure(figsize=(18, 9), constrained_layout=True)

    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[2.4, 1.6, 1.25], height_ratios=[1,1])



    axA = fig.add_subplot(gs[:,0])

    imA = axA.imshow(matA, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")

    axA.set_title("Channel-level predicted SOZ probability", fontweight="bold", fontsize=13)

    axA.set_ylabel("Channels", fontsize=11)

    axA.set_xlabel("Multiple epileptic seizures", fontsize=11)

    axA.set_yticks(range(len(channels))); axA.set_yticklabels(channels, fontsize=6)

    axA.set_xticks([])

    cbA = plt.colorbar(imA, ax=axA, fraction=0.028, pad=0.01); cbA.set_label("Probability")

    axA.text(-0.10, 1.02, "a", transform=axA.transAxes, fontsize=16, fontweight="bold")



    axB = fig.add_subplot(gs[0,1])

    draw_grid(axB, matB, labB, "Epileptic discharge occurrence", 0, 1, "viridis", "Occurrence rate", False, red_set)

    axB.text(-0.10, 1.04, "b", transform=axB.transAxes, fontsize=16, fontweight="bold")



    axC = fig.add_subplot(gs[1,1])

    vmaxF = max(1, int(np.nanmax(matC)))

    draw_grid(axC, matC, labC, "Onset channel frequency", 0, vmaxF, "viridis", "Times as onset site", True, red_set)

    axC.text(-0.10, 1.04, "c", transform=axC.transAxes, fontsize=16, fontweight="bold")



    axD = fig.add_subplot(gs[:,2])

    draw_brain_panel(axD, channels[:15])

    axD.text(-0.07, 1.02, "d", transform=axD.transAxes, fontsize=16, fontweight="bold")



    base = out_dir / "brainwave_style_allseizures_nature"

    fig.savefig(str(base)+".png", dpi=600, bbox_inches="tight")

    fig.savefig(str(base)+".pdf", dpi=600, bbox_inches="tight")

    rank.reset_index().rename(columns={"index":"clean_channel_name"}).to_csv(out_dir / "brainwave_style_allseizures_channel_ranking.csv", index=False)

    plt.close(fig)

    print("Saved:", str(base)+".png")

    print("Saved:", str(base)+".pdf")

    print("Saved ranking:", out_dir / "brainwave_style_allseizures_channel_ranking.csv")



if __name__ == "__main__":

    main()


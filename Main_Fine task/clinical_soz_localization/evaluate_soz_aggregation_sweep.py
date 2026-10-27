
import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from sklearn.metrics import (

    accuracy_score, balanced_accuracy_score, precision_score, recall_score,

    f1_score, fbeta_score, cohen_kappa_score, matthews_corrcoef,

    roc_auc_score, average_precision_score, confusion_matrix

)



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



    raise ValueError("Unknown method: {}".format(method))



def make_channel_table(seg_df, method):

    rows = []



    # Final clinical unit: one score per patient-channel

    for (patient, channel), g in seg_df.groupby(["patient_name", "channel_name"]):

        score = aggregate_values(g["prob_soz"].values, method)



        rows.append({

            "patient_name": patient,

            "channel_name": channel,

            "prob_soz": score,

            "label": int(g["label"].max()),

            "soz": int(g["soz"].max()) if "soz" in g.columns else int(g["label"].max()),

            "resection": int(g["resection"].max()) if "resection" in g.columns else 0,

            "outcome": str(g["outcome"].iloc[0]) if "outcome" in g.columns else "",

            "anatomical": str(g["anatomical"].iloc[0]) if "anatomical" in g.columns else "",

            "n_segments": int(len(g)),

            "n_recordings": int(g["edf_name"].nunique()) if "edf_name" in g.columns else 1,

            "aggregation": method,

        })



    ch = pd.DataFrame(rows)

    ch["rank_within_patient"] = ch.groupby("patient_name")["prob_soz"].rank(

        ascending=False, method="first"

    ).astype(int)

    return ch.sort_values(["patient_name", "rank_within_patient"])



def get_metrics(y, p, th):

    pred = (p >= th).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()



    out = {

        "threshold": float(th),

        "accuracy": accuracy_score(y, pred),

        "balanced_accuracy": balanced_accuracy_score(y, pred),

        "precision": precision_score(y, pred, zero_division=0),

        "sensitivity": recall_score(y, pred, zero_division=0),

        "specificity": tn / max(1, tn + fp),

        "f1_score": f1_score(y, pred, zero_division=0),

        "f2_score": fbeta_score(y, pred, beta=2, zero_division=0),

        "kappa": cohen_kappa_score(y, pred),

        "mcc": matthews_corrcoef(y, pred),

        "tn": int(tn),

        "fp": int(fp),

        "fn": int(fn),

        "tp": int(tp),

    }



    try:

        out["auroc"] = roc_auc_score(y, p)

    except Exception:

        out["auroc"] = np.nan



    try:

        out["auprc"] = average_precision_score(y, p)

    except Exception:

        out["auprc"] = np.nan



    return out



def best_threshold(y, p):

    best = None

    for th in np.linspace(0, 1, 1001):

        m = get_metrics(y, p, th)

        if best is None or m["balanced_accuracy"] > best["balanced_accuracy"]:

            best = m

    return best["threshold"]



def topk_hit_rate(ch, ks=(1, 3, 5, 10)):

    rows = []

    for k in ks:

        hits = []

        for patient, g in ch.groupby("patient_name"):

            g = g.sort_values("prob_soz", ascending=False)

            if g["label"].sum() == 0:

                continue

            hits.append(int(g.head(k)["label"].max() == 1))



        rows.append({

            "top_k": k,

            "hit_rate": float(np.mean(hits)) if hits else np.nan,

            "n_patients": len(hits),

        })

    return pd.DataFrame(rows)



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", required=True)

    parser.add_argument("--out_name", default="aggregation_sweep_patient_channel")

    args = parser.parse_args()



    run_dir = Path(args.run_dir)

    out_dir = run_dir / args.out_name

    out_dir.mkdir(parents=True, exist_ok=True)



    val_seg = pd.read_csv(run_dir / "val_segment_predictions.csv")

    test_seg = pd.read_csv(run_dir / "test_segment_predictions.csv")



    methods = ["mean", "median", "max", "p90", "p95", "top10_mean", "top20_mean", "top30_mean"]



    all_rows = []



    for method in methods:

        val_ch = make_channel_table(val_seg, method)

        test_ch = make_channel_table(test_seg, method)



        yv = val_ch["label"].values.astype(int)

        pv = val_ch["prob_soz"].values.astype(float)



        th = best_threshold(yv, pv)



        val_m = get_metrics(yv, pv, th)

        test_m = get_metrics(

            test_ch["label"].values.astype(int),

            test_ch["prob_soz"].values.astype(float),

            th,

        )



        tk = topk_hit_rate(test_ch)



        row = {"aggregation": method}

        row.update({"val_" + k: v for k, v in val_m.items()})

        row.update({"test_" + k: v for k, v in test_m.items()})



        for _, r in tk.iterrows():

            row["test_top{}_hit_rate".format(int(r["top_k"]))] = r["hit_rate"]

            row["test_top{}_n_patients".format(int(r["top_k"]))] = r["n_patients"]



        all_rows.append(row)



        val_ch.to_csv(out_dir / "val_channel_scores_{}.csv".format(method), index=False)

        test_ch.to_csv(out_dir / "test_channel_scores_{}.csv".format(method), index=False)

        tk.to_csv(out_dir / "test_topk_{}.csv".format(method), index=False)



    summary = pd.DataFrame(all_rows)

    summary = summary.sort_values(

        ["test_auprc", "test_auroc", "test_balanced_accuracy"],

        ascending=False

    )

    summary.to_csv(out_dir / "aggregation_sweep_summary.csv", index=False)



    print("\nSaved:", out_dir / "aggregation_sweep_summary.csv")

    print("\nMain comparison:")

    print(summary[[

        "aggregation",

        "test_auroc",

        "test_auprc",

        "test_balanced_accuracy",

        "test_precision",

        "test_sensitivity",

        "test_specificity",

        "test_top1_hit_rate",

        "test_top3_hit_rate",

        "test_top5_hit_rate",

        "test_top10_hit_rate"

    ]])



if __name__ == "__main__":

    main()


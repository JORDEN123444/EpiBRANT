
import argparse

from pathlib import Path

import numpy as np

import pandas as pd

import yaml

import torch

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, fbeta_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix



from downstream.clinical_soz_localization.data_soz import OmniSOZDataset

from downstream.clinical_soz_localization.model_soz import build_soz_model, logits_from_output





def forward_model(model, batch, device):

    x = batch["x"].to(device).float()

    fs = batch["fs"].to(device).float()

    mask = batch["channel_mask"].to(device).float()

    try:

        out = model(x, fs, mask)

    except TypeError:

        out = model(x)

    return logits_from_output(out)





def prob_from_logits(logits):

    if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):

        return torch.sigmoid(logits.view(-1))

    elif logits.dim() == 2 and logits.size(1) >= 2:

        return torch.softmax(logits, dim=1)[:, 1]

    else:

        raise RuntimeError("Unsupported logits shape: {}".format(tuple(logits.shape)))





def get_metrics(y, p, th):

    pred = (p >= th).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    out = {

        "threshold": th,

        "accuracy": accuracy_score(y, pred),

        "balanced_accuracy": balanced_accuracy_score(y, pred),

        "precision": precision_score(y, pred, zero_division=0),

        "sensitivity": recall_score(y, pred, zero_division=0),

        "specificity": tn / max(1, tn + fp),

        "f1_score": f1_score(y, pred, zero_division=0),

        "f2_score": fbeta_score(y, pred, beta=2, zero_division=0),

        "kappa": cohen_kappa_score(y, pred),

        "mcc": matthews_corrcoef(y, pred),

        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),

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

    best_th, best_ba = 0.5, -1

    for th in np.linspace(0, 1, 201):

        ba = get_metrics(y, p, th)["balanced_accuracy"]

        if ba > best_ba:

            best_ba = ba

            best_th = th

    return float(best_th)





@torch.no_grad()

def predict_split(model, ds, device, batch_size):

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()

    rows = []



    for batch in loader:

        logits = forward_model(model, batch, device)

        prob = prob_from_logits(logits).detach().cpu().numpy()

        labels = batch["label"].detach().cpu().numpy()



        for i in range(len(labels)):

            rows.append({

                "patient_name": batch["patient_name"][i],

                "edf_name": batch["edf_name"][i],

                "channel_name": batch["channel_name"][i],

                "anatomical": batch["anatomical"][i],

                "segment_start_sec": float(batch["segment_start_sec"][i]),

                "segment_end_sec": float(batch["segment_end_sec"][i]),

                "label": int(labels[i]),

                "soz": int(batch["soz"][i]),

                "resection": int(batch["resection"][i]),

                "outcome": str(batch["outcome"][i]),

                "prob_soz": float(prob[i]),

            })



    return pd.DataFrame(rows)





def aggregate_channel(seg_df):

    gcols = ["patient_name", "edf_name", "channel_name"]

    ch = seg_df.groupby(gcols).agg(

        prob_soz=("prob_soz", "mean"),

        prob_std=("prob_soz", "std"),

        n_segments=("prob_soz", "size"),

        label=("label", "max"),

        soz=("soz", "max"),

        resection=("resection", "max"),

        outcome=("outcome", "first"),

        anatomical=("anatomical", "first"),

    ).reset_index()



    ch["rank_within_patient"] = ch.groupby("patient_name")["prob_soz"].rank(ascending=False, method="first").astype(int)

    ch = ch.sort_values(["patient_name", "rank_within_patient"])

    return ch





def topk_hit_rate(ch, ks=[1, 3, 5]):

    rows = []

    for k in ks:

        hits = []

        for patient, g in ch.groupby("patient_name"):

            g = g.sort_values("prob_soz", ascending=False)

            if g["label"].sum() == 0:

                continue

            hits.append(int(g.head(k)["label"].max() == 1))

        rows.append({"top_k": k, "hit_rate": float(np.mean(hits)) if hits else np.nan, "n_patients": len(hits)})

    return pd.DataFrame(rows)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", required=True)

    parser.add_argument("--split", default="test")

    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()



    run_dir = Path(args.run_dir)



    with open(run_dir / "used_config.yaml", "r") as f:

        cfg = yaml.safe_load(f)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    model = build_soz_model(cfg).to(device)

    ckpt = torch.load(run_dir / "best_model.pt", map_location=device)

    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model.load_state_dict(sd, strict=False)



    val_ds = OmniSOZDataset(

        cfg["data"]["manifest_csv"],

        split="val",

        split_col="split_internal",

        context_sec=cfg["data"]["context_sec"],

        patch_sec=cfg["data"].get("patch_sec", 1),

        target_fs=cfg["data"]["target_fs"],

        max_channels=cfg["data"]["max_channels"],

    )



    val_seg = predict_split(model, val_ds, device, args.batch_size)

    th = best_threshold(val_seg["label"].values.astype(int), val_seg["prob_soz"].values.astype(float))



    ds = OmniSOZDataset(

        cfg["data"]["manifest_csv"],

        split=args.split,

        split_col="split_internal",

        context_sec=cfg["data"]["context_sec"],

        patch_sec=cfg["data"].get("patch_sec", 1),

        target_fs=cfg["data"]["target_fs"],

        max_channels=cfg["data"]["max_channels"],

    )



    seg = predict_split(model, ds, device, args.batch_size)

    ch = aggregate_channel(seg)



    seg_m = get_metrics(seg["label"].values.astype(int), seg["prob_soz"].values.astype(float), th)

    ch_m = get_metrics(ch["label"].values.astype(int), ch["prob_soz"].values.astype(float), th)

    topk = topk_hit_rate(ch)



    seg.to_csv(run_dir / "{}_segment_predictions.csv".format(args.split), index=False)

    ch.to_csv(run_dir / "{}_channel_localization_scores.csv".format(args.split), index=False)

    pd.DataFrame([seg_m]).to_csv(run_dir / "{}_segment_metrics.csv".format(args.split), index=False)

    pd.DataFrame([ch_m]).to_csv(run_dir / "{}_channel_metrics.csv".format(args.split), index=False)

    topk.to_csv(run_dir / "{}_topk_hit_rate.csv".format(args.split), index=False)



    print("Validation-selected threshold:", th)

    print("\nSegment metrics:")

    print(pd.DataFrame([seg_m]).T)



    print("\nChannel-level localization metrics:")

    print(pd.DataFrame([ch_m]).T)



    print("\nTop-k SOZ hit rate:")

    print(topk)



    print("\nSaved:")

    print(run_dir / "{}_channel_localization_scores.csv".format(args.split))





if __name__ == "__main__":

    main()


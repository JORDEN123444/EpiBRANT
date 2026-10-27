
import argparse

from pathlib import Path

import random

import numpy as np

import pandas as pd

import yaml

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix



from downstream.clinical_soz_localization.data_soz import OmniSOZDataset

from downstream.clinical_soz_localization.model_soz import build_soz_model, load_checkpoint_shape_safe, logits_from_output





def seed_all(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)





def make_internal_split(manifest_csv, out_csv, seed=42, val_frac=0.15):

    df = pd.read_csv(manifest_csv)

    s = df["split"].astype(str).str.lower()



    df["split_internal"] = s



    if "val" not in set(df["split_internal"]):

        train_mask = df["split_internal"].isin(["train", "training"])

        train_patients = sorted(df.loc[train_mask, "patient_name"].dropna().unique())



        rng = np.random.default_rng(seed)

        rng.shuffle(train_patients)



        n_val = max(1, int(round(len(train_patients) * val_frac)))

        val_patients = set(train_patients[:n_val])



        df.loc[train_mask & df["patient_name"].isin(val_patients), "split_internal"] = "val"

        df.loc[train_mask & ~df["patient_name"].isin(val_patients), "split_internal"] = "train"



    df.to_csv(out_csv, index=False)

    print("Saved internal split manifest:", out_csv)

    print(df["split_internal"].value_counts())

    return out_csv





def forward_model(model, batch, device):

    x = batch["x"].to(device).float()

    fs = batch["fs"].to(device).float()

    mask = batch["channel_mask"].to(device).float()



    try:

        out = model(x, fs, mask)

    except TypeError:

        out = model(x)



    return logits_from_output(out)









def compute_loss(logits, y):

    # Supports both:

    # 1) binary one-logit output: [B] or [B, 1]

    # 2) two-class output: [B, 2]

    if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):

        return torch.nn.functional.binary_cross_entropy_with_logits(

            logits.view(-1), y.float()

        )

    elif logits.dim() == 2 and logits.size(1) >= 2:

        return torch.nn.functional.cross_entropy(logits, y.long())

    else:

        raise RuntimeError("Unsupported logits shape: {}".format(tuple(logits.shape)))





def prob_from_logits(logits):

    # Returns probability of SOZ/pathological class

    if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):

        return torch.sigmoid(logits.view(-1))

    elif logits.dim() == 2 and logits.size(1) >= 2:

        return torch.softmax(logits, dim=1)[:, 1]

    else:

        raise RuntimeError("Unsupported logits shape: {}".format(tuple(logits.shape)))



def eval_split(model, loader, device, threshold=0.5):

    model.eval()

    ys, ps = [], []

    losses = []

    ce = nn.CrossEntropyLoss()



    with torch.no_grad():

        for batch in loader:

            y = batch["label"].to(device)

            logits = forward_model(model, batch, device)

            loss = compute_loss(logits, y)

            prob = prob_from_logits(logits)

            ys.extend(y.detach().cpu().numpy().tolist())

            ps.extend(prob.detach().cpu().numpy().tolist())

            losses.append(float(loss.item()))



    y = np.asarray(ys).astype(int)

    p = np.asarray(ps).astype(float)

    pred = (p >= threshold).astype(int)



    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()



    m = {

        "loss": float(np.mean(losses)) if losses else np.nan,

        "accuracy": accuracy_score(y, pred),

        "balanced_accuracy": balanced_accuracy_score(y, pred),

        "precision": precision_score(y, pred, zero_division=0),

        "sensitivity": recall_score(y, pred, zero_division=0),

        "specificity": tn / max(1, tn + fp),

        "f1_score": f1_score(y, pred, zero_division=0),

        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),

    }



    try:

        m["auroc"] = roc_auc_score(y, p)

    except Exception:

        m["auroc"] = np.nan

    try:

        m["auprc"] = average_precision_score(y, p)

    except Exception:

        m["auprc"] = np.nan



    return m





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    args = parser.parse_args()



    with open(args.config, "r") as f:

        cfg = yaml.safe_load(f)



    seed_all(int(cfg["train"].get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)



    out_dir = Path(cfg["train"]["out_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)



    split_manifest = out_dir / "manifest_with_internal_split.csv"

    make_internal_split(cfg["data"]["manifest_csv"], split_manifest, seed=int(cfg["train"].get("seed", 42)))



    cfg["data"]["manifest_csv"] = str(split_manifest)



    with open(out_dir / "used_config.yaml", "w") as f:

        yaml.safe_dump(cfg, f)



    train_ds = OmniSOZDataset(split_manifest, split="train", split_col="split_internal", context_sec=cfg["data"]["context_sec"], patch_sec=cfg["data"].get("patch_sec", 1), target_fs=cfg["data"]["target_fs"], max_channels=cfg["data"]["max_channels"])

    val_ds = OmniSOZDataset(split_manifest, split="val", split_col="split_internal", context_sec=cfg["data"]["context_sec"], patch_sec=cfg["data"].get("patch_sec", 1), target_fs=cfg["data"]["target_fs"], max_channels=cfg["data"]["max_channels"])



    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"].get("num_workers", 0))

    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"].get("num_workers", 0))



    model = build_soz_model(cfg).to(device)

    model = load_checkpoint_shape_safe(model, cfg["train"]["pretrained_checkpoint"], device=device)



    backbone_params = []

    head_params = []

    for n, p in model.named_parameters():

        if not p.requires_grad:

            continue

        if "classifier" in n.lower() or "head" in n.lower() or n.lower().endswith("fc.weight") or n.lower().endswith("fc.bias"):

            head_params.append(p)

        else:

            backbone_params.append(p)



    optimizer = torch.optim.AdamW([

        {"params": backbone_params, "lr": float(cfg["train"]["backbone_lr"])},

        {"params": head_params, "lr": float(cfg["train"]["head_lr"])},

    ], weight_decay=float(cfg["train"]["weight_decay"]))



    ce = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")



    best_ba = -1

    logs = []



    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):

        model.train()

        train_losses = []



        for batch in train_loader:

            y = batch["label"].to(device)



            optimizer.zero_grad(set_to_none=True)



            with torch.cuda.amp.autocast(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda"):

                logits = forward_model(model, batch, device)

                loss = compute_loss(logits, y)



            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)



            if cfg["train"].get("grad_clip", None) is not None:

                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))



            scaler.step(optimizer)

            scaler.update()

            train_losses.append(float(loss.item()))



        val_m = eval_split(model, val_loader, device, threshold=0.5)



        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses))}

        row.update({"val_" + k: v for k, v in val_m.items()})

        logs.append(row)



        print("Epoch {:03d} | train_loss {:.4f} | val_BA {:.4f} | val_AUROC {:.4f} | val_AUPRC {:.4f}".format(

            epoch, row["train_loss"], val_m["balanced_accuracy"], val_m["auroc"], val_m["auprc"]

        ))



        pd.DataFrame(logs).to_csv(out_dir / "training_log.csv", index=False)



        if val_m["balanced_accuracy"] > best_ba:

            best_ba = val_m["balanced_accuracy"]

            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": val_m}, out_dir / "best_model.pt")

            print("Saved best_model.pt")



        torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "last_model.pt")



    print("Training finished. Output:", out_dir)





if __name__ == "__main__":

    main()


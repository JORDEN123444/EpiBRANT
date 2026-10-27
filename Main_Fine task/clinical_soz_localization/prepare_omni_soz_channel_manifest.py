
import argparse

from pathlib import Path

import pandas as pd

import numpy as np





def as_bool01(x):

    if pd.isna(x):

        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):

        return int(x)

    s = str(x).strip().lower()

    if s in ["1", "true", "yes", "y", "t"]:

        return 1

    if s in ["0", "false", "no", "n", "f"]:

        return 0

    return np.nan





def normalize_type(x):

    if pd.isna(x):

        return ""

    return str(x).strip().upper()





def find_file_by_name(root, name):

    name = str(name)

    hits = list(root.rglob(name))

    if len(hits) > 0:

        return hits[0]

    return None





def channel_tsv_from_edf(edf_path):

    edf_path = Path(edf_path)

    candidates = [

        edf_path.with_name(edf_path.name.replace("_ieeg.edf", "_channels.tsv")),

        edf_path.with_name(edf_path.name.replace(".edf", "_channels.tsv")),

    ]

    for c in candidates:

        if c.exists():

            return c



    hits = list(edf_path.parent.glob("*_channels.tsv"))

    if len(hits) == 1:

        return hits[0]



    return None





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--omni_root", required=True, help="Full Omni-iEEG BIDS root")

    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--modality", default="SEEG", help="SEEG first; later ECOG/all")

    parser.add_argument("--require_soz", action="store_true", default=True)

    parser.add_argument("--require_resection", action="store_true", default=True)

    args = parser.parse_args()



    root = Path(args.omni_root)

    split_csv = root / "derivatives" / "datasplit" / "final_split.csv"



    if not split_csv.exists():

        raise FileNotFoundError(f"Cannot find final_split.csv: {split_csv}")



    split_df = pd.read_csv(split_csv)

    print("Loaded final_split:", split_df.shape)

    print("Columns:", split_df.columns.tolist())



    rows = []

    skipped = []



    for _, r in split_df.iterrows():

        edf_name = r.get("edf_name", None)

        if pd.isna(edf_name):

            skipped.append(("missing_edf_name", None))

            continue



        # Optional run-level filters

        has_soz = as_bool01(r.get("has_soz", np.nan))

        has_resection = as_bool01(r.get("has_resection", np.nan))

        outcome = r.get("outcome", np.nan)



        if args.require_soz and has_soz != 1:

            continue

        if args.require_resection and has_resection != 1:

            continue



        edf_path = find_file_by_name(root, edf_name)

        if edf_path is None:

            skipped.append(("edf_not_found", edf_name))

            continue



        ch_tsv = channel_tsv_from_edf(edf_path)

        if ch_tsv is None or not ch_tsv.exists():

            skipped.append(("channels_tsv_not_found", str(edf_path)))

            continue



        try:

            ch = pd.read_csv(ch_tsv, sep="\t")

        except Exception as e:

            skipped.append((f"channels_read_error:{e}", str(ch_tsv)))

            continue



        needed = ["name", "type", "good", "soz", "resection"]

        for c in needed:

            if c not in ch.columns:

                ch[c] = np.nan



        for _, cr in ch.iterrows():

            ch_type = normalize_type(cr.get("type", ""))

            if args.modality.upper() != "ALL" and ch_type != args.modality.upper():

                continue



            good = as_bool01(cr.get("good", np.nan))

            soz = as_bool01(cr.get("soz", np.nan))

            resection = as_bool01(cr.get("resection", np.nan))



            if good != 1:

                continue



            # Label rule

            label = None

            label_rule = None



            # Positive SOZ/pathological channel

            if soz == 1:

                label = 1

                label_rule = "positive_good_soz"



            # High-confidence negative:

            # non-SOZ, non-resected channel, in seizure-free patient

            elif soz == 0 and resection == 0 and str(outcome).strip() in ["1", "1.0", "seizure_free", "Seizure-free"]:

                label = 0

                label_rule = "negative_preserved_nonsoz_seizurefree"



            # Ambiguous channel: exclude

            else:

                continue



            duration = cr.get("edf_length", r.get("length", np.nan))

            fs = cr.get("sampling_frequency", r.get("frequency", np.nan))



            rows.append({

                "patient_name": r.get("patient_name", ""),

                "dataset": r.get("dataset", ""),

                "split": r.get("split", ""),

                "edf_name": edf_name,

                "edf_path": str(edf_path),

                "channels_tsv": str(ch_tsv),

                "outcome": outcome,

                "has_soz": has_soz,

                "has_resection": has_resection,

                "channel_name": cr.get("name", ""),

                "channel_type": ch_type,

                "good": good,

                "soz": soz,

                "resection": resection,

                "anatomical": cr.get("anatomical", ""),

                "sampling_frequency": fs,

                "duration_sec": duration,

                "label": label,

                "label_name": "SOZ_pathological" if label == 1 else "high_confidence_normal",

                "label_rule": label_rule,

            })



    out = pd.DataFrame(rows)

    out_path = Path(args.out_csv)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_path, index=False)



    print("\nSaved:", out_path)

    print("Rows:", len(out))



    if len(out) > 0:

        print("\nLabel counts:")

        print(out["label"].value_counts(dropna=False))

        print("\nSplit counts:")

        print(out["split"].value_counts(dropna=False))

        print("\nDataset counts:")

        print(out["dataset"].value_counts(dropna=False).head(20))

        print("\nPatients:", out["patient_name"].nunique())



    if len(skipped) > 0:

        skip_df = pd.DataFrame(skipped, columns=["reason", "item"])

        skip_path = out_path.with_name(out_path.stem + "_skipped.csv")

        skip_df.to_csv(skip_path, index=False)

        print("Skipped log:", skip_path)





if __name__ == "__main__":

    main()


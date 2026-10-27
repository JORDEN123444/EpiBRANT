
import argparse

from pathlib import Path

import pandas as pd

import numpy as np



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--channel_manifest", required=True)

    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--context_sec", type=float, default=60.0)

    parser.add_argument("--stride_sec", type=float, default=60.0)

    parser.add_argument("--max_segments_per_channel", type=int, default=20)

    args = parser.parse_args()



    df = pd.read_csv(args.channel_manifest)

    rows = []



    for _, r in df.iterrows():

        dur = r.get("duration_sec", np.nan)

        try:

            dur = float(dur)

        except Exception:

            continue



        if np.isnan(dur) or dur < args.context_sec:

            continue



        starts = np.arange(0, dur - args.context_sec + 1e-6, args.stride_sec)



        if args.max_segments_per_channel > 0 and len(starts) > args.max_segments_per_channel:

            idx = np.linspace(0, len(starts) - 1, args.max_segments_per_channel).round().astype(int)

            starts = starts[idx]



        for s in starts:

            out = r.to_dict()

            out["segment_start_sec"] = float(s)

            out["segment_end_sec"] = float(s + args.context_sec)

            out["context_sec"] = args.context_sec

            out["stride_sec"] = args.stride_sec

            rows.append(out)



    out_df = pd.DataFrame(rows)

    out_path = Path(args.out_csv)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_path, index=False)



    print("Saved:", out_path)

    print("Rows:", len(out_df))



    if len(out_df) > 0:

        print("\nLabel counts:")

        print(out_df["label"].value_counts(dropna=False))

        print("\nSplit counts:")

        print(out_df["split"].value_counts(dropna=False))

        print("\nPatients:", out_df["patient_name"].nunique())



if __name__ == "__main__":

    main()



import argparse

import re

from pathlib import Path



import numpy as np

import pandas as pd





METRIC_ALIASES = {

    "accuracy": [

        "accuracy",

        "acc",

    ],

    "balanced_accuracy": [

        "balanced_accuracy",

        "balanced_acc",

        "ba",

    ],

    "precision_sleep": [

        "precision_sleep",

        "precision",

    ],

    "sensitivity_sleep": [

        "sensitivity_sleep",

        "sensitivity",

        "recall",

        "recall_sleep",

    ],

    "specificity_awake": [

        "specificity_awake",

        "specificity",

    ],

    "f1_score": [

        "f1_score",

        "f1",

        "f1_sleep",

    ],

    "mcc": [

        "mcc",

        "matthews_corrcoef",

        "matthews_correlation_coefficient",

    ],

    "kappa": [

        "kappa",

        "cohen_kappa",

    ],

    "auroc": [

        "auroc",

        "roc_auc",

        "auc_roc",

        "auc",

    ],

    "auprc": [

        "auprc",

        "average_precision",

        "pr_auc",

        "auc_pr",

    ],

}





def safe_name(value):

    value = str(value).strip()



    value = re.sub(

        r"[^A-Za-z0-9_.-]+",

        "_",

        value,

    )



    return value.strip("_")





def normalize_column_name(column):

    return re.sub(

        r"[^a-z0-9]+",

        "_",

        str(column).strip().lower(),

    ).strip("_")





def infer_integer_from_path(path, patterns):

    path_string = str(path)



    for pattern in patterns:

        match = re.search(

            pattern,

            path_string,

            flags=re.IGNORECASE,

        )



        if match:

            return int(match.group(1))



    return np.nan





def canonicalize_dataframe_columns(dataframe):

    renamed = {}



    for column in dataframe.columns:

        normalized = normalize_column_name(

            column

        )



        if normalized not in renamed.values():

            renamed[column] = normalized



    return dataframe.rename(

        columns=renamed

    )





def add_canonical_metrics(dataframe):

    for canonical_metric, aliases in METRIC_ALIASES.items():

        if canonical_metric not in dataframe.columns:

            dataframe[canonical_metric] = np.nan



        for alias in aliases:

            normalized_alias = normalize_column_name(

                alias

            )



            if normalized_alias in dataframe.columns:

                values = pd.to_numeric(

                    dataframe[normalized_alias],

                    errors="coerce",

                )



                existing = pd.to_numeric(

                    dataframe[canonical_metric],

                    errors="coerce",

                )



                dataframe[canonical_metric] = (

                    existing.combine_first(values)

                )



    return dataframe





def load_metrics_file(

    metrics_path,

    dataset_name,

    method_name,

    root_dir,

    episode_metadata=None,

):

    metrics = pd.read_csv(

        metrics_path

    )



    metrics = canonicalize_dataframe_columns(

        metrics

    )



    if metrics.empty:

        raise RuntimeError(

            "Metrics file is empty: {}".format(

                metrics_path

            )

        )



    rows = []



    for metric_row_index, metric_row in metrics.iterrows():

        record = {

            "dataset": str(dataset_name),

            "method": str(method_name),

            "root_dir": str(root_dir),

            "metrics_path": str(metrics_path),

            "result_dir": str(

                metrics_path.parent

            ),

            "metrics_row_index": int(

                metric_row_index

            ),

        }



        if episode_metadata is not None:

            for key, value in episode_metadata.items():

                normalized_key = normalize_column_name(

                    key

                )



                record[normalized_key] = value



        for key, value in metric_row.to_dict().items():

            record[key] = value



        rows.append(record)



    return rows





def collect_using_episode_table(

    episode_table_path,

    dataset_name,

    method_name,

    root_dir,

):

    episodes = pd.read_csv(

        episode_table_path

    )



    episodes = canonicalize_dataframe_columns(

        episodes

    )



    if "result_dir" not in episodes.columns:

        raise RuntimeError(

            "Episode table has no result_dir column: {}"

            .format(episode_table_path)

        )



    records = []



    for _, episode in episodes.iterrows():

        result_dir = Path(

            str(episode["result_dir"])

        )



        metrics_path = (

            result_dir / "metrics.csv"

        )



        if not metrics_path.exists():

            print(

                "WARNING: missing metrics file:",

                metrics_path,

            )



            continue



        episode_metadata = episode.to_dict()



        records.extend(

            load_metrics_file(

                metrics_path=metrics_path,

                dataset_name=dataset_name,

                method_name=method_name,

                root_dir=root_dir,

                episode_metadata=episode_metadata,

            )

        )



    return records





def collect_recursively(

    root_dir,

    dataset_name,

    method_name,

):

    metrics_paths = sorted(

        root_dir.rglob("metrics.csv")

    )



    records = []



    for metrics_path in metrics_paths:

        records.extend(

            load_metrics_file(

                metrics_path=metrics_path,

                dataset_name=dataset_name,

                method_name=method_name,

                root_dir=root_dir,

                episode_metadata=None,

            )

        )



    return records





def infer_missing_identifiers(dataframe):

    if "shot" not in dataframe.columns:

        dataframe["shot"] = np.nan



    if "seed" not in dataframe.columns:

        dataframe["seed"] = np.nan



    for index, row in dataframe.iterrows():

        path = row["metrics_path"]



        if pd.isna(row["shot"]):

            dataframe.loc[index, "shot"] = (

                infer_integer_from_path(

                    path,

                    [

                        r"(?:^|[/_-])k(\d+)(?:[/_-]|$)",

                        r"(\d+)[_-]?shot",

                        r"shot[_-]?(\d+)",

                    ],

                )

            )



        if pd.isna(row["seed"]):

            dataframe.loc[index, "seed"] = (

                infer_integer_from_path(

                    path,

                    [

                        r"seed[_-]?(\d+)",

                        r"(?:^|[/_-])s(\d+)(?:[/_-]|$)",

                    ],

                )

            )



    dataframe["shot"] = pd.to_numeric(

        dataframe["shot"],

        errors="coerce",

    ).astype("Int64")



    dataframe["seed"] = pd.to_numeric(

        dataframe["seed"],

        errors="coerce",

    ).astype("Int64")



    return dataframe





def create_long_format(wide_dataframe):

    id_columns = [

        "dataset",

        "method",

        "shot",

        "seed",

        "patient_id",

        "query_patient_ids",

        "support_patient_ids",

        "original_split",

        "split",

        "source_dataset",

        "metrics_path",

        "result_dir",

    ]



    existing_id_columns = [

        column

        for column in id_columns

        if column in wide_dataframe.columns

    ]



    metric_columns = [

        metric

        for metric in METRIC_ALIASES

        if metric in wide_dataframe.columns

    ]



    long_dataframe = wide_dataframe.melt(

        id_vars=existing_id_columns,

        value_vars=metric_columns,

        var_name="metric",

        value_name="value",

    )



    long_dataframe["value"] = pd.to_numeric(

        long_dataframe["value"],

        errors="coerce",

    )



    long_dataframe = long_dataframe.dropna(

        subset=["value"]

    ).reset_index(drop=True)



    return long_dataframe





def create_summary(long_dataframe):

    grouping_columns = [

        "dataset",

        "method",

        "shot",

        "metric",

    ]



    summary = (

        long_dataframe

        .groupby(

            grouping_columns,

            dropna=False,

        )["value"]

        .agg(

            number_of_results="count",

            mean="mean",

            standard_deviation="std",

            median="median",

            minimum="min",

            maximum="max",

        )

        .reset_index()

    )



    return summary





def create_paired_comparison(long_dataframe):

    candidate_pair_columns = [

        "dataset",

        "method",

        "seed",

    ]



    if (

        "patient_id" in long_dataframe.columns

        and long_dataframe["patient_id"].notna().any()

    ):

        candidate_pair_columns.append(

            "patient_id"

        )



    if (

        "query_patient_ids" in long_dataframe.columns

        and long_dataframe[

            "query_patient_ids"

        ].notna().any()

    ):

        candidate_pair_columns.append(

            "query_patient_ids"

        )



    pair_columns = [

        column

        for column in candidate_pair_columns

        if column in long_dataframe.columns

    ]



    pair_columns.append("metric")



    paired_source = long_dataframe[

        long_dataframe["shot"].isin(

            [3, 5]

        )

    ].copy()



    if paired_source.empty:

        return pd.DataFrame()



    pivot = paired_source.pivot_table(

        index=pair_columns,

        columns="shot",

        values="value",

        aggfunc="mean",

    ).reset_index()



    if 3 not in pivot.columns or 5 not in pivot.columns:

        return pd.DataFrame()



    pivot = pivot.rename(

        columns={

            3: "three_shot",

            5: "five_shot",

        }

    )



    pivot["five_minus_three"] = (

        pivot["five_shot"]

        - pivot["three_shot"]

    )



    return pivot





def save_per_experiment_files(

    wide_dataframe,

    output_dir,

):

    per_experiment_dir = (

        output_dir / "per_experiment"

    )



    per_experiment_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    for (

        dataset_name,

        method_name,

    ), group in wide_dataframe.groupby(

        ["dataset", "method"],

        dropna=False,

    ):

        filename = "{}__{}.csv".format(

            safe_name(dataset_name),

            safe_name(method_name),

        )



        output_path = (

            per_experiment_dir / filename

        )



        group.sort_values(

            [

                column

                for column in [

                    "shot",

                    "seed",

                    "patient_id",

                ]

                if column in group.columns

            ]

        ).to_csv(

            output_path,

            index=False,

        )



        print(

            "Saved per-experiment CSV:",

            output_path,

        )





def main():

    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--registry",

        required=True,

    )



    parser.add_argument(

        "--out_dir",

        required=True,

    )



    args = parser.parse_args()



    registry_path = Path(

        args.registry

    )



    output_dir = Path(

        args.out_dir

    )



    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    registry = pd.read_csv(

        registry_path

    )



    registry = canonicalize_dataframe_columns(

        registry

    )



    required_registry_columns = [

        "dataset",

        "method",

        "root_dir",

    ]



    missing_registry_columns = [

        column

        for column in required_registry_columns

        if column not in registry.columns

    ]



    if missing_registry_columns:

        raise RuntimeError(

            "Registry is missing columns: {}"

            .format(missing_registry_columns)

        )



    all_records = []



    for _, experiment in registry.iterrows():

        dataset_name = str(

            experiment["dataset"]

        )



        method_name = str(

            experiment["method"]

        )



        root_dir = Path(

            str(experiment["root_dir"])

        )



        if not root_dir.exists():

            print(

                "WARNING: experiment root does not exist:",

                root_dir,

            )



            continue



        episode_table_path = None



        if (

            "episode_table" in registry.columns

            and not pd.isna(

                experiment["episode_table"]

            )

            and str(

                experiment["episode_table"]

            ).strip()

        ):

            episode_table_path = Path(

                str(

                    experiment["episode_table"]

                )

            )



        print("\n" + "=" * 100)

        print("Dataset:", dataset_name)

        print("Method:", method_name)

        print("Root:", root_dir)

        print("=" * 100)



        if (

            episode_table_path is not None

            and episode_table_path.exists()

        ):

            print(

                "Collecting from episode table:",

                episode_table_path,

            )



            records = collect_using_episode_table(

                episode_table_path=episode_table_path,

                dataset_name=dataset_name,

                method_name=method_name,

                root_dir=root_dir,

            )



        else:

            if episode_table_path is not None:

                print(

                    "WARNING: episode table does not exist:",

                    episode_table_path,

                )



            print(

                "Falling back to recursive metrics.csv search."

            )



            records = collect_recursively(

                root_dir=root_dir,

                dataset_name=dataset_name,

                method_name=method_name,

            )



        print(

            "Collected metric rows:",

            len(records),

        )



        all_records.extend(records)



    if not all_records:

        raise RuntimeError(

            "No result files were collected."

        )



    wide = pd.DataFrame(

        all_records

    )



    wide = canonicalize_dataframe_columns(

        wide

    )



    wide = infer_missing_identifiers(

        wide

    )



    wide = add_canonical_metrics(

        wide

    )



    sort_columns = [

        column

        for column in [

            "dataset",

            "method",

            "shot",

            "seed",

            "patient_id",

        ]

        if column in wide.columns

    ]



    wide = wide.sort_values(

        sort_columns

    ).reset_index(drop=True)



    long_dataframe = create_long_format(

        wide

    )



    summary = create_summary(

        long_dataframe

    )



    paired = create_paired_comparison(

        long_dataframe

    )



    wide_path = (

        output_dir

        / "all_results_wide.csv"

    )



    long_path = (

        output_dir

        / "all_results_long.csv"

    )



    summary_path = (

        output_dir

        / "all_results_summary.csv"

    )



    paired_path = (

        output_dir

        / "paired_3shot_vs_5shot.csv"

    )



    registry_copy_path = (

        output_dir

        / "experiment_registry_used.csv"

    )



    wide.to_csv(

        wide_path,

        index=False,

    )



    long_dataframe.to_csv(

        long_path,

        index=False,

    )



    summary.to_csv(

        summary_path,

        index=False,

    )



    paired.to_csv(

        paired_path,

        index=False,

    )



    registry.to_csv(

        registry_copy_path,

        index=False,

    )



    save_per_experiment_files(

        wide,

        output_dir,

    )



    print("\n" + "=" * 100)

    print("COLLECTION SUMMARY")

    print("=" * 100)



    print("Total result rows:", len(wide))

    print("Long-format rows:", len(long_dataframe))



    print("\nResults by dataset, method, and shot:")



    print(

        wide.groupby(

            [

                "dataset",

                "method",

                "shot",

            ],

            dropna=False,

        ).size().to_string()

    )



    print("\nSaved:")

    print(wide_path)

    print(long_path)

    print(summary_path)

    print(paired_path)

    print(registry_copy_path)





if __name__ == "__main__":

    main()


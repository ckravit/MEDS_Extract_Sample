import functools
import logging
import glob
import multiprocessing
import os
import re
import shutil
from argparse import ArgumentParser
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer

# ---- Sharding/streaming knobs (Option A) ----
# Target ≈ 2.0M events per shard (~3–4 GB per H5 file with your schema)
EVENTS_PER_SHARD = 2_000_000

# Flush cadence: append to H5 every # of events to cap peak RAM
# FLUSH_EVERY = 250_000
FLUSH_EVERY = 100_000


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pool_manager = multiprocessing.Manager()
# warned_codes = pool_manager.list()

# NOTE:
#  - Do NOT create multiprocessing objects at import time when using spawn.
#  - The tokenizer should NOT be passed to workers (pickling is costly/brittle).
#    We load it once per worker via an initializer and fetch it with get_tokenizer().

# --- Multiprocessing / tokenizer helpers (spawn-safe) ---
_TOKENIZER = None  # one per worker process

def _init_mp_spawn():
    """Force 'spawn' start method before creating any pools/managers."""
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set elsewhere; safe to ignore.
        pass

def _worker_init(tokenizer_name: str):
    """Runs once per worker. Loads the tokenizer from local disk."""
    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)
    
def get_tokenizer():
    """Retrieve the per-process tokenizer that _worker_init loaded."""
    return _TOKENIZER
####

def find_boundary_between(tuples_list, start, end):
    starts = [s for s, e in tuples_list]
    ends = [e for s, e in tuples_list]

    start_index = bisect_left(starts, start)
    end_index = bisect_right(ends, end)
    assert start_index < end_index

    return start_index, end_index


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "root",
        help="path to MEDS dataset. it can be either of directory or the exact file path "
        "with the file extension. if provided with directory, it tries to scan *.csv or "
        "*.parquet files contained in the directory, including sub-directories, to process "
        "all of them.",
    )
    parser.add_argument(
        "--metadata_dir",
        help="path to metadata directory for the input MEDS dataset, which contains codes.parquet",
    )

    parser.add_argument(
        "--birth_code", type=str, default="MEDS_BIRTH", help="string code for the birth event in the dataset."
    )

    parser.add_argument(
        "--cohort",
        type=str,
        help="path to the defined cohort, which must be a result of ACES. it can be either of "
        "directory or the exact file path that has the same extension with the MEDS dataset "
        "to be processed. the file structure of this cohort directory should be the same with "
        "the provided MEDS dataset directory to match each cohort to its corresponding shard "
        "data.",
    )
    parser.add_argument(
        "--cohort_label_name",
        type=str,
        default="boolean_value",
        help="column name in the cohort dataframe to be used for label",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="directory to save processed outputs.",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="whether or not to skip the processing if the output directory already "
        "exists.",
    )
    parser.add_argument(
        "--rebase",
        action="store_true",
        help="whether or not to rebase the output directory if exists.",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="whether or not to enable the debug mode, which forces the script to be run with "
        "only one worker."
    )
    parser.add_argument(
        "--workers",
        metavar="N",
        default=1,
        type=int,
        help="number of parallel workers.",
    )
    parser.add_argument(
        "--max_event_length",
        metavar="N",
        default=128,
        type=int,
        help="maximum number of tokens in an event.",
    )

    parser.add_argument(
        "--mimic_dir",
        default=None,
        help="path to directory for MIMIC-IV database containing hosp/ and icu/ as a subdirectory. "
        "this is used for addressing missing descriptions in the metadata for MIMIC-IV codes.",
    )

    return parser


def main():
    # --- Ensure we use 'spawn' BEFORE creating any pools/managers ---
    _init_mp_spawn()

    parser = get_parser()
    args = parser.parse_args()

    root_path = Path(args.root)
    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)
    mimic_dir = Path(args.mimic_dir) if args.mimic_dir is not None else None

    num_workers = max(args.workers, 1)
    if args.debug:
        num_workers = 1
        os.environ["RAYON_RS_NUM_CPUS"] = "1"
    else:
        cpu_count = multiprocessing.cpu_count()
        if num_workers > cpu_count:
            logger.warning(
                f"Number of workers (--workers) is greater than the number of available CPUs "
                f"({cpu_count}). Setting the number of workers to {cpu_count}."
            )
            num_workers = cpu_count

    if root_path.is_dir():
        data_paths = glob.glob(str(root_path / "**/*.csv"), recursive=True)
        if len(data_paths) == 0:
            data_paths = glob.glob(str(root_path / "**/*.parquet"), recursive=True)
        if len(data_paths) == 0:
            raise ValueError("Data directory does not contain any supported file formats: .csv or .parquet")
    else:
        data_paths = [root_path]

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    else:
        if args.rebase:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        elif output_dir.exists():
            if args.skip_if_exists:
                ls = glob.glob(str(output_dir / "**/*"), recursive=True)
                expected_files = []
                for subset in set(os.path.dirname(x) for x in data_paths):
                    expected_files.extend([
                        os.path.join(str(output_dir), os.path.basename(subset), f"{i}.h5")
                        for i in range(num_workers)
                    ])
                if set(expected_files).issubset(set(ls)):
                    logger.info(
                        f"Output directory already contains the expected files. Skipping the "
                        "processing as --skip-if-exists is set. If you want to rebase the directory, "
                        "please run the script with --rebase."
                    )
                    return
            else:
                raise ValueError(
                    f"File exists: '{str(output_dir.resolve())}'. If you want to rebase the "
                    "directory automatically, please run the script with --rebase."
                )

    label_col_name = args.cohort_label_name

    # IMPORTANT: do NOT instantiate the tokenizer here for passing to workers.
    # Instead, load it per worker via _worker_init using this local path:
    # tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    # tokenizer = AutoTokenizer.from_pretrained("/opt/data/workingdir/ckravit/MEDS/MEDS220/genHPF/emilyalsentzer/Bio_ClinicalBERT")
    tokenizer_name = "/opt/data/workingdir/ckravit/MEDS/MEDS220/genHPF/emilyalsentzer/Bio_ClinicalBERT"

    codes_metadata = pl.read_parquet(metadata_dir / "codes.parquet").to_pandas()
    codes_metadata = codes_metadata.set_index("code")["description"].to_dict()
    # do not allow to use static events or birth event
    birth_code = args.birth_code
    # if birth_code not in codes_metadata:
    #     print(
    #         f'"{birth_code}" is not found in the codes metadata, which may lead to '
    #         "unexpected results since we currently exclude this event from the input data. "
    #     )

    if mimic_dir is not None:
        d_items = pd.read_csv(mimic_dir / "icu" / "d_items.csv.gz")
        d_items["itemid"] = d_items["itemid"].astype("str")
        d_items = d_items.set_index("itemid")["label"].to_dict()
        d_labitems = pd.read_csv(mimic_dir / "hosp" / "d_labitems.csv.gz")
        d_labitems["itemid"] = d_labitems["itemid"].astype("str")
        d_labitems = d_labitems.set_index("itemid")["label"].to_dict()
    else:
        d_items = None
        d_labitems = None

    max_event_length = args.max_event_length

    progress_bar = tqdm(data_paths, total=len(data_paths))
    for data_path in progress_bar:
        progress_bar.set_description(str(data_path))

        data_path = Path(data_path)
        subdir = data_path.relative_to(root_path).parent
        if data_path.suffix == ".csv":
            data = pl.scan_csv(
                data_path,
                low_memory=True if args.debug else False,
            )
        elif data_path.suffix == ".parquet":
            data = pl.scan_parquet(
                data_path,
                parallel="none" if args.debug else "auto",
                low_memory=True if args.debug else False,
            )
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        data = data.with_columns(
            pl.when(pl.col("code") == birth_code).then(None).otherwise(pl.col("time")).alias("time")
        )
        data = data.drop_nulls(subset=["subject_id", "time"])

        cohort_path = Path(args.cohort) / subdir / data_path.name

        if cohort_path.suffix == ".csv":
            cohort = pl.scan_csv(cohort_path)
        elif cohort_path.suffix == ".parquet":
            cohort = pl.scan_parquet(cohort_path)
        else:
            raise ValueError(f"Unsupported file format: {cohort_path.suffix}")

        cohort = cohort.drop_nulls(label_col_name)
        cohort = cohort.unique()

        cohort = cohort.select(
            [
                pl.col("subject_id"),
                pl.col(label_col_name),
                # pl.col("input.end_summary").struct.field("timestamp_at_start").alias("starttime"),
                pl.col("prediction_time").alias("endtime"),
            ]
        )
        cohort = (
            cohort.group_by("subject_id", maintain_order=True)
            .agg(pl.col(["endtime", label_col_name]))
            .collect()
        )  # omitted "starttime"
        cohort_dict = {
            x["subject_id"]: {
                # "starttime": x["starttime"],
                "endtime": x["endtime"],
                "label": x[label_col_name],
            }
            for x in cohort.iter_rows(named=True)
        }

        def extract_cohort(row):
            subject_id = row["subject_id"]
            time = row["time"]
            if subject_id not in cohort_dict:
                # return {"cohort_start": None, "cohort_end": None, "cohort_label": None}
                return {"cohort_end": None, "cohort_label": None}

            cohort_criteria = cohort_dict[subject_id]
            # starts = cohort_criteria["starttime"]
            ends = cohort_criteria["endtime"]
            labels = cohort_criteria["label"]

            # for start, end, label in zip(starts, ends, labels):
            #     if start <= time and time <= end:
            #         return {"cohort_start": start, "cohort_end": end, "cohort_label": label}

            # assume it is possible that each event goes into multiple different cohorts
            cohort_ends = []
            cohort_labels = []
            for end, label in zip(ends, labels):
                if time <= end:
                    # return {"cohort_start": start, "cohort_end": end, "cohort_label": label}
                    cohort_ends.append(end)
                    cohort_labels.append(label)

            if len(cohort_ends) > 0:
                return {"cohort_end": cohort_ends, "cohort_label": cohort_labels}
            else:
                # return {"cohort_start": None, "cohort_end": None, "cohort_label": None}
                return {"cohort_end": None, "cohort_label": None}

        data = data.group_by(["subject_id", "time"], maintain_order=True).agg(pl.all())
        data = (
            data.with_columns(
                pl.struct(["subject_id", "time"])
                .map_elements(
                    extract_cohort,
                    return_dtype=pl.Struct(
                        {
                            "cohort_end": pl.List(pl.Datetime()),
                            "cohort_label": pl.List(pl.Boolean),
                        }
                    ),
                )
                .alias("cohort_criteria")
            )
            .unnest("cohort_criteria")
            .collect()
        )

        data = data.drop_nulls("cohort_label")

        data = data.with_columns(pl.col("time").dt.strftime("%Y-%m-%d %H:%M:%S").cast(pl.List(str)))
        data = data.with_columns(
            pl.col("time").list.sample(n=pl.col("code").list.len(), with_replacement=True)
        )

        if args.debug:
            data = data[:5000]

        if str(subdir) != ".":
            output_name = str(subdir)
        else:
            output_name = data_path.stem

        if not os.path.exists(output_dir / output_name):
            os.makedirs(output_dir / output_name)

        with open(str(output_dir / (output_name + ".tsv")), "a") as manifest_f:
            if os.path.getsize(output_dir / (output_name + ".tsv")) == 0:
                manifest_f.write(f"{output_dir}/{output_name}\n")

            must_have_columns = [
                "subject_id",
                "cohort_end",
                "cohort_label",
                "time",
                "code",
                "numeric_value",
            ]
            rest_of_columns = [x for x in data.columns if x not in must_have_columns]
            column_name_idcs = {col: i for i, col in enumerate(data.columns)}


            # Create a Manager/list for warnings *after* spawn is set (and only if we multiplex)
            if num_workers > 1:
                ctx = multiprocessing.get_context("spawn")
                manager = ctx.Manager()
                warned_codes = manager.list()
            else:
                warned_codes = []  # simple list in single-process path


            # meds_to_remed_partial = functools.partial(
            #     meds_to_remed,
            #     tokenizer,
            #     rest_of_columns,
            #     column_name_idcs,
            #     codes_metadata,
            #     output_dir,
            #     output_name,
            #     num_workers,
            #     d_items,
            #     d_labitems,
            #     warned_codes,
            #     max_event_length,
            #     args.debug,
            # )
            meds_to_remed_partial = functools.partial(
                meds_to_remed,
                tokenizer_name,
                rest_of_columns,
                column_name_idcs,
                codes_metadata,
                output_dir,
                output_name,
                num_workers,
                d_items,
                d_labitems,
                warned_codes,
                max_event_length,
                args.debug,
            )




            # meds --> remed
            logger.info(f"Start processing {data_path}")
            # if num_workers <= 1:
            #     length_per_subject_gathered = [meds_to_remed_partial(data)]
            #     del data
            if num_workers <= 1:
            # IMPORTANT: in single-process mode we still need a tokenizer.
            # Initialize it once here so get_tokenizer() returns a real tokenizer.
                _worker_init(tokenizer_name)
                # length_per_subject_gathered = [meds_to_remed_partial(data)]
                length_per_subject_gathered = [meds_to_remed_partial((0, data))]
                del data
            else:
                # subject_ids = data["subject_id"].unique().to_list()
                # n = num_workers
                # subject_id_chunks = [subject_ids[i::n] for i in range(n)]
                # data_chunks = []
                # for subject_id_chunk in subject_id_chunks:
                #     data_chunks.append(data.filter(pl.col("subject_id").is_in(subject_id_chunk)))
                # del data
                subject_ids = data["subject_id"].unique().to_list()
                n = num_workers
                subject_id_chunks = [subject_ids[i::n] for i in range(n)]
                data_chunks = []
                for worker_id, subject_id_chunk in enumerate(subject_id_chunks):
                    chunk_df = data.filter(pl.col("subject_id").is_in(subject_id_chunk))
                    data_chunks.append((worker_id, chunk_df))
                del data

                num_valid_data_chunks = sum(map(lambda x: len(x) > 0, data_chunks))
                if num_valid_data_chunks < num_workers:
                    raise ValueError(
                        "Number of valid data chunks (= number of unique subjects) were smaller "
                        "than the specified num workers (--workers) due to the small size of data. "
                        "Consider reducing the number of workers."
                    )

                # pool = multiprocessing.get_context("spawn").Pool(processes=num_workers)
                ctx = multiprocessing.get_context("spawn")
                # pool = ctx.Pool(processes=num_workers)

                # # the order is preserved
                # length_per_subject_gathered = pool.map(meds_to_remed_partial, data_chunks)
                # pool.close()
                # pool.join()


                # Load tokenizer once per worker via initializer
                with ctx.Pool(
                    processes=num_workers,
                    initializer=_worker_init,
                    initargs=(tokenizer_name,),
                ) as pool:
                    # order preserved
                    length_per_subject_gathered = pool.map(meds_to_remed_partial, data_chunks)
         


                del data_chunks

            if len(length_per_subject_gathered) != num_workers:
                raise ValueError(
                    "Number of processed workers were smaller than the specified num workers "
                    "(--workers) due to the small size of data. Consider reducing the number of "
                    "workers."
                )

            # for length_per_subject in length_per_subject_gathered:
            #     for subject_id, (length, shard_id) in length_per_subject.items():
            #         manifest_f.write(f"{subject_id}\t{length}\t{shard_id}\n")
                # Merge per-worker manifests into the split-level TSV (header is already written)
            split_dir = output_dir / output_name
            manifests_dir = split_dir / "manifests"
            final_manifest = output_dir / f"{output_name}.tsv"

            if manifests_dir.exists():
                with open(final_manifest, "a") as out_f:
                    for frag in sorted(manifests_dir.glob("manifest-worker-*.tsv")):
                        with open(frag, "r") as in_f:
                            shutil.copyfileobj(in_f, out_f)
                # Optional cleanup:
                # shutil.rmtree(manifests_dir, ignore_errors=True)


# def meds_to_remed(
#     tokenizer,
# def meds_to_remed(
#     tokenizer_name,
#     rest_of_columns,
#     column_name_idcs,
#     codes_metadata,
#     output_dir,
#     output_name,
#     num_shards,
#     d_items,
#     d_labitems,
#     warned_codes,
#     max_event_length,
#     debug,
#     df_chunk,
# ):
def meds_to_remed(
    tokenizer_name,                    # path string; NOT used directly here
    rest_of_columns,
    column_name_idcs,
    codes_metadata,
    output_dir,
    output_name,
    num_shards,
    d_items,
    d_labitems,
    warned_codes,
    max_event_length,
    debug,
    worker_and_chunk,                  # NEW: (worker_id, df_chunk)
):
    worker_id, df_chunk = worker_and_chunk
    # Fetch the per-process tokenizer loaded by _worker_init()
    tokenizer = get_tokenizer()

    # -- ADD D1
        # --- Per-worker sharding & streaming setup (Option A) ---
    import h5py
    from pathlib import Path

    # Each split writes under: <output_dir>/<output_name>/
    # split_dir = Path(output_dir)
    split_dir = Path(output_dir) / output_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Per-worker manifest fragments (merged in main at the end)
    manifests_dir = split_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    worker_manifest_path = manifests_dir / f"manifest-worker-{worker_id:03d}.tsv"

    # Per-worker shard bookkeeping
    shard_seq = 0
    events_in_current_shard = 0
    events_since_flush = 0
    current_h5 = None

    def _open_new_shard():
        nonlocal shard_seq, current_h5, events_in_current_shard, events_since_flush
        if current_h5 is not None:
            current_h5.flush()
            current_h5.close()
        shard_path = split_dir / f"worker-{worker_id:03d}-shard-{shard_seq:04d}.h5"
        current_h5 = h5py.File(shard_path, "w")
        events_in_current_shard = 0
        events_since_flush = 0
        shard_seq += 1
        return shard_path

    def _close_shard():
        nonlocal current_h5
        if current_h5 is not None:
            current_h5.flush()
            current_h5.close()
            current_h5 = None

    def _maybe_rollover():
        if events_in_current_shard >= EVENTS_PER_SHARD:
            _open_new_shard()

    # Start first shard now
    current_shard_path = _open_new_shard()


    # -- END D1
    code_matching_pattern = re.compile(r"\d+")

    def meds_to_remed_unit(row):
        events = []
        digit_offsets = []
        col_name_offsets = []
        for event_index in range(len(row[column_name_idcs["code"]])):
            event = ""
            digit_offset = []
            col_name_offset = []
            for col_name in ["code", "numeric_value"] + rest_of_columns:
                # do not process something like "icustay_id" or "hadm_id"
                if "id" in col_name:
                    continue

                col_event = row[column_name_idcs[col_name]][event_index]
                if col_event is not None:
                    col_event = str(col_event)
                    if col_name == "code":
                        if col_event in codes_metadata and codes_metadata[col_event] != "":
                            col_event = codes_metadata[col_event]
                        else:
                            do_break = False
                            items = col_event.split("//")
                            is_code = [bool(code_matching_pattern.fullmatch(item)) for item in items]
                            if True in is_code:
                                if d_items is not None and d_labitems is not None:
                                    code_idx = is_code.index(True)
                                    code = items[code_idx]

                                    if code in d_items:
                                        desc = d_items[code]
                                    elif code in d_labitems:
                                        desc = d_labitems[code]
                                    else:
                                        do_break = True

                                    if not do_break:
                                        items[code_idx] = desc
                                        col_event = "//".join(items)
                                else:
                                    do_break = True

                            if do_break and col_event not in warned_codes:
                                warned_codes.append(col_event)
                                logger.warning(
                                    "The dataset contains some codes that are not specified in "
                                    "the codes metadata, which may not be intended. Note that we "
                                    f"process this code as it is for now: {col_event}."
                                )
                    else:
                        col_event = re.sub(
                            r"\d*\.\d+",
                            lambda x: str(round(float(x.group(0)), 4)),
                            col_event,
                        )
                        event_offset = len(event) + len(col_name) + 1
                        digit_offset_tmp = [
                            g.span() for g in re.finditer(r"([0-9]+([.][0-9]*)?|[0-9]+|\.+)", col_event)
                        ]

                        internal_offset = 0
                        for start, end in digit_offset_tmp:
                            digit_offset.append(
                                (
                                    event_offset + start + internal_offset,
                                    event_offset + end + (end - start) * 2 + internal_offset,
                                )
                            )
                            internal_offset += (end - start) * 2

                        col_event = re.sub(r"([0-9\.])", r" \1 ", col_event)

                    col_name_offset.append((len(event), len(event) + len(col_name)))
                    event += " " + col_name + " " + col_event
            if len(event) > 0:
                events.append(event[1:])
                digit_offsets.append(digit_offset)
                col_name_offsets.append(col_name_offset)

        tokenized_events = tokenizer(
            events,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_event_length,
            truncation=True,
            return_tensors="np",
            return_token_type_ids=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        lengths_before_padding = tokenized_events["attention_mask"].sum(axis=1)

        input_ids = tokenized_events["input_ids"]
        dpe_ids = np.zeros(input_ids.shape, dtype=int)
        for i, digit_offset in enumerate(digit_offsets):
            for start, end in digit_offset:
                start_index, end_index = find_boundary_between(
                    tokenized_events[i].offsets[: lengths_before_padding[i] - 1],
                    start,
                    end,
                )

                # define dpe ids for digits found
                num_digits = end_index - start_index
                # 119: token id for "."
                num_decimal_points = (input_ids[i][start_index:end_index] == 119).sum()

                # integer without decimal point
                # e.g., for "1 2 3 4 5", assign [10, 9, 8, 7, 6]
                if num_decimal_points == 0:
                    dpe_ids[i][start_index:end_index] = list(range(num_digits + 5, 5, -1))
                # floats
                # e.g., for "1 2 3 4 5 . 6 7 8 9", assign [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
                elif num_decimal_points == 1:
                    num_decimals = (
                        num_digits
                        - np.where(input_ids[i][start_index:end_index] == 119)[0][0]  # 119: token id for "."
                    )
                    dpe_ids[i][start_index:end_index] = list(
                        range(num_digits + 5 - num_decimals, 5 - num_decimals, -1)
                    )
                # 1 > decimal points where we cannot define dpe ids
                else:
                    continue
        # define type ids
        # for column names: 2
        # for column values (contents): 3
        # for CLS tokens: 5
        # for SEP tokens: 6
        type_ids = np.zeros(input_ids.shape, dtype=int)
        type_ids[:, 0] = 5  # CLS tokens
        for i, col_name_offset in enumerate(col_name_offsets):
            type_ids[i][lengths_before_padding[i] - 1] = 6  # SEP tokens
            # fill with type ids for column values
            type_ids[i][1 : lengths_before_padding[i] - 1] = 3
            for start, end in col_name_offset:
                start_index, end_index = find_boundary_between(
                    tokenized_events[i].offsets[1 : lengths_before_padding[i] - 1],
                    start,
                    end,
                )
                # the first offset is always (0, 0) for CLS token, so we adjust it
                start_index += 1
                end_index += 1
                # finally replace with type ids for column names
                type_ids[i][start_index:end_index] = 2

        return np.stack([input_ids, type_ids, dpe_ids], axis=1).astype(np.uint16)

    events_data = []
    # worker_id = multiprocessing.current_process().name.split("-")[-1]
    # if worker_id == "MainProcess":
    #     worker_id = 0
    # else:
    #     # worker_id is incremental for every generated pool, so divide with num_shards
    #     worker_id = (int(worker_id) - 1) % num_shards  # 1-based -> 0-based indexing
    if worker_id == 0:
        progress_bar = tqdm(df_chunk.iter_rows(), total=len(df_chunk))
        # progress_bar.set_description(f"Processing from worker-{worker_id}:")
        progress_bar.set_description(f"Writing data from worker-{worker_id}:")

    else:
        progress_bar = df_chunk.iter_rows()

    for row in progress_bar:
        events_data.append(meds_to_remed_unit(row))
    data_length = list(map(len, events_data))
    data_index_offset = np.zeros(len(data_length), dtype=np.int64)
    data_index_offset[1:] = np.cumsum(data_length[:-1])
    data_index = pl.Series(
        "data_index",
        map(
            lambda x: [data_index_offset[x] + y for y in range(data_length[x])],
            range(len(data_length)),
        ),
    )
    # Do not want monolith
    # events_data = np.concatenate(events_data)

    df_chunk = df_chunk.select(["subject_id", "cohort_end", "cohort_label", "time"])
    df_chunk = df_chunk.insert_column(4, data_index)
    df_chunk = df_chunk.explode(["cohort_end", "cohort_label"])
    df_chunk = df_chunk.group_by(
        # ["subject_id", "cohort_start", "cohort_end", "cohort_label"], maintain_order=True
        ["subject_id", "cohort_end", "cohort_label"],
        maintain_order=True,
    ).agg(pl.all())

    if debug:
        df_chunk = df_chunk.with_columns(
            [
            pl.col("time").map_elements(lambda x: x[-100:], return_dtype=pl.List(pl.List(str))),
            pl.col("data_index").map_elements(lambda x: x[-100:], return_dtype=pl.List(pl.List(int)))
            ]
        )

    df_chunk = df_chunk.sort(by=["subject_id", "cohort_end"])
    # regard {subject_id} as {cohort_id}: {subject_id}_{cohort_number}
    df_chunk = df_chunk.with_columns(pl.col("subject_id").cum_count().over("subject_id").alias("suffix"))
    df_chunk = df_chunk.with_columns(
        (pl.col("subject_id").cast(str) + "_" + pl.col("suffix").cast(str)).alias("subject_id")
    )
    # data = data.drop("suffix", "cohort_start", "cohort_end")
    df_chunk = df_chunk.drop("suffix", "cohort_end")

    length_per_subject = {}
    progress_bar = tqdm(
        df_chunk.iter_rows(),
        total=len(df_chunk),
        desc=f"Writing data from worker-{worker_id}:",
    )

    # for sample in progress_bar:
    #     with h5py.File(str(output_dir / output_name / f"{worker_id}.h5"), "a") as f:
    #         if "ehr" in f:
    #             result = f["ehr"]
    #         else:
    #             result = f.create_group("ehr")

    #         sample_result = result.create_group(sample[0])

    #         times = np.concatenate(sample[2])
    #         data_indices = np.concatenate(sample[3])
    #         if debug:
    #             data_indices = data_indices[-100:]
    #             times = times[-100:]

    #         data = events_data[data_indices]
    #         sample_result.create_dataset("hi", data=data, dtype="i2", compression="lzf", shuffle=True)

    #         times = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in times]
    #         times = np.cumsum(np.diff(times))
    #         times = list(map(lambda x: round(x.total_seconds() / 60), times))
    #         times = np.array([0] + times)

    #         sample_result.create_dataset("time", data=times, dtype="i")
    #         sample_result.create_dataset("label", data=int(sample[1]))

    #         length_per_subject[sample[0]] = (len(times), worker_id)
    for sample in progress_bar:
        # sample fields (unchanged from your code):
        #   sample[0] -> subject_id (string after your suffixing)
        #   sample[1] -> label (scalar)
        #   sample[2] -> list of time lists (you concatenate them below)
        #   sample[3] -> list of data_index lists (global event indices for this subject)

        subject_id = sample[0]
        label_val  = int(sample[1])

        # Build times (unchanged)
        times = np.concatenate(sample[2])
        if debug:
            times = times[-100:]
        times_dt = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in times]
        times_delta = np.cumsum(np.diff(times_dt))
        times_minutes = list(map(lambda x: round(x.total_seconds() / 60), times_delta))
        times_arr = np.array([0] + times_minutes, dtype=np.int32)

        # Build this subject's hi by GATHERING from events_data (list of arrays)
        # without concatenating all events globally.
        #
        # We will map each global index back to (row_idx, offset_in_row) using
        # your existing data_index_offset and data_length arrays.
        data_indices = np.concatenate(sample[3])
        if debug:
            data_indices = data_indices[-100:]

        # Prepare a container and fill row-by-row to avoid a huge global array.
        E = int(data_indices.shape[0])
        hi_subj = np.empty((E, 3, max_event_length), dtype=np.uint16)

        # Helper: for a given global index g, find row j s.t. data_index_offset[j] <= g < data_index_offset[j]+data_length[j]
        from bisect import bisect_right
        for write_pos, g in enumerate(data_indices):
            j = bisect_right(data_index_offset, g) - 1  # row index in events_data
            local = int(g - data_index_offset[j])       # offset within that row's array
            hi_subj[write_pos] = events_data[j][local]

        # If adding this subject would overflow the shard budget, roll to a new shard first
        if events_in_current_shard + E > EVENTS_PER_SHARD:
            _close_shard()
            current_shard_path = _open_new_shard()

        # Ensure the root group exists in this shard
        if "ehr" in current_h5:
            result = current_h5["ehr"]
        else:
            result = current_h5.create_group("ehr")

        # Preserve your naming convention (you already suffixed subject_id earlier)
        sample_result = result.create_group(subject_id)
        sample_result.create_dataset("hi",   data=hi_subj,   dtype="i2", compression="lzf", shuffle=True)
        sample_result.create_dataset("time", data=times_arr, dtype="i")
        sample_result.create_dataset("label", data=label_val)

        # Update counters and manifest
        events_in_current_shard += E
        events_since_flush      += E
        length_per_subject[subject_id] = (int(times_arr.shape[0]), worker_id)

        # Append one manifest row for this subject (subject_id, events, shard filename)
        from pathlib import Path as _P
        with open(worker_manifest_path, "a") as _mf:
            _mf.write(f"{subject_id}\t{E}\t{_P(current_shard_path).name}\n")

        # Flush cadence to cap peak RAM
        if events_since_flush >= FLUSH_EVERY:
            current_h5.flush()
            events_since_flush = 0

        # Rollover if shard is full
        if events_in_current_shard >= EVENTS_PER_SHARD:
            _close_shard()
            current_shard_path = _open_new_shard()
    # Close last shard from this worker
    _close_shard()

    del df_chunk

    return length_per_subject


if __name__ == "__main__":
    main()

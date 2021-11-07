import re
from collections import Counter
from multiprocessing import cpu_count, Pool
from pathlib import Path
from typing import List

from tqdm.auto import tqdm

from anlp_a2.config import DATASET_DIR

SUBTEXT_RE = re.compile(r"\[.*\]|[^ a-z]")


def preprocess_file(filepath: Path):
    dst = DATASET_DIR / "train" / f"{filepath.name.split('-')[0]}.txt"
    processed_lines: List[str] = []
    wf = Counter()

    for line in filepath.read_text().split("\n"):
        processed_line = re.sub(SUBTEXT_RE, "", line.lower()).strip()
        # ignore no/single word lines
        if len(processed_line) == 0 or processed_line.find(" ") == -1:
            continue

        for w in processed_line.split(" "):
            wf[w] += 1

        processed_lines.append(processed_line + " .")

    dst.write_text("\n".join(processed_lines))
    return wf


def preprocess_files(file_list: List[Path], index: int):
    wf = Counter()
    if index == 0:
        file_iter = tqdm(file_list)
    else:
        file_iter = file_list

    for filepath in file_iter:
        wf += preprocess_file(filepath)

    return wf


def preprocess_dataset(n_procs=cpu_count() - 1):
    file_list = list(DATASET_DIR.glob("raw/*-trans.text"))
    n_files = len(file_list)
    chunk_size = n_files // n_procs

    split_indices = [i * chunk_size for i in range(n_procs)]
    split_indices.append(n_files)

    # reverse the argument list
    # this displays the progress bar where `index` is 0
    # by reversing the list, the progress bar mimics actual completion closely as
    # the progress bar process will be the last one to start
    args = [
        (file_list[split_indices[i] : split_indices[i + 1]], i)
        for i in range(len(split_indices) - 1)
    ][::-1]

    # populate vocab with EOS, BOS, UNK, fullstop
    wf = Counter(
        {
            "</S>": float("inf"),
            "<S>": float("inf"),
            "@@UNKNOWN@@": float("inf"),
            ".": float("inf"),
        }
    )

    with Pool(n_procs) as pool:
        local_wfs = pool.starmap(preprocess_files, args)

    for local_wf in tqdm(local_wfs, desc="Accumulating word frequencies"):
        wf += local_wf

    # vocabulary file
    vocab_file = (DATASET_DIR / "tokens.txt").open("w+")
    for (w, f) in wf.items():
        if f >= 5:
            vocab_file.write(f"{w}\n")
    vocab_file.close()

    # non padded namespaces file
    (DATASET_DIR / "non_padded_namespaces.txt").write_text(
        "\n".join(["*labels", "*tags"])
    )


if __name__ == "__main__":
    (DATASET_DIR / "train").mkdir(parents=True, exist_ok=True)
    preprocess_dataset()

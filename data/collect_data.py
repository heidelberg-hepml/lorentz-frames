import numpy as np
import os, sys
import hashlib
import tarfile
import wget

# dataset sizes: toptagging 1.5G, event-generation 4.7G, JetClass ~190G (full)
BASE_URL = "https://www.thphys.uni-heidelberg.de/~plehn/data"
FILENAMES = {
    "toptagging": "toptagging_full.npz",
    "event-generation": "event_generation_ttbar.hdf5",
}
DATA_DIR = "data"

# JetClass (Pythia) -- https://zenodo.org/records/6619768. The repo's JetClass loader
# (experiments/tagging/jetclassexperiment.py) reads
#     <data.data_dir>/{train_100M,val_5M,test_20M}/<ClassName>_<NNN>.root
# and config/jctagging.yaml sets data.data_dir = data/JetClass/Pythia -- which is exactly
# the layout these official tars unpack to, so no post-processing or path edits are needed.
JETCLASS_BASE = "https://zenodo.org/record/6619768/files"
JETCLASS = {
    # split: (extract subdir under data/JetClass, [(tar filename, md5), ...])
    "train": (
        "Pythia/train_100M",
        [
            (f"JetClass_Pythia_train_100M_part{i}.tar", md5)
            for i, md5 in enumerate(
                [
                    "de4fd2dca2e68ab3c85d5cfd3bcc65c3",
                    "9722a359c5ef697bea0fbf79bf50f003",
                    "1e9f66cd1f915f9d10e90ae1d7761720",
                    "47348fc8985319fa4806da87500482fa",
                    "6b0ce16bd93b442a8d51914466990279",
                    "416e347512e716de51d392bee327b8e9",
                    "e9b9c1557b1b39bf0a16e4ab631ae451",
                    "5bfc6cb285ccb7680cefa9ac82ad1a2e",
                    "540c1a0d66dfad78d2b363c5740ccf86",
                    "668f40b3275167ff7104c48317c0ae2a",
                ]
            )
        ],
    ),
    "val": ("Pythia", [("JetClass_Pythia_val_5M.tar", "7235ccb577ed85023ea3ab4d5e6160cf")]),
    "test": ("Pythia", [("JetClass_Pythia_test_20M.tar", "64e5156d26d101adeb43b8388207d767")]),
}

# TopTagXL (binary qcd-vs-top at JetClass scale, 100M/25M/10M jets) --
# https://zenodo.org/records/10878355, the LLoCa paper's extended top-tagging set.
# Unlike JETCLASS above, the file list and md5 checksums are pulled from the Zenodo
# API at download time (nothing hardcoded here can go stale). The loader
# (experiments/tagging/toptagxlexperiment.py) reads
#     <data.data_dir>/{train_100M,test_25M,val_10M}/{qcd,top}_<NNN>.root
# with file numbering continuous across the splits (000-499 / 500-624 / 625-674),
# and config/toptagxl.yaml sets data.data_dir = data/toptagxl.
TOPTAGXL_RECORD = "10878355"
TOPTAGXL_DIR = "toptagxl"
TOPTAGXL_FOLDERS = ("train_100M", "test_25M", "val_10M")


def load(filename):
    url = os.path.join(BASE_URL, filename)
    print(f"Started to download {url}")
    target_path = os.path.join(DATA_DIR, filename)
    wget.download(url, out=target_path)
    print("")
    print(f"Successfully downloaded {target_path}")


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def collect_jetclass(splits):
    """Download + verify + extract the JetClass (Pythia) tars for the given splits.

    Idempotent: a tar whose md5 already matches is not re-downloaded, and an already
    extracted tar (marked by a hidden ``.<tar>.extracted`` file) is skipped. The tars are
    large (~190 GB total) and can be deleted after extraction to reclaim disk.
    """
    base = os.path.join(DATA_DIR, "JetClass")
    for split in splits:
        subdir, files = JETCLASS[split]
        dest = os.path.join(base, subdir)
        os.makedirs(dest, exist_ok=True)
        for fname, md5 in files:
            tar_path = os.path.join(base, fname)
            marker = os.path.join(base, f".{fname}.extracted")
            if os.path.exists(marker):
                print(f"{fname} already extracted, skipping")
                continue
            url = f"{JETCLASS_BASE}/{fname}"
            if os.path.exists(tar_path) and _md5(tar_path) == md5:
                print(f"{fname} already downloaded (md5 ok)")
            else:
                if os.path.exists(tar_path):
                    os.remove(tar_path)  # partial/corrupt -> re-download
                print(f"Downloading {url}")
                wget.download(url, out=tar_path)
                print("")
                if _md5(tar_path) != md5:
                    raise RuntimeError(f"md5 mismatch for {fname}; delete it and retry")
            print(f"Extracting {fname} -> {dest}")
            with tarfile.open(tar_path) as tar:
                try:
                    tar.extractall(dest, filter="data")  # python >= 3.12 safe extraction
                except TypeError:
                    tar.extractall(dest)
            open(marker, "w").close()
            print(f"Extracted {fname}  (you may delete {tar_path} to reclaim disk)")
    print(f"JetClass ready under {base}/Pythia -- matches config/jctagging.yaml data.data_dir.")


def _zenodo_record_files(record_id):
    """File inventory ``[(name, md5, url), ...]`` of a Zenodo record via its public API."""
    import json
    import urllib.request

    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api_url) as response:
        record = json.load(response)
    files = []
    for entry in record["files"]:
        name = entry.get("key") or entry.get("filename")
        checksum = entry.get("checksum") or ""
        md5 = checksum.split(":", 1)[-1] if checksum else None  # zenodo format 'md5:<hex>'
        url = (entry.get("links") or {}).get("self") or (
            f"https://zenodo.org/records/{record_id}/files/{name}?download=1"
        )
        files.append((name, md5, url))
    return files


def collect_toptagxl(splits, inventory=None):
    """Download + verify + extract the TopTagXL record for the given splits.

    Mirrors ``collect_jetclass`` (md5 verification, idempotent ``.<file>.extracted``
    markers, tars deletable after extraction), except the file list + checksums come
    from the Zenodo API at runtime. ``splits`` filters record files by name substring
    ('train'/'val'/'test'); files matching no split keyword (e.g. a README) are only
    fetched when all three splits are requested.
    """
    dest = os.path.join(DATA_DIR, TOPTAGXL_DIR)
    os.makedirs(dest, exist_ok=True)
    if inventory is None:
        inventory = _zenodo_record_files(TOPTAGXL_RECORD)
    want_all = {"train", "val", "test"}.issubset(splits)

    selected = []
    for name, md5, url in inventory:
        matched = [s for s in ("train", "val", "test") if s in name.lower()]
        if want_all or any(s in splits for s in matched):
            selected.append((name, md5, url))
    if not selected:
        raise RuntimeError(
            f"No files in Zenodo record {TOPTAGXL_RECORD} match split(s) {splits}; "
            f"record contains: {[name for name, _, _ in inventory]}"
        )

    for name, md5, url in selected:
        path = os.path.join(dest, name)
        marker = os.path.join(dest, f".{name}.extracted")
        if os.path.exists(marker):
            print(f"{name} already extracted, skipping")
            continue
        if os.path.exists(path) and md5 is not None and _md5(path) == md5:
            print(f"{name} already downloaded (md5 ok)")
        else:
            if os.path.exists(path):
                os.remove(path)  # partial/corrupt -> re-download
            print(f"Downloading {url}")
            wget.download(url, out=path)
            print("")
            if md5 is not None and _md5(path) != md5:
                raise RuntimeError(f"md5 mismatch for {name}; delete it and retry")
        if tarfile.is_tarfile(path):
            print(f"Extracting {name} -> {dest}")
            with tarfile.open(path) as tar:
                try:
                    tar.extractall(dest, filter="data")  # python >= 3.12 safe extraction
                except TypeError:
                    tar.extractall(dest)
            open(marker, "w").close()
            print(f"Extracted {name}  (you may delete {path} to reclaim disk)")
        else:
            open(marker, "w").close()

    present = [f for f in TOPTAGXL_FOLDERS if os.path.isdir(os.path.join(dest, f))]
    missing = [f for f in TOPTAGXL_FOLDERS if f not in present]
    if missing and not want_all:
        missing = [f for f in missing if any(s in f for s in splits)]
    if missing:
        print(
            f"WARNING: expected folder(s) {missing} not found under {dest} after "
            f"extraction -- inspect the extracted layout and symlink/move it so the "
            f"loader finds <data_dir>/<split>/<class>_<NNN>.root (config/toptagxl.yaml "
            f"data.data_dir = {dest})."
        )
    else:
        print(f"TopTagXL ready under {dest} -- matches config/toptagxl.yaml data.data_dir.")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python data/collect_data.py "
            "<toptagging | eventgen | jetclass [train|val|test|all] "
            "| toptagxl [train|val|test|all]>"
        )
        sys.exit(1)
    dataset = sys.argv[1]

    # collect toptagging dataset
    # this is a npz version of the original dataset at https://zenodo.org/records/2603256
    filename = FILENAMES["toptagging"]
    if dataset == "toptagging":
        load(filename)

    # collect event generation dataset
    # this dataset is described in https://arxiv.org/abs/2411.00446
    filename = FILENAMES["event-generation"]
    if dataset == "eventgen":
        import h5py
        import hdf5plugin  # noqa: F401  (registers the hdf5 filters used by the file)

        load(filename)
        filename = os.path.join(DATA_DIR, filename)
        with h5py.File(filename, "r") as file:
            for njets in range(5):
                data = file[f"ttbar+{njets}jet"]
                target_path = os.path.join(DATA_DIR, f"ttbar_{njets}j.npy")
                np.save(target_path, data)
                print(f"Successfully created {target_path}")

    # collect the JetClass tagging dataset (https://zenodo.org/records/6619768)
    # second arg selects the split(s); default 'all'. Full download is ~190 GB.
    if dataset == "jetclass":
        arg = sys.argv[2] if len(sys.argv) > 2 else "all"
        splits = ["train", "val", "test"] if arg == "all" else [arg]
        unknown = [s for s in splits if s not in JETCLASS]
        if unknown:
            print(f"Unknown JetClass split(s) {unknown}; choose from train/val/test/all")
            sys.exit(1)
        collect_jetclass(splits)

    # collect the TopTagXL dataset (https://zenodo.org/records/10878355)
    # second arg selects the split(s); default 'all'. ~JetClass-sized download; the
    # file list + md5 checksums come from the Zenodo API at download time.
    if dataset == "toptagxl":
        arg = sys.argv[2] if len(sys.argv) > 2 else "all"
        splits = ["train", "val", "test"] if arg == "all" else [arg]
        unknown = [s for s in splits if s not in ("train", "val", "test")]
        if unknown:
            print(f"Unknown TopTagXL split(s) {unknown}; choose from train/val/test/all")
            sys.exit(1)
        collect_toptagxl(splits)


if __name__ == "__main__":
    main()

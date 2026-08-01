import numpy as np
import os, sys
import hashlib
import tarfile
import time
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
# The dataset ships as split-tagged tars (train parts 1-5 are QCD, 6-10 are top;
# see the Zenodo description), not the per-class-numbered layout JetClass uses. The
# file list + md5 checksums below are the record's published inventory (revision 8);
# the collector verifies each tar against its md5 and extracts it. config/toptagxl.yaml
# sets data.data_dir = data/toptagxl and the loader
# (experiments/tagging/toptagxlexperiment.py) reads
#     <data.data_dir>/{train_100M,test_25M,val_10M}/{qcd,top}_<NNN>.root
# -- if the tars do not unpack to exactly that tree, the collector prints how to
# symlink/rename (the internal .root layout is not documented on Zenodo).
TOPTAGXL_RECORD = "10878355"
TOPTAGXL_DIR = "toptagxl"
TOPTAGXL_FOLDERS = ("train_100M", "test_25M", "val_10M")
TOPTAGXL_BASE = f"https://zenodo.org/api/records/{TOPTAGXL_RECORD}/files"
TOPTAGXL = {
    # split: [(tar filename, md5), ...]  (simulation_cards.zip is metadata, skipped)
    "train": [
        (f"TopTagXL_train_part{i}.tar", md5)
        for i, md5 in zip(
            range(1, 11),
            [
                "4d179db4694c793a62bd42500d44c7e7",  # part1  (QCD)
                "1bf6264a4245cbcb4d1881c8836280a4",  # part2  (QCD)
                "8f5bf1f62a9ef91234820bed749cf7e1",  # part3  (QCD)
                "90544659da6724e7a4023c0bd5958f68",  # part4  (QCD)
                "2b753db0c733d81e6b1cc3ffd06dc8e9",  # part5  (QCD)
                "6af57a9a14141b146e3a44d8d70b5014",  # part6  (top)
                "f37cd5f2818d8dcc63ad62517d5f308e",  # part7  (top)
                "28d32c8e4b1614f42a5e9ec8b0b7afd9",  # part8  (top)
                "e250a00a60a146730fb61a4a1cb4306d",  # part9  (top)
                "e562c15ea5fa0945f127c59440127250",  # part10 (top)
            ],
        )
    ],
    "test": [
        ("TopTagXL_test_part1.tar", "a3927369b70eba5f9d77d3b6da0c26a4"),
        ("TopTagXL_test_part2.tar", "c48c81b3cd2a1dfb0a679033e5d74f6c"),
    ],
    "val": [("TopTagXL_val.tar", "120a53c067c3169bc8d3e2f1e68f0a32")],
}


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


def _refresh_times(root):
    """Stamp everything under ``root`` with the current time.

    tarfile.extractall restores each member's ARCHIVE timestamps (atime = mtime =
    when the file was packed, often years ago), so freshly extracted files look
    years-idle to scratch purge daemons and get deleted on the next sweep -- observed
    on Oscar: a fully extracted JetClass tree wiped days after extraction, with only
    the fresh 0-byte .extracted markers surviving. Cheap insurance: touch it all.
    """
    now = time.time()
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            try:
                os.utime(os.path.join(dirpath, f), (now, now))
            except OSError:
                pass


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
                if not os.listdir(dest):
                    print(
                        f"WARNING: {fname} is marked extracted but {dest} is EMPTY -- "
                        f"scratch purge? Delete the .*.extracted markers under {base} "
                        f"to force a re-download/re-extract."
                    )
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
            _refresh_times(dest)  # archive timestamps look years-idle to purge daemons
            open(marker, "w").close()
            print(f"Extracted {fname}  (you may delete {tar_path} to reclaim disk)")
    print(f"JetClass ready under {base}/Pythia -- matches config/jctagging.yaml data.data_dir.")


def collect_toptagxl(splits):
    """Download + verify + extract the TopTagXL tars for the given splits.

    Mirrors ``collect_jetclass`` exactly (hardcoded md5 verification, idempotent
    ``.<file>.extracted`` markers, tars deletable after extraction). The only
    structural difference from JetClass is that TopTagXL ships split-tagged tars
    (10 train / 2 test / 1 val) rather than per-class-numbered files, so there is no
    per-class subdir list -- the tars carry their own internal layout, and the
    post-extract check below reports if it does not match what the loader expects.
    """
    dest = os.path.join(DATA_DIR, TOPTAGXL_DIR)
    os.makedirs(dest, exist_ok=True)
    for split in splits:
        for fname, md5 in TOPTAGXL[split]:
            tar_path = os.path.join(dest, fname)
            marker = os.path.join(dest, f".{fname}.extracted")
            if os.path.exists(marker):
                if not any(
                    os.path.isdir(os.path.join(dest, d)) and os.listdir(os.path.join(dest, d))
                    for d in TOPTAGXL_FOLDERS
                ):
                    print(
                        f"WARNING: {fname} is marked extracted but no split folder under "
                        f"{dest} has any files -- scratch purge? Delete the .*.extracted "
                        f"markers there to force a re-download/re-extract."
                    )
                print(f"{fname} already extracted, skipping")
                continue
            url = f"{TOPTAGXL_BASE}/{fname}/content"
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
            _refresh_times(dest)  # archive timestamps look years-idle to purge daemons
            open(marker, "w").close()
            print(f"Extracted {fname}  (you may delete {tar_path} to reclaim disk)")

    want = [f for f in TOPTAGXL_FOLDERS if any(s in f for s in splits) or set(splits) >= {"train", "val", "test"}]
    missing = [f for f in want if not os.path.isdir(os.path.join(dest, f))]
    if missing:
        print(
            f"WARNING: expected folder(s) {missing} not found under {dest} after "
            f"extraction. Inspect the extracted layout and symlink/rename it so the "
            f"loader finds <data_dir>/<split_folder>/<class>_<NNN>.root, i.e. "
            f"{TOPTAGXL_FOLDERS} with qcd_/top_ .root files (config/toptagxl.yaml "
            f"data.data_dir = {dest}); the Zenodo tars do not document their internal tree."
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

    # collect the TopTagXL dataset (https://zenodo.org/records/10878355) introduced in SciPost LLoCa paper
    # second arg selects the split(s); default 'all'. ~JetClass-sized download; the
    # file list + md5 checksums come from the Zenodo API at download time.
    if dataset == "toptagxl":
        arg = sys.argv[2] if len(sys.argv) > 2 else "all"
        splits = ["train", "val", "test"] if arg == "all" else [arg]
        unknown = [s for s in splits if s not in TOPTAGXL]
        if unknown:
            print(f"Unknown TopTagXL split(s) {unknown}; choose from train/val/test/all")
            sys.exit(1)
        collect_toptagxl(splits)


if __name__ == "__main__":
    main()

import glob


def to_filelist(flist):
    # keyword-based: 'a:/path/to/a b:/path/to/b'
    file_dict = {}
    missing = []
    for f in flist:
        if ":" in f:
            name, fp = f.split(":")
        else:
            name, fp = "_", f
        files = glob.glob(fp)
        if not files:
            # a glob that matches nothing is silently dropped otherwise -- a mistyped
            # or out-of-range file range then resolves to fewer files (or zero) than
            # requested. Surface it.
            missing.append(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    filelist = sum(file_dict.values(), [])
    assert len(filelist) == len(set(filelist))
    if missing:
        print(
            f"[miniweaver] WARNING: {len(missing)} of {len(flist)} requested file "
            f"pattern(s) matched no files on disk (e.g. {missing[0]}); "
            f"resolved to {len(filelist)} actual file(s)."
        )
    return file_dict, filelist

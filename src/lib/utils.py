from pathlib import Path


def memoizer(
    load,
    dump,
    path: Path,
):
    def get_memoized():
        with open(path, "rb") as f:
            data = load(f)
        return data

    def memoize(data):
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            dump(data, f)

    def has_memoized():
        return path.exists()

    return get_memoized, memoize, has_memoized


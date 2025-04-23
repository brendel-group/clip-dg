import pickle as pk
from pathlib import Path
from typing import Any, Optional

import numpy as np


def save_if_out_specified(data, out_file=None):
    if out_file is not None:
        if not isinstance(out_file, Path):
            out_file = Path(out_file)

        print(f"Saving to {out_file}...")
        with out_file.open("wb") as f:
            pk.dump(data, f)


def load_if_file(arg: Any) -> Optional[Any]:
    file_candidate = arg

    if isinstance(file_candidate, str):
        file_candidate = Path(file_candidate)
    if (
        isinstance(file_candidate, Path)
        and file_candidate.is_file()
        and file_candidate.exists()
    ):
        with file_candidate.open("rb") as f:
            if file_candidate.suffix in [".np", ".npy"]:
                data = np.load(f)
            else:
                try:
                    data = pk.load(f)
                except pk.UnpicklingError:
                    raise ValueError(
                        f"Can't open {file_candidate} with pickle. Are you sure it's a pickled file?"
                    )
        return data

    return arg

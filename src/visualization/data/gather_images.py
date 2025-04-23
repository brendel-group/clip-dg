import re
import tarfile
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import DefaultDict, List, Optional, OrderedDict, Union, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
from .utils import save_if_out_specified, load_if_file
from torchvision.transforms.functional import resize, center_crop
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os


def get_ids_from_idcs(
    idcs_by_file,
    meta_dir,
    out_file=None,
) -> List[str]:
    """Get image IDs based on their index in an embedding file.

    Args:
        idcs_by_file (DefaultDict[int  |  str, List[int]]): list of indices per embedding file
        meta_dir (Path | str): directory containing meta data per embedding file

    Returns:
        List[str]: list of all image ids
    """
    if not isinstance(meta_dir, Path):
        meta_dir = Path(meta_dir)

    img_ids = []
    for file_id, img_idcs in tqdm(idcs_by_file.items()):
        meta_file = meta_dir / f"metadata_{int(file_id):02d}.parquet"
        metadata = pd.read_parquet(meta_file)
        ids = metadata["image_path"].tolist()

        for img_idx in img_idcs:
            img_ids.append(ids[img_idx])

    # save results
    save_if_out_specified(img_ids, out_file)

    return img_ids


def get_imgs_from_ids_folder(
        ids,
        img_dir,
) -> List[Image.Image]:
    ''' Gets images in a list from ImageNet-X
    '''
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    dataset = datasets.ImageFolder(img_dir, transform=transform)

    imgs = []
    for id in ids:
        img = dataset[id][0]
        imgs.append(img)

    return imgs


def get_imgs_from_ids(
    ids,
    img_dir,
    n_jobs=1,
    out_file=None,
) -> List[Image.Image]:
    """Load images specified by their ids from tarballs located in `img_dir` in parallel.

    Use this function if
     - you need to load images from within python
     - you want to load a comparatively small number of images (e.g. for visualization)
     - you need the order of images to be the same as the order of IDs
     - you might want to load duplicate images

    Otherwise, use the slurm-parallelized script `sampling/subsample_dataset.sh`, which should
    generally be faster.

    Args:
        ids (List[int] | List[str]): image IDs, assumed to be 9 digits, where the first 5 digits
            are the id of the tarball
        img_dir (Path | str): directory containing all tarballs
        n_jobs (int, optional): number of parallel runners; if in doubt set to 4*n_cpus

    Returns:
        List: images in same order as provided IDs
    """

    if not isinstance(img_dir, Path):
        img_dir = Path(img_dir)

    print("Preprocessing IDs...")
    # convert ids to strings, which will make subsequent path operations easier
    ids = _format_ids(ids)

    # sort IDs
    idcs_sorted = np.argsort(ids)
    idcs_reverse = np.argsort(idcs_sorted)
    ids = ids[idcs_sorted]

    # split into tarballs by number of runners
    ids_by_tarball = {}
    for id in ids:
        tarball_id = id[:5]
        if tarball_id in ids_by_tarball:
            ids_by_tarball[tarball_id].append(id)
        else:
            ids_by_tarball[tarball_id] = [id]

    # load images from tarballs in parallel
    print("Loading images...")
    res = Parallel(n_jobs=n_jobs)(
        delayed(_get_imgs_runner)(tarball_id, img_ids, img_dir)
        for tarball_id, img_ids in tqdm(ids_by_tarball.items())
    )

    print("Postprocessing images...")
    # concat results
    imgs_by_tarball = {tarball_id: imgs for tarball_id, imgs in res}
    imgs = np.concatenate(
        [
            np.array(imgs_by_tarball[tarball_id], dtype=object)
            for tarball_id in ids_by_tarball.keys()
        ]
    )

    # reorder into original order
    imgs = imgs[idcs_reverse]

    # save results
    save_if_out_specified(imgs, out_file)

    return imgs


def _format_ids(ids) -> np.ndarray:
    """
    Formats IDs into a consistent string representation.

    Args:
        ids: IDs to be formatted.

    Returns:
        Formatted IDs as a numpy array.
    """
    out = []

    for id in ids:
        if isinstance(id, str):
            assert re.match(r"\d{9}", id), f"Received invalid id: {id}"  # Check if string ID is valid
            out.append(id)
        else:
            assert 0 < id < 999999999, f"Received invalid id: {id}"  # Check if numerical ID is within valid range
            out.append(f"{id:09d}")

    return np.array(out)


def _get_imgs_runner(
    tarball_id: str,  # ID of the tarball
    ids: List[str],  # List of image IDs
    img_dir: Path  # Directory where tarball and images are stored
) -> Tuple[str, List[Image.Image]]:
    """
    Extracts images from a tarball based on provided IDs.

    Args:
        tarball_id: ID of the tarball containing images.
        ids: List of image IDs to extract.
        img_dir: Directory where tarball and images are stored.

    Returns:
        Tuple containing tarball ID and list of extracted images.
    """
    tarball = img_dir / f"{tarball_id}.tar"
    try:
        with tarfile.open(tarball) as tb:
            imgs = []
            for id in ids:
                fileinfo = tb.getmember(f"{id}.jpg")
                img = tb.extractfile(fileinfo).read()
                img = Image.open(BytesIO(img))
                imgs.append(img)
            return tarball_id, imgs
    except FileNotFoundError:
        print(f"Tarball with id {tarball_id} not found at {tarball}")
        return tarball_id, []
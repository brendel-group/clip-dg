from math import ceil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
from .data import (
    get_imgs_from_ids,
    get_imgs_from_ids_folder,
    load_if_file,
)
from PIL import Image
from torchvision.transforms.functional import resize, center_crop


def plot_imgs(
    imgs,  # List of images or path to file containing images
    out_file,  # Output file name for the plot
    n_cols=25,  # Number of columns in the plot grid
    img_size=4.0,  # Size of each image in inches
    **kwargs  # Additional arguments
):
    """
    Plots images in a grid.

    Args:
        imgs: List of images or path to file containing images.
        out_file: Output file name for the plot.
        n_cols: Number of columns in the plot grid.
        img_size: Size of each image in inches.
        **kwargs: Additional arguments.

    Returns:
        None
    """
    imgs = load_if_file(imgs)  # Load images if provided as file path

    n_rows = ceil(len(imgs) / n_cols)  # Calculate number of rows required
    fig, axes = plt.subplots(
        ncols=n_cols, nrows=n_rows, figsize=(img_size * n_cols, img_size * n_rows)
    )  # Create subplots grid
    ax = axes.flatten()

    for i, img in enumerate(imgs):
        # Convert greyscale images to RGB
        if img.mode == 'L':
            img = img.convert('RGB')

        # Center crop and resize
        img = np.array(center_crop(resize(img, 255), 224))
        img = np.transpose(img, (1, 2, 0)) if img.shape[0] in [1, 3] else img
        if (len(img.shape) < 3) or (1 in img.shape):
            ax[i].imshow(img, cmap='gray')
        else:
            ax[i].imshow(img)

    # Turn off grid
    [a.axis("off") for a in ax]

    # Remove whitespace
    fig.subplots_adjust(wspace=0, hspace=0)

    # Save the plot
    if not isinstance(out_file, Path):
        out_file = Path(out_file)
    plt.savefig(out_file, bbox_inches="tight")


def get_and_plot_images(ids, dataset_dir, save_dir, out_file, dataset):
    """
    Fetches images based on the dataset and plots them.

    Args:
        ids: IDs of images to plot.
        dataset_dir: Directory where images are stored.
        out_file: Output file name for the plot.
        dataset: Name of the dataset.

    Returns:
        None
    """
    if 'laion' in dataset:
        imgs = get_imgs_from_ids(ids, dataset_dir)  # Fetch images from IDs
    else:
        imgs = get_imgs_from_ids_folder(ids, dataset_dir)  # Fetch images from folder
    plot_imgs(imgs, save_dir+'/'+out_file)  # Plot fetched images


def visualize(
    dataset,
    dataset_dir,
    ids,
    save_dir,
    logits=None,
    preds=None,
    sample='random',
    n_imgs=1000,
    **kwargs
):
    """
    Visualizes images based on given inputs.

    Args:
        ids: IDs of images or path to file containing IDs.
        dataset_dir: Directory where images are stored.
        dataset: Name of the dataset.
        logits: Model logits for each image or path to file containing logits.
        preds: Predicted classes for each image.
        sample: Sample strategy for visualization.
        n_imgs: Number of images to visualize.
        **kwargs: Additional arguments.

    Returns:
        None
    """
    # Ensure ids and logits are loaded properly
    ids = load_if_file(ids)
    logits = load_if_file(logits)
    preds = load_if_file(preds)
    # Select a random seed and shuffle ids for random selection
    random_number = np.random.randint(1, 100)
    np.random.seed(random_number)

    if (logits is None) and (preds is None):
        np.random.shuffle(ids)
        ids_plot = ids[:n_imgs]
        # if preds and logits are none, then we are creating random images from a dataset. create directory to save
        os.makedirs(save_dir, exist_ok=True)

    elif preds is not None:
        ids_plot = {}
        logits_pred = {}
        for pred in np.unique(preds):
            print(f"pred = {pred}")
            preds_locs = np.where(preds == pred)[0]
            print(f"predlocs = {preds_locs}")
            ids_plot[pred] = ids[preds_locs]
            if (logits is not None) and (sample != 'random'):
                # select best/worst images ids to plot
                logits_pred[pred] = logits[preds_locs]
                logits_pred[pred] = logits_pred[pred] if sample == 'worst' else -logits_pred[pred]
                idcs_sorted = np.argsort(logits_pred[pred])
                ids_plot[pred] = ids_plot[pred][idcs_sorted][:n_imgs]
            else:
                np.random.shuffle(ids_plot[pred])
                ids_plot[pred] = ids_plot[pred][:n_imgs]
    else:
        raise ValueError

    if isinstance(ids_plot, dict):
        # Plot images for each class if ids are grouped by class
        for key in ids_plot.keys():
            out_file = f"{dataset}_{key}_sample={sample}_seed={random_number}.png"
            get_and_plot_images(ids_plot[key], dataset_dir, save_dir, out_file, dataset)
    else:
        # Plot images if ids are not grouped by class
        out_file = f"{dataset}_sample={sample}_seed={random_number}.png"
        get_and_plot_images(ids_plot, dataset_dir, save_dir, out_file, dataset)
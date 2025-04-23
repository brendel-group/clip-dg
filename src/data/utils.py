import os
import io
import tarfile
from torchvision.transforms.functional import resize, center_crop, to_pil_image
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
from PIL import Image


# dataset director dict
dataset_directories = {
    'imagenet-train': '/mnt/qb/datasets/ImageNet2012/train/',
    'imagenet-val': '/mnt/qb/datasets/ImageNet2012/val/',
    'imagenet-200': '/mnt/qb/work/bethge/pmayilvahanan31/datasets/imagenet_val_for_imagenet_r/',
    'imagenet-r': '/mnt/qb/datasets/imagenet-r/',
    'imagenet-sketch': '/mnt/qb/work/bethge/mwolff70/datasets/imagenet-sketch/',
    'imagenet-v2': '/mnt/qb/work/bethge/pmayilvahanan31/datasets/imagenetv2/',
    'laion400m': '/mnt/qb/datasets/laion400m/laion400m-data/',
    'imagenet-a': '/mnt/qb/work/bethge/pmayilvahanan31/datasets/imagenet-a/',
    'objectnet-subsample': '/mnt/qb/work/bethge/pmayilvahanan31/datasets/objectnet_subsample/',
    'celeba': '/mnt/qb/work/bethge/pmayilvahanan31/datasets/objectnet_subsample/',
    'MNIST': '/mnt/qb/work/bethge/pmayilvahanan31/datasets/',
    'SVHN': '/mnt/qb/work/bethge/pmayilvahanan31/datasets/',
    'val_sketch_style': '/mnt/qb/work/bethge/pmayilvahanan31/datasets/val_sketch_style/'
}


# Load image
def load_img(path, dataset, transform=None):
    """
    Given a path number, returns the image.

    Parameters:
    - path (int): Path number of the image.
    - dataset (str): Name of the dataset ('laion400m' or any dataset supported by torchvision.ImageFolder).
    - transform (torchvision.transforms.Compose or None, optional): Image transformations to be applied to the image.
      Defaults to None.

    Returns:
    - np.ndarray: The loaded image as a NumPy array.

    Raises:
    - AssertionError: If the loaded image does not have the expected shape (224, 224, 3) for RGB images.
    """

    # Get the directory for the specified dataset
    dir = dataset_directories[dataset]

    if dataset == 'laion400m':
        # For 'laion400m' dataset, load image from tar file
        path = f'{path:09d}'

        # Open the tar file and extract the image
        with tarfile.open(os.path.join(dir, f'{path[:5]}.tar')) as tf:
            fileinfo = tf.getmember(f'{path}.jpg')
            img = tf.extractfile(fileinfo).read()

            # Open the image using PIL
            img = Image.open(io.BytesIO(img))

        # Convert greyscale images to RGB
        if img.mode == 'L':
            img = img.convert('RGB')

        # Apply center crop to the image
        if transform is None:
            img = np.array(center_crop(resize(img, 255), 224))
        else:
            img = transform(img)

    else:
        # For other datasets, load image using torchvision.ImageFolder
        if transform is None:
            # If no transform is provided, use a standard transform
            transform_standard = transforms.Compose(
                [transforms.ToTensor()]
            )

            # Load the image and convert it to a NumPy array with center crop
            dataset = datasets.ImageFolder(dir, transform=transform_standard)
            img = dataset[path][0]
            img = np.array(center_crop(resize(img, 255), 224)).transpose(1, 2, 0)
        else:
            # Load the image using the provided transform
            dataset = datasets.ImageFolder(dir, transform=transform)
            img = dataset[path][0]

    # Check if the loaded image has the expected shape (224, 224, 3)
    assert img.ndim == 3, f"Image {path} is expected to be of shape (224, 224, 3), but has shape {img.shape}"

    return img

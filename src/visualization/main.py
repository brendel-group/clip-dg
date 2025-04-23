# Import necessary libraries
import argparse
from .visualize_imgs import visualize  # Assuming this is a custom function for visualizing images
from pathlib import Path
import os
import yaml
import numpy as np
import time


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=str, help="Path to the file containing image IDs")
    parser.add_argument("--cluster", type=str, default="")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Path to the logits file (optional, for selecting best images)")
    parser.add_argument("--directory", type=str, default="",
                        help="Path to the results dir of style classifier")
    parser.add_argument("--sample", type=str, default="random", choices=["random", "worst", "best"],
                        help="Sampling method for selecting images: 'random' or 'kmeans'")
    parser.add_argument("--preds", type=str, default=None,
                        help="Path to the predictions file (optional, for selecting best images)")
    parser.add_argument("--use_preds", type=boolean_string, default=True,
                        help="Path to the predictions file (optional, for selecting best images)")
    parser.add_argument("--n_imgs", type=int, default=1000, help="Number of images to visualize")
    args = parser.parse_args()

    # Build the path to the data using relative paths and command-line arguments
    current_script_dir = Path(__file__).parent  # Get the directory of the script
    paths_folder = current_script_dir / "../paths"  # Path to the "paths" folder
    paths_file_name = f"paths_{args.cluster}.yaml"  # Construct the paths file name with cluster
    paths_file_path = paths_folder / paths_file_name  # Combine paths for the full path

    # Load the data paths from the YAML file
    with open(paths_file_path, "r") as ymlfile:
        paths = yaml.safe_load(ymlfile)
    paths = paths['datasets']

    # paths of logits, preds, ids
    if args.use_preds:
        args.preds = os.path.join(args.directory, args.eval_data, 'preds.npy')
        args.logits = os.path.join(args.directory, args.eval_data, 'logits.npy')
    else:
        args.preds = None
        args.logits = None

    args.ids = os.path.join(args.directory, args.eval_data, 'meta.npy')
    print(args)
    # make edits to datapath if laion
    data_path = paths[args.eval_data] if 'laion' not in args.eval_data else paths['laion400m'].split('{')[0] # laion200m exception

    start_time = time.time()
    # Call the visualization function with the parsed arguments and loaded data paths
    visualize(
        args.eval_data,
        data_path,  # Access dataset path using loaded `paths` dictionary
        args.ids,
        args.directory+'/'+args.eval_data,
        logits=args.logits,
        preds=args.preds,
        sample=args.sample,
        n_imgs=args.n_imgs,
    )
    print(f"plotting {args.n_imgs} {args.sample} {args.eval_data} images took {time.time()-start_time}")


if __name__ == "__main__":
    main()
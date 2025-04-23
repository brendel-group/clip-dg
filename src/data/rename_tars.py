import os
from argparse import ArgumentParser


def main(args):
    dir_loc = args.out_dir
    file_list = os.listdir(dir_loc)
    file_list.sort()
    i = 0
    for file in file_list:
        file_name = f"{i:05}"
        os.rename(dir_loc+file, dir_loc+file_name+'.tar')
        i += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--out_dir',
        help='output folder')
    args = parser.parse_args()
    main(args)
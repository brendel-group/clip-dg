from time import time
import tarfile
import numpy as np
import sys
import os

# # with 'art'
# word_set = {'vignette', 'depict', 'sticker', 'grafitti', 'chalk_out', 'cast', 'scribble', 'carving',
#             'nontextual-matter', 'pictorial', 'engrave', 'toon', 'draw', 'tracing', 'videogame', 'draft',
#             'trace', 'carve', 'chalk-out', 'doodling', 'picture', 'chalk out', 'embellishment', 'fancywork',
#             'miniature', 'gummed label', 'in-writing', 'nontextual_matter', 'tattoo', 'gummed_label',
#             'embroidery', 'origami', 'gummed-label', 'paint', 'engraving', 'doodle', 'graphic',
#             'in_writing', 'sculpt', 'sketch', 'art', 'in writing', 'nontextual matter', 'toy'}


# with ' art'
word_set = {'vignette', 'depict', 'sticker', 'grafitti', 'chalk_out', 'cast', 'scribble', 'carving',
            'nontextual-matter', 'pictorial', 'engrave', 'toon', 'draw', 'tracing', 'videogame', 'draft',
            'trace', 'carve', 'chalk-out', 'doodling', 'picture', 'chalk out', 'embellishment', 'fancywork',
            'miniature', 'gummed label', 'in-writing', 'nontextual_matter', 'tattoo', 'gummed_label',
            'embroidery', 'origami', 'gummed-label', 'paint', 'engraving', 'doodle', 'graphic',
            'in_writing', 'sculpt', 'sketch', ' art', 'in writing', 'nontextual matter', 'toy'}


def mark_natural_or_sketch(tar_file, word_set):
    meta = []
    labels = []
    with tarfile.open(tar_file, 'r') as tar:
        for member in tar.getmembers():
            if '.txt' in member.name:
                meta.append(member.name.split('.txt')[0])
                txt = tar.extractfile(member).read().decode('utf-8')
                txt = txt.lower()

                label = 0
                for word in word_set:
                    if word in txt:
                        label = 1
                        break
                labels.append(label)

    return meta, labels


# Main script
num = int(sys.argv[1])
save_dir = '/is/cluster/fast/pmayilvahanan/clip_ood_part2/text_filtering_2/'
base_dir = '/is/cluster/fast/pmayilvahanan/datasets/200M/'
tarball_id = f"{num:05d}"
tarball = base_dir + f"{tarball_id}.tar"
meta, labels = mark_natural_or_sketch(tarball, word_set)

os.makedirs(save_dir, exist_ok=True)
np.save(save_dir+'meta_'+str(num)+'.npy', meta)
np.save(save_dir+'labels_'+str(num)+'.npy', labels)
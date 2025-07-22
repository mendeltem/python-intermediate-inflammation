import os
import numpy as np

def get_feature_paths(start_dir, extensions = ['nii','gz']):
    """Returns all image paths with the given extensions in the directory.
    Arguments:
        start_dir: directory the search starts from.
        extensions: extensions of image file to be recognized.
    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    img_paths = []
    for roots,dirs,files in os.walk(start_dir):
        for name in files:
            for e in extensions:
                if name.endswith('.' + e):
                    img_paths.append(roots + '/' + name)
    img_paths.sort()
    return img_paths  


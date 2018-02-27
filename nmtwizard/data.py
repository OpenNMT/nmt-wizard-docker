"""Data utilities."""

import os
import shutil


def merge_files(files, output):
    """Merge all files in output."""
    with open(output, 'wb') as output_file:
        for f in files:
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, output_file, 1024 * 1024 * 10)  # Chunk of 10MB.

def merge_files_in_directory(input_dir, output_dir, src_suffix, tgt_suffix):
    """Merge all source and target files in the directory input_dir to a single
    parallel file in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    src_files = [os.path.join(input_dir, f) for f in files if f.endswith(src_suffix)]
    tgt_files = [os.path.join(input_dir, f) for f in files if f.endswith(tgt_suffix)]
    merge_files(src_files, os.path.join(output_dir, 'train.%s' % src_suffix))
    merge_files(tgt_files, os.path.join(output_dir, 'train.%s' % tgt_suffix))

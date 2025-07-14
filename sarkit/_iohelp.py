"""
Common IO Helper functionality
"""

import numpy as np

BUFFER_SIZE = 2**27


def fromfile(file_obj, dtype, count):
    values = np.empty((count,), dtype)
    max_read_items = max(BUFFER_SIZE // dtype.itemsize, 1)
    num_to_read = count
    num_already_read = 0
    while num_to_read > 0:
        read_count = min(max_read_items, num_to_read)
        nbytes_requested = read_count * dtype.itemsize
        array = file_obj.read(nbytes_requested)
        nbytes_read = len(array)
        if nbytes_read != nbytes_requested:
            raise RuntimeError(f"Expected {nbytes_requested=}; only read {nbytes_read}")
        values[num_already_read : num_already_read + read_count] = np.frombuffer(
            array, dtype
        )
        num_already_read += read_count
        num_to_read -= read_count
    return values

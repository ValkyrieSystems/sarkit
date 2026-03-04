"""
Common IO Helper functionality
"""

import numpy as np

MAX_READ_SIZE = 2**27


def fromfile(file_obj, dtype, count, out=None):
    # Read as bytes because some io libraries (eg smart_open) don't always like buffers with uncommon dtypes
    if out is not None:
        out = out.view(np.uint8)
    values = ensure_array(out, (count * dtype.itemsize,), np.uint8)

    bytes_remaining = values.nbytes
    num_already_read = 0

    # read chunks because some libraries buffer inside readinto()
    while bytes_remaining:
        nbytes_requested = min(bytes_remaining, MAX_READ_SIZE)
        buff = values[num_already_read : num_already_read + nbytes_requested].data

        nbytes_read = file_obj.readinto(buff)
        if nbytes_read != nbytes_requested:
            raise RuntimeError(f"Expected {nbytes_requested=}; only read {nbytes_read}")

        num_already_read += nbytes_requested
        bytes_remaining -= nbytes_requested

    return values.view(dtype)


def ensure_array(array, shape, dtype):
    if array is None:
        array = np.empty(shape, dtype)
    else:
        if array.shape != tuple(shape):
            raise ValueError(f"array must have shape {shape}, not {array.shape}")
        if array.dtype != dtype:
            raise ValueError(f"array must have dtype '{dtype}', not '{array.dtype}'")
    return array

import numpy as np
import pytest

import sarkit._iohelp as skio


@pytest.mark.parametrize("dtype", (np.dtype(np.float64), np.dtype(np.int64)))
def test_compare_fromfile(tmp_path, dtype):
    temp_file = tmp_path / "temp.bin"
    num_items = 2.5 * skio.BUFFER_SIZE // dtype.itemsize
    cube_root = int(num_items ** (1 / 3))
    shape = (cube_root // 2, cube_root, 2 * cube_root)

    rng = np.random.default_rng()
    values = (rng.random(shape) * 1000).astype(dtype)
    with open(temp_file, "wb") as f:
        values.tofile(f)
    with open(temp_file, "rb") as f:
        np_arr = np.fromfile(f, dtype, np.prod(shape)).reshape(shape)
    with open(temp_file, "rb") as f:
        sk_arr = skio.fromfile(f, dtype, np.prod(shape)).reshape(shape)
    assert np.array_equal(np_arr, sk_arr)

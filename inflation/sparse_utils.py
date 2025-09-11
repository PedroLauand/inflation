from pyroaring import BitMap
import numpy as np
from typing import Iterator, Tuple, Union

class Sparse2DBitArray:
    """
    A 2D sparse bit array implementation using pyroaring BitMaps for each row.
    This class is designed to be a memory-efficient replacement for a 2D NumPy
    boolean array where the rows are sparse. It exposes a NumPy-compatible
    interface for reading data.
    """
    def __init__(self, num_cols: int):
        """
        Initializes the Sparse2DBitArray.

        :param num_cols: The total number of columns in the conceptual 2D array.
                         If not provided, it's inferred from the max index in data.
        """

        self.num_cols = num_cols
        self.num_rows = 0
        self._bm = BitMap()

    @property
    def size(self) -> int:
        return self.num_rows * self.num_cols


    @property
    def shape(self):
        """Returns the shape of the array as a tuple (num_rows, num_cols)."""
        return (self.num_rows, self.num_cols)

    def __len__(self):
        """Returns the number of rows."""
        return self.num_rows

    def __iter__(self):
        """Returns an iterator over the rows, yielding dense NumPy arrays."""
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key: Union[int,
                                     Tuple[int, int],
                                     np.ndarray,
                                     list]) -> Union[np.ndarray, bool]:
        """Row access (int) → 1D dense boolean row.
           Element access (row, col) → bool.
        """
        nrows, ncols = self.shape
        if isinstance(key, int):
            # Return entire row as dense boolean array
            row = key
            if not (0 <= row < nrows):
                raise IndexError("Row index out of range")
            row_arr = np.zeros(ncols, dtype=np.bool_)
            start = row * ncols
            end = start + ncols
            for i in self._bm.iter_equal_or_larger(start):
                if i >= end:
                    break
                row_arr[i-start] = True
            return row_arr
        elif isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if not (0 <= row < nrows and 0 <= col < ncols):
                raise IndexError("Index out of range")
            flat = row * ncols + col
            return flat in self._bm
        elif isinstance(key, np.ndarray):
            assert key.ndim == 1, "Only 1D arrays are supported for indexing 2D BitMaps"
            arr_2D = np.zeros((key.shape[0], ncols), dtype=np.bool_)
            for new_row, old_row in enumerate(key):
                start = old_row * ncols
                end = start + ncols
                for i in self._bm.iter_equal_or_larger(start):
                    if i >= end:
                        break
                    arr_2D[new_row, i - start] = True
            return arr_2D
        else:
            raise TypeError("Invalid index type")

    @classmethod
    def from_array(cls, data: np.ndarray) -> "Sparse2DBitArray":
        assert data.ndim == 2, "Data must be a 2D NumPy array"
        (nrows, ncols) = data.shape
        new_2D = cls(ncols)
        new_2D.num_rows = nrows
        new_2D._bm = BitMap(np.flatnonzero(data))
        return new_2D

    def extend(self, other: Union[np.ndarray, "Sparse2DBitArray"]) -> None:
        if isinstance(other, Sparse2DBitArray):
            assert self.num_cols == other.num_cols, "Cannot add Sparse2DBitArray with different number of columns"
            self.num_rows += other.num_rows
            self._bm.update(other._bm.shift(self.size))
        if isinstance(other, np.ndarray):
            self.extend(self.from_array(other))

    def __repr__(self):
        return f"Sparse2DBitArray(shape={self.shape})"

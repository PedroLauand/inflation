import pyroaring
import numpy as np

class Sparse2DBitArray:
    """
    A 2D sparse bit array implementation using pyroaring BitMaps for each row.
    This class is designed to be a memory-efficient replacement for a 2D NumPy
    boolean array where the rows are sparse. It exposes a NumPy-compatible
    interface for reading data.
    """
    def __init__(self, data=None, num_cols=0):
        """
        Initializes the Sparse2DBitArray.

        :param data: A list of lists/iterables of integers, where each inner list
                     contains the indices of the set bits (1s) for a row.
        :param num_cols: The total number of columns in the conceptual 2D array.
                         If not provided, it's inferred from the max index in data.
        """
        if data is None:
            self._rows = []
            self.num_cols = num_cols
        else:
            self._rows = [pyroaring.BitMap(row) for row in data]
            if self._rows and num_cols == 0:
                max_val = 0
                for r in self._rows:
                    if r:
                        max_val = max(max_val, max(r))
                self.num_cols = max_val + 1
            else:
                self.num_cols = num_cols

    @property
    def shape(self):
        """Returns the shape of the array as a tuple (num_rows, num_cols)."""
        return (len(self._rows), self.num_cols)

    def __len__(self):
        """Returns the number of rows."""
        return len(self._rows)

    def __iter__(self):
        """Returns an iterator over the rows, yielding dense NumPy arrays."""
        for i in range(len(self)):
            yield self[i]

    def _bitmap_to_numpy_row(self, bitmap):
        row = np.zeros(self.num_cols, dtype=bool)
        if len(bitmap) > 0:
            row[list(bitmap)] = True
        return row

    def __getitem__(self, index):
        """
        Supports indexing to retrieve rows as dense NumPy boolean arrays.
        - An integer index returns a single row.
        - A slice or list/array index returns a new NumPy array
          containing the selected rows.
        """
        if isinstance(index, int):
            return self._bitmap_to_numpy_row(self._rows[index])
        elif isinstance(index, slice):
            bitmaps = self._rows[index]
            arr = np.zeros((len(bitmaps), self.num_cols), dtype=bool)
            for i, bm in enumerate(bitmaps):
                if len(bm) > 0:
                    arr[i, list(bm)] = True
            return arr
        elif isinstance(index, (list, np.ndarray)):
            bitmaps = [self._rows[i] for i in index]
            arr = np.zeros((len(bitmaps), self.num_cols), dtype=bool)
            for i, bm in enumerate(bitmaps):
                if len(bm) > 0:
                    arr[i, list(bm)] = True
            return arr
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

    @classmethod
    def from_bitmaps(cls, bitmaps, num_cols):
        """Creates a Sparse2DBitArray from a list of existing BitMap objects."""
        instance = cls(num_cols=num_cols)
        instance._rows = list(bitmaps)
        return instance

    def __repr__(self):
        return f"Sparse2DBitArray(shape={self.shape}, rows={len(self._rows)})"

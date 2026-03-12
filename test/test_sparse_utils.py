import unittest

import numpy as np

from inflation.sparse_utils import Sparse2DBitArray


class TestSparse2DBitArray(unittest.TestCase):
    def test_extend_numpy_array_on_empty_container(self):
        data = np.array(
            [
                [True, False, False, False, False, False, False, False, False],
                [False, False, False, True, False, False, False, False, False],
                [False, False, False, False, False, False, True, False, False],
            ],
            dtype=bool,
        )
        sparse = Sparse2DBitArray(9)
        sparse.extend(data)

        self.assertEqual(sparse.shape, (3, 9))
        self.assertListEqual(np.flatnonzero(sparse[0]).tolist(), [0])
        self.assertListEqual(np.flatnonzero(sparse[1]).tolist(), [3])
        self.assertListEqual(np.flatnonzero(sparse[2]).tolist(), [6])

    def test_extend_multiple_numpy_appends_preserves_offsets(self):
        sparse = Sparse2DBitArray(9)
        sparse.extend(
            np.array(
                [
                    [True, False, False, False, False, False, False, False, False],
                    [False, False, False, True, False, False, False, False, False],
                ],
                dtype=bool,
            )
        )
        sparse.extend(
            np.array(
                [[False, False, False, False, False, False, True, False, False]],
                dtype=bool,
            )
        )

        self.assertEqual(sparse.shape, (3, 9))
        self.assertListEqual(np.flatnonzero(sparse[0]).tolist(), [0])
        self.assertListEqual(np.flatnonzero(sparse[1]).tolist(), [3])
        self.assertListEqual(np.flatnonzero(sparse[2]).tolist(), [6])

    def test_extend_sparse_array_preserves_high_column_bits(self):
        first = Sparse2DBitArray.from_array(
            np.array(
                [[False, False, False, False, False, False, True, True]],
                dtype=bool,
            )
        )
        second = Sparse2DBitArray.from_array(
            np.array(
                [[False, False, False, False, False, False, False, False],
                 [False, False, False, False, False, False, True, True]],
                dtype=bool,
            )
        )

        first.extend(second)

        self.assertEqual(first.shape, (3, 8))
        self.assertListEqual(np.flatnonzero(first[0]).tolist(), [6, 7])
        self.assertListEqual(np.flatnonzero(first[1]).tolist(), [])
        self.assertListEqual(np.flatnonzero(first[2]).tolist(), [6, 7])

    def test_extend_recovers_high_lex_indices_24_and_25(self):
        rows = np.zeros((3, 26), dtype=bool)
        rows[0, 24] = True
        rows[1, 25] = True
        rows[2, [24, 25]] = True
        sparse = Sparse2DBitArray(26)
        sparse.extend(rows)

        self.assertListEqual(np.flatnonzero(sparse[0]).tolist(), [24])
        self.assertListEqual(np.flatnonzero(sparse[1]).tolist(), [25])
        self.assertListEqual(np.flatnonzero(sparse[2]).tolist(), [24, 25])
if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import itertools
import logging
import math
from typing import TYPE_CHECKING

import anndata as ad
import h5py
import numpy as np
from anndata.abc import CSCDataset, CSRDataset
from scipy import sparse

if TYPE_CHECKING:
    from collections.abc import Iterator

Matrix = np.ndarray | sparse.spmatrix | sparse.sparray | ad.AnnData


class CellChunks:
    """The cells of a count matrix, read by chunks of rows.

    It takes a dense numpy array, a scipy sparse matrix or array, or an AnnData object
    (its ``.X``), in memory or backed. A chunk is handed out as a float32 CSR matrix,
    built from the rows of the chunk only, so the whole matrix is never copied. A CSC
    matrix in memory is converted to CSR once; a backed CSC matrix is refused, because
    rows cannot be read from it at a useful speed.

    The cells are split into chunks of equal size, at most ``chunk_size``, so that no
    small last chunk weighs too much in an online update. With shuffling, every chunk
    is a random set of cells, also for a backed matrix: a file is often sorted by
    sample, and chunks of consecutive cells would then each hold only a few samples.
    The rows of a chunk are read in increasing order.
    """

    def __init__(self, X: Matrix, chunk_size: int, binarize: bool = False) -> None:
        if chunk_size < 1:
            msg = f"chunk_size must be at least 1, not {chunk_size}."
            raise ValueError(msg)
        matrix = X.X if isinstance(X, ad.AnnData) else X
        if matrix is None:
            msg = "The AnnData object has no matrix in .X."
            raise ValueError(msg)
        if isinstance(matrix, CSCDataset):
            msg = (
                "The backed matrix is stored as CSC, from which rows cannot be read "
                "efficiently. Write it as CSR, or load it into memory."
            )
            raise ValueError(msg)
        if sparse.issparse(matrix) and matrix.format != "csr":
            logging.warning("Converting the %s matrix to CSR.", matrix.format)
            matrix = sparse.csr_matrix(matrix)
        if not isinstance(
            matrix, np.ndarray | CSRDataset | h5py.Dataset
        ) and not sparse.issparse(matrix):
            msg = (
                f"X format unrecognised: {type(X)} != np.ndarray, scipy sparse matrix "
                "or anndata.AnnData"
            )
            raise ValueError(msg)
        if len(matrix.shape) != 2:
            msg = f"X must be a 2D matrix, not of shape {matrix.shape}."
            raise ValueError(msg)

        self.matrix = matrix
        self.chunk_size = chunk_size
        self.binarize = binarize
        self.cell_totals = self._scan()

    @property
    def n_cells(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def total_counts(self) -> float:
        return float(self.cell_totals.sum())

    def _read(self, rows: slice | np.ndarray) -> sparse.csr_matrix:
        """The rows as a float32 CSR matrix that shares no memory with the input."""
        block = self.matrix[rows]
        block = sparse.csr_matrix(block, dtype=np.float32, copy=True)
        block.sum_duplicates()
        block.eliminate_zeros()
        if self.binarize:
            block.data[:] = 1.0
        return block

    def _scan(self) -> np.ndarray:
        """Checks every value once, and returns the counts of each cell."""
        totals = np.empty(self.n_cells, dtype=np.float64)
        for start in range(0, self.n_cells, self.chunk_size):
            stop = min(start + self.chunk_size, self.n_cells)
            block = self._read(slice(start, stop))
            if block.nnz and block.data.min() < 0:
                msg = "The count matrix has negative values."
                raise ValueError(msg)
            totals[start:stop] = np.asarray(block.sum(axis=1)).ravel()
        n_empty = int((totals == 0).sum())
        if n_empty:
            logging.warning(
                "%d cells have no counts; their topic proportions stay at the prior.",
                n_empty,
            )
        return totals

    @property
    def n_chunks(self) -> int:
        return math.ceil(self.n_cells / self.chunk_size)

    def chunks(
        self, rng: np.random.Generator | None = None
    ) -> Iterator[tuple[np.ndarray, sparse.csr_matrix]]:
        """The (cell indices, CSR matrix) of every chunk; shuffled if `rng` is given."""
        bounds = (
            np.linspace(0, self.n_cells, self.n_chunks + 1).round().astype(np.int64)
        )
        order = None if rng is None else rng.permutation(self.n_cells)
        for start, stop in itertools.pairwise(bounds):
            if order is None:
                yield np.arange(start, stop), self._read(slice(start, stop))
            else:
                rows = np.sort(order[start:stop])
                yield rows, self._read(rows)

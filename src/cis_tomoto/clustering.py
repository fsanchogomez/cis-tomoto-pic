from __future__ import annotations

import random
from enum import Enum

import igraph
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import sparse

CM_PER_INCH = 2.54
SEED = 42
EXACT_NEIGHBORS_LIMIT = 8192


class PlotFileFormat(str, Enum):
    png = "png"
    jpg = "jpg"
    svg = "svg"
    pdf = "pdf"


def nearest_neighbors(
    coordinates: np.ndarray, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    """The indices and distances of the `n_neighbors` nearest cells, each cell first.

    As `scanpy.pp.neighbors`: exact below EXACT_NEIGHBORS_LIMIT cells, and by
    pynndescent, with the trees and iterations of umap-learn, above it. The distance
    of a cell to itself is set to 0, as scanpy does, and not left at the rounding
    error of the exact search, which UMAP would take for its nearest neighbor.
    """
    n_cells = coordinates.shape[0]
    if n_neighbors > n_cells:
        n_neighbors = 1 + int(0.5 * n_cells)
    if n_cells < EXACT_NEIGHBORS_LIMIT:
        from sklearn.neighbors import KNeighborsTransformer  # noqa: PLC0415

        search = KNeighborsTransformer(
            algorithm="brute", n_jobs=1, n_neighbors=min(n_cells - 1, n_neighbors)
        )
    else:
        from pynndescent import PyNNDescentTransformer  # noqa: PLC0415

        search = PyNNDescentTransformer(
            n_neighbors=n_neighbors,
            metric="euclidean",
            random_state=SEED,
            n_jobs=1,
            n_trees=min(64, 5 + round(n_cells**0.5 / 20.0)),
            n_iters=max(5, round(np.log2(n_cells))),
        )
    distances = sparse.csr_matrix(search.fit_transform(coordinates))
    width = distances.getnnz(axis=1)
    if not np.all(width == width[0]):
        msg = "The neighbor search gave a different number of neighbors per cell."
        raise ValueError(msg)
    indices = distances.indices.reshape(n_cells, width[0])
    values = distances.data.reshape(n_cells, width[0])
    values[indices == np.arange(n_cells)[:, None]] = 0
    if not (indices[:, 0] == np.arange(n_cells)).any():
        indices = np.hstack([np.arange(n_cells)[:, None], indices])
        values = np.hstack([np.zeros((n_cells, 1)), values])
    return indices[:, :n_neighbors], values[:, :n_neighbors]


def neighbor_graph(coordinates: np.ndarray, n_neighbors: int) -> sparse.csr_matrix:
    """The UMAP connectivities of the cells, as `scanpy.pp.neighbors` computes them."""
    from umap.umap_ import fuzzy_simplicial_set  # noqa: PLC0415

    indices, distances = nearest_neighbors(coordinates, n_neighbors)
    graph, _, _ = fuzzy_simplicial_set(
        sparse.coo_matrix((coordinates.shape[0], 1)),
        indices.shape[1],
        None,
        None,
        knn_indices=indices,
        knn_dists=distances,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )
    return graph.tocsr()


class _IgraphRandom:
    """The random numbers that `scanpy.tl.leiden` gives igraph for a seed."""

    def __init__(self, seed: int) -> None:
        self._state = np.random.RandomState(seed)

    def getrandbits(self, k: int) -> int:
        return self._state.tomaxint() & ((1 << k) - 1)

    def randint(self, a: int, b: int) -> int:
        return self._state.randint(a, b + 1)

    def __getattr__(self, attr: str) -> object:
        return getattr(self._state, "normal" if attr == "gauss" else attr)


def umap_leiden(
    coordinates: np.ndarray, n_neighbors: int, resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    """The UMAP and the Leiden clusters of `scanpy.tl.umap` and `scanpy.tl.leiden`.

    Both read one neighbor graph, as in scanpy, and Leiden is its igraph flavor.
    """
    from umap.umap_ import find_ab_params, simplicial_set_embedding  # noqa: PLC0415

    graph = neighbor_graph(coordinates, n_neighbors)
    a, b = find_ab_params(1.0, 0.5)
    embedding, _ = simplicial_set_embedding(
        data=coordinates,
        graph=graph.tocoo(copy=True),
        n_components=2,
        initial_alpha=1.0,
        a=a,
        b=b,
        gamma=1.0,
        negative_sample_rate=5,
        n_epochs=500 if graph.shape[0] <= 10_000 else 200,
        init="spectral",
        random_state=np.random.RandomState(SEED),
        metric="euclidean",
        metric_kwds={},
        densmap=False,
        densmap_kwds={},
        output_dens=False,
    )

    sources, targets = graph.nonzero()
    network = igraph.Graph(
        n=graph.shape[0],
        edges=list(zip(sources, targets, strict=True)),
        edge_attrs={"weight": np.asarray(graph[sources, targets]).ravel().tolist()},
    )
    igraph.set_random_number_generator(_IgraphRandom(SEED))
    try:
        partition = network.community_leiden(
            objective_function="modularity",
            weights="weight",
            resolution=resolution,
            n_iterations=-1,
        )
    finally:
        igraph.set_random_number_generator(random)
    return np.asarray(embedding), np.asarray(partition.membership)


def leiden_categories(clusters: np.ndarray) -> pd.Categorical:
    return pd.Categorical(
        clusters.astype(str),
        categories=[str(cluster) for cluster in range(clusters.max() + 1)],
    )


def plot_umap(
    embedding: np.ndarray,
    clusters: np.ndarray,
    path: str,
    width: float,
    height: float,
    file_format: PlotFileFormat,
    dpi: int,
) -> None:
    figure = Figure(figsize=(width / CM_PER_INCH, height / CM_PER_INCH))
    axes = figure.subplots()
    colormap = mpl.colormaps["tab20"]
    axes.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colormap(clusters % colormap.N),
        s=min(120_000 / len(clusters), 100),
        linewidths=0,
    )
    for cluster in np.unique(clusters):
        x, y = np.median(embedding[clusters == cluster], axis=0)
        axes.text(x, y, str(cluster), ha="center", va="center", fontweight="bold")
    axes.set_title("Leiden clusters")
    axes.set_xlabel("UMAP1")
    axes.set_ylabel("UMAP2")
    axes.set_xticks([])
    axes.set_yticks([])
    figure.savefig(path, dpi=dpi, format=file_format.value)

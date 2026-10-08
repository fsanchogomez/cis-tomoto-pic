from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal, overload

import anndata as ad
import numpy as np

from cis_tomoto.cistopic import cistopic_region_names, require_pycistopic, to_cistopic
from cis_tomoto.clustering import SEED, leiden_categories, umap_leiden
from cis_tomoto.gibbs import DEFAULT_CHUNK_SIZE, DEFAULT_ITERATIONS, GibbsLDA

if TYPE_CHECKING:
    import os

    from pycisTopic.cistopic_class import CistopicObject

    from cis_tomoto.inputs import Matrix


def fit(X: Matrix, **params: object) -> tuple[GibbsLDA, np.ndarray, float]:
    """The model, the topic proportions of the cells and the time of the fit (s)."""
    start = time.perf_counter()
    model = GibbsLDA(**params)
    theta = model.fit_transform(X)
    return model, theta, time.perf_counter() - start


def parameter_value(value: object) -> object:
    if value is None or isinstance(value, str):
        return "symmetric" if value is None else value
    return str(value) if np.ndim(value) == 0 else np.asarray(value)


def lda_elements(model: GibbsLDA, theta: np.ndarray) -> dict[str, dict[str, object]]:
    """The results of the fit, by AnnData group."""
    return {
        "obsm": {"X_lda": theta},
        "varm": {"LDA_topics": np.ascontiguousarray(model.topics_.T)},
        "uns": {
            "LDA": {
                "method": "collapsed_gibbs",
                "params": {
                    "n_topics": model.n_topics,
                    "alpha": parameter_value(model.alpha),
                    "eta": parameter_value(model.eta),
                    "iterations": model.iterations,
                    "binarize": model.binarize,
                    "chunk_size": model.chunk_size,
                    "n_jobs": model.n_jobs,
                },
                "alpha": model.alpha_,
                "eta": model.eta_,
                "log_likelihood": np.asarray(model.log_likelihood_),
            }
        },
    }


@overload
def cis_lda(
    data: ad.AnnData | str | os.PathLike[str],
    n_topics: int = ...,
    *,
    return_cistopic: Literal[False] = ...,
    **kwargs: object,
) -> ad.AnnData: ...


@overload
def cis_lda(
    data: ad.AnnData | str | os.PathLike[str],
    n_topics: int = ...,
    *,
    return_cistopic: Literal[True],
    **kwargs: object,
) -> CistopicObject: ...


def cis_lda(
    data: ad.AnnData | str | os.PathLike[str],
    n_topics: int = 20,
    *,
    alpha: str | float | np.ndarray = "symmetric",
    eta: str | float | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    binarize: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    n_jobs: int = 1,
    random_state: int = SEED,
    umap: bool = False,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    return_cistopic: bool = False,
    project: str = "cisTopic",
) -> ad.AnnData | CistopicObject:
    """Finds the topics of the cells of `data` with LDA, fitted by collapsed Gibbs
    sampling (tomotopy).

    Parameters
    ----------
    data
        An AnnData object, in memory or backed, or the path of an .h5ad file, with the
        counts in ``.X``, cells in rows and features in columns.
    n_topics
        Number of topics.
    alpha
        Prior of the topic proportions of a cell: "symmetric" (1/n_topics),
        "asymmetric", "auto" (learned every 10 iterations), a number or an array of
        n_topics values. pycisTopic uses 50/n_topics.
    eta
        Prior of the feature weights of a topic: "symmetric" or None (1/n_topics), or a
        number. pycisTopic uses 0.1.
    iterations
        Number of Gibbs sampling iterations over all the tokens.
    binarize
        Count every non-zero entry as 1, as pycisTopic does. Without it, the counts
        must be integers.
    chunk_size
        Number of cells read from the input at a time.
    n_jobs
        Number of sampling threads. The result of a seed depends on it.
    random_state
        Seed of the sampler.
    umap
        Also compute a UMAP and Leiden clusters of the cells, from the square root of
        their topic proportions.
    n_neighbors
        Number of nearest neighbors of the UMAP and the clusters.
    resolution
        Resolution of the Leiden clusters.
    return_cistopic
        Return a pycisTopic CistopicObject instead of an AnnData object. It needs
        pycisTopic, and features named as regions (chrom:start-end or
        chrom_start_end).
    project
        Project name of the CistopicObject.

    Returns
    -------
    Without `return_cistopic`, the AnnData object with:

    * ``obsm["X_lda"]``: the topic proportions of the cells.
    * ``varm["LDA_topics"]``: the feature distribution of the topics.
    * ``uns["LDA"]``: the parameters of the fit, the priors and the log-likelihood
      per token every 10 iterations.
    * with `umap`, ``obsm["X_umap"]`` and ``obs["leiden"]``.

    An AnnData object given as `data` gets the results and is returned; a path is read
    into memory. With `return_cistopic`, a CistopicObject of the counts, with the fit
    as its selected model, and with `umap`, ``projections["cell"]["UMAP"]`` and
    ``cell_data["leiden"]``.
    """
    adata = data if isinstance(data, ad.AnnData) else ad.read_h5ad(data)
    if return_cistopic:
        require_pycistopic()
        cistopic_region_names(adata.var_names)

    model, theta, elapsed = fit(
        adata,
        n_topics=n_topics,
        alpha=alpha,
        eta=eta,
        iterations=iterations,
        binarize=binarize,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    embedding = clusters = None
    if umap:
        embedding, labels = umap_leiden(np.sqrt(theta), n_neighbors, resolution)
        clusters = leiden_categories(labels)

    if return_cistopic:
        return to_cistopic(adata, model, theta, project, elapsed, embedding, clusters)

    for group, values in lda_elements(model, theta).items():
        for key, value in values.items():
            getattr(adata, group)[key] = value
    if umap:
        adata.obsm["X_umap"] = embedding
        adata.obs["leiden"] = clusters
    return adata

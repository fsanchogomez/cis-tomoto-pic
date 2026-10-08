from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pytest
from cis_tomoto import GibbsLDA, cis_lda
from scipy import sparse

if TYPE_CHECKING:
    from pathlib import Path

N_CELLS = 40
N_FEATURES = 12
N_TOPICS = 3


def counts() -> sparse.csr_matrix:
    return sparse.csr_matrix(
        np.random.default_rng(0).poisson(3.0, size=(N_CELLS, N_FEATURES)),
        dtype=np.float32,
    )


def anndata() -> ad.AnnData:
    adata = ad.AnnData(counts())
    adata.obs_names = [f"cell{index}" for index in range(N_CELLS)]
    adata.var_names = [
        f"chr1_{index * 100}_{index * 100 + 100}" for index in range(N_FEATURES)
    ]
    return adata


def test_cis_lda_adds_the_results_to_the_anndata() -> None:
    adata = anndata()

    result = cis_lda(adata, N_TOPICS, iterations=20)

    assert result is adata
    model = GibbsLDA(N_TOPICS, iterations=20)
    theta = model.fit_transform(counts())
    np.testing.assert_array_equal(adata.obsm["X_lda"], theta)
    np.testing.assert_array_equal(adata.varm["LDA_topics"], model.topics_.T)
    report = adata.uns["LDA"]
    assert report["method"] == "collapsed_gibbs"
    assert report["params"]["alpha"] == "symmetric"
    assert report["params"]["eta"] == "symmetric"
    assert report["params"]["iterations"] == 20
    assert report["eta"] == pytest.approx(1 / N_TOPICS)
    assert len(report["log_likelihood"]) == 2
    assert "X_umap" not in adata.obsm


def test_cis_lda_reads_a_path_and_a_backed_file(tmp_path: Path) -> None:
    path = tmp_path / "counts.h5ad"
    anndata().write_h5ad(path)
    expected = cis_lda(anndata(), N_TOPICS, iterations=10).obsm["X_lda"]

    from_path = cis_lda(path, N_TOPICS, iterations=10)
    backed = cis_lda(ad.read_h5ad(path, backed="r"), N_TOPICS, iterations=10)

    np.testing.assert_array_equal(from_path.obsm["X_lda"], expected)
    np.testing.assert_array_equal(backed.obsm["X_lda"], expected)


def test_cis_lda_stores_numbers_and_arrays_of_the_priors() -> None:
    adata = cis_lda(anndata(), N_TOPICS, alpha=2.5, eta=0.1, iterations=10)
    params = adata.uns["LDA"]["params"]
    assert params["alpha"] == "2.5"
    assert params["eta"] == "0.1"

    alpha = np.array([0.1, 0.2, 0.3])
    adata = cis_lda(anndata(), N_TOPICS, alpha=alpha, iterations=10)
    np.testing.assert_array_equal(adata.uns["LDA"]["params"]["alpha"], alpha)


def test_cis_lda_adds_the_umap_and_leiden_clusters() -> None:
    adata = cis_lda(anndata(), N_TOPICS, iterations=10, umap=True, n_neighbors=5)

    assert adata.obsm["X_umap"].shape == (N_CELLS, 2)
    assert adata.obs["leiden"].dtype == "category"


def test_cis_lda_refuses_non_integer_counts() -> None:
    adata = anndata()
    adata.X = adata.X * 0.5

    with pytest.raises(ValueError, match="integer counts"):
        cis_lda(adata, N_TOPICS, iterations=1)

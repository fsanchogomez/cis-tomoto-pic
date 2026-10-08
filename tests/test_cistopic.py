from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from cis_tomoto import GibbsLDA, cis_lda
from cis_tomoto.cistopic import cistopic_region_names
from scipy import sparse

N_CELLS = 40
N_FEATURES = 12
N_TOPICS = 3


def test_region_names_accept_pycistopic_and_sincei_names() -> None:
    names = pd.Index(["chr1:0-100", "chr1_100_200", "chrUn_KI270302v1_0_50::GENE"])

    assert cistopic_region_names(names) == [
        "chr1:0-100",
        "chr1:100-200",
        "chrUn_KI270302v1:0-50",
    ]


def test_region_names_refuse_other_names() -> None:
    with pytest.raises(ValueError, match="not a region"):
        cistopic_region_names(pd.Index(["GAPDH"]))


def test_region_names_refuse_duplicate_regions() -> None:
    with pytest.raises(ValueError, match="same region"):
        cistopic_region_names(pd.Index(["chr1_0_100::A", "chr1_0_100::B"]))


def counts() -> sparse.csr_matrix:
    X = np.random.default_rng(0).poisson(3.0, size=(N_CELLS, N_FEATURES))
    X[:, -1] = 0
    return sparse.csr_matrix(X, dtype=np.float32)


def anndata() -> ad.AnnData:
    adata = ad.AnnData(counts())
    adata.obs_names = [f"cell{index}" for index in range(N_CELLS)]
    adata.var_names = [
        f"chr1_{index * 100}_{index * 100 + 100}" for index in range(N_FEATURES)
    ]
    adata.obs["batch"] = ["a", "b"] * (N_CELLS // 2)
    return adata


def test_cis_lda_returns_a_cistopic_object() -> None:
    pytest.importorskip("pycisTopic")
    from pycisTopic.cistopic_class import CistopicObject  # noqa: PLC0415
    from pycisTopic.lda_models import CistopicLDAModel  # noqa: PLC0415

    cistopic = cis_lda(
        anndata(), N_TOPICS, iterations=20, binarize=True, return_cistopic=True
    )

    assert isinstance(cistopic, CistopicObject)
    cell_names = [f"cell{index}" for index in range(N_CELLS)]
    assert cistopic.cell_names == cell_names
    assert len(cistopic.region_names) == N_FEATURES - 1
    assert cistopic.region_names[0] == "chr1:0-100"
    assert cistopic.region_data["var_name"].iloc[0] == "chr1_0_100"
    assert cistopic.cell_data["batch"].tolist() == ["a", "b"] * (N_CELLS // 2)
    assert (cistopic.fragment_matrix != counts().T[: N_FEATURES - 1]).nnz == 0

    model = GibbsLDA(N_TOPICS, iterations=20, binarize=True)
    theta = model.fit_transform(counts())
    lda = cistopic.selected_model
    assert isinstance(lda, CistopicLDAModel)
    assert lda.cell_topic.index.tolist() == ["Topic1", "Topic2", "Topic3"]
    np.testing.assert_array_equal(lda.cell_topic.to_numpy(), theta.T)
    topics = model.topics_[:, : N_FEATURES - 1]
    assert lda.topic_region.index.tolist() == cistopic.region_names
    np.testing.assert_allclose(
        lda.topic_region.to_numpy(),
        (topics / topics.sum(axis=1, keepdims=True)).T,
        rtol=1e-6,
    )
    assert lda.topic_ass["Assignments"].sum() == (counts() > 0).sum()
    assert lda.metrics.columns.tolist() == [
        "Arun_2010",
        "Cao_Juan_2009",
        "Mimno_2011",
        "loglikelihood",
    ]
    assert np.isfinite(lda.metrics.to_numpy(dtype=float)).all()
    assert lda.marg_topic["Marg_Topic"].sum() == pytest.approx(1.0)
    assert lda.parameters.loc["package", "Parameter"] == "tomotopy"
    assert lda.parameters.loc["eta", "Parameter"] == pytest.approx(1 / N_TOPICS)


def test_cis_lda_adds_the_umap_to_the_cistopic_object() -> None:
    pytest.importorskip("pycisTopic")

    cistopic = cis_lda(
        anndata(),
        N_TOPICS,
        iterations=10,
        umap=True,
        n_neighbors=5,
        return_cistopic=True,
    )

    umap = cistopic.projections["cell"]["UMAP"]
    assert umap.columns.tolist() == ["UMAP_1", "UMAP_2"]
    assert umap.index.tolist() == cistopic.cell_names
    assert cistopic.cell_data["leiden"].notna().all()


def test_cis_lda_checks_the_region_names_before_the_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pycisTopic")
    adata = anndata()
    adata.var_names = [f"gene{index}" for index in range(N_FEATURES)]

    def no_fit(*_: object, **__: object) -> None:
        raise AssertionError

    monkeypatch.setattr(GibbsLDA, "fit_transform", no_fit)
    with pytest.raises(ValueError, match="not a region"):
        cis_lda(adata, N_TOPICS, return_cistopic=True)

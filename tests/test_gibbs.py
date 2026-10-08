from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pytest
from cis_tomoto import GibbsLDA
from scipy import sparse
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

N_CELLS = 300
N_FEATURES = 80
N_TOPICS = 4


def corpus(seed: int = 0) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Counts drawn from an LDA with well separated topics, and the topics."""
    rng = np.random.default_rng(seed)
    topics = rng.dirichlet(np.full(N_FEATURES, 0.1), N_TOPICS)
    theta = rng.dirichlet(np.full(N_TOPICS, 0.3), N_CELLS)
    X = np.vstack([
        rng.multinomial(rng.integers(50, 200), theta[cell] @ topics)
        for cell in range(N_CELLS)
    ])
    return sparse.csr_matrix(X.astype(np.float32)), topics


def test_the_fit_recovers_the_topics() -> None:
    X, topics = corpus()

    model = GibbsLDA(N_TOPICS, iterations=200)
    theta = model.fit_transform(X)

    distance = np.abs(model.topics_[:, None] - topics[None]).sum(axis=2)
    rows, columns = linear_sum_assignment(distance)
    assert distance[rows, columns].max() < 0.15
    assert theta.shape == (N_CELLS, N_TOPICS)
    np.testing.assert_allclose(theta.sum(axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(model.topics_.sum(axis=1), 1.0, rtol=1e-5)


def test_theta_is_the_last_sample_with_the_prior() -> None:
    X, _ = corpus()
    model = GibbsLDA(N_TOPICS, alpha=0.5, iterations=20)

    theta = model.fit_transform(X)

    totals = np.asarray(X.sum(axis=1)).ravel()
    smallest = (0.5 / (totals + N_TOPICS * 0.5))[:, None]
    assert np.all(theta >= smallest - 1e-6)
    counts = theta * (totals + N_TOPICS * 0.5)[:, None] - 0.5
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-2)


def test_the_topics_cover_every_feature() -> None:
    X, _ = corpus()
    X = sparse.hstack([X, sparse.csr_matrix((N_CELLS, 3))]).tocsr()

    model = GibbsLDA(N_TOPICS, eta=0.1, iterations=10)
    model.fit_transform(X)

    assert model.topics_.shape == (N_TOPICS, N_FEATURES + 3)
    assert np.all(model.topics_[:, -3:] > 0)
    np.testing.assert_allclose(model.topics_.sum(axis=1), 1.0, rtol=1e-5)


def test_the_same_seed_gives_the_same_fit() -> None:
    X, _ = corpus()

    first = GibbsLDA(N_TOPICS, iterations=20, random_state=3).fit_transform(X)
    second = GibbsLDA(N_TOPICS, iterations=20, random_state=3).fit_transform(X)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "matrix_type",
    [np.asarray, sparse.csc_matrix, ad.AnnData],
    ids=["dense", "csc", "anndata"],
)
def test_every_input_type_gives_the_same_fit(
    matrix_type: Callable[[Any], Any],
) -> None:
    X, _ = corpus()
    expected = GibbsLDA(N_TOPICS, iterations=10).fit_transform(X)

    result = GibbsLDA(N_TOPICS, iterations=10).fit_transform(
        matrix_type(X.toarray() if matrix_type is np.asarray else X)
    )

    np.testing.assert_array_equal(result, expected)


def test_a_backed_matrix_gives_the_same_fit(tmp_path: Path) -> None:
    X, _ = corpus()
    path = tmp_path / "counts.h5ad"
    ad.AnnData(X).write_h5ad(path)

    expected = GibbsLDA(N_TOPICS, iterations=10).fit_transform(X)
    result = GibbsLDA(N_TOPICS, iterations=10).fit_transform(
        ad.read_h5ad(path, backed="r")
    )

    np.testing.assert_array_equal(result, expected)


def test_a_cell_without_counts_keeps_the_prior_proportions() -> None:
    X, _ = corpus()
    X = X.tolil()
    X[7] = 0
    alpha = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    theta = GibbsLDA(N_TOPICS, alpha=alpha, iterations=10).fit_transform(X.tocsr())

    np.testing.assert_allclose(theta[7], alpha / alpha.sum(), rtol=1e-6)


def test_binarize_fits_the_binarized_matrix() -> None:
    X, _ = corpus()

    result = GibbsLDA(N_TOPICS, binarize=True, iterations=10).fit_transform(X)
    expected = GibbsLDA(N_TOPICS, iterations=10).fit_transform(
        (X > 0).astype(np.float32)
    )

    np.testing.assert_array_equal(result, expected)


def test_alpha_auto_learns_the_prior() -> None:
    X, _ = corpus()

    model = GibbsLDA(N_TOPICS, alpha="auto", iterations=50)
    model.fit_transform(X)

    assert model.alpha_.shape == (N_TOPICS,)
    assert not np.allclose(model.alpha_, 1 / N_TOPICS)


def test_the_log_likelihood_is_recorded_every_10_iterations() -> None:
    X, _ = corpus()

    model = GibbsLDA(N_TOPICS, iterations=35)
    model.fit_transform(X)

    assert len(model.log_likelihood_) == 4
    assert model.log_likelihood_[-1] > model.log_likelihood_[0]


def test_non_integer_counts_are_refused() -> None:
    X, _ = corpus()
    X = X * 0.5

    with pytest.raises(ValueError, match="integer counts"):
        GibbsLDA(N_TOPICS, iterations=1).fit_transform(X)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"n_topics": 0}, "n_topics must be at least 1"),
        ({"iterations": 0}, "iterations must be at least 1"),
        ({"n_jobs": 0}, "n_jobs must be at least 1"),
        ({"alpha": "flat"}, "alpha must be one of"),
        ({"eta": "auto"}, "eta must be 'symmetric'"),
        ({"eta": -1.0}, "eta must be positive"),
    ],
)
def test_invalid_options_are_refused(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GibbsLDA(**{"n_topics": N_TOPICS, **options})


def test_the_topics_need_a_fit() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        _ = GibbsLDA(N_TOPICS).topics_


def test_the_topic_counts_add_up_to_the_tokens() -> None:
    X, _ = corpus()

    model = GibbsLDA(N_TOPICS, iterations=10)
    model.fit_transform(X)

    assert model.topic_counts_.shape == (N_TOPICS,)
    assert model.topic_counts_.sum() == X.sum()

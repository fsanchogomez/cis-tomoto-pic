from __future__ import annotations

import pickle
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from cis_tomoto.cli import AVAILABLE_PROCESSORS, app, parse_eta
from cis_tomoto.clustering import SEED
from cis_tomoto.gibbs import GibbsLDA
from scipy import sparse
from typer.testing import CliRunner

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from typer.testing import Result

N_CELLS = 40
N_FEATURES = 12
N_TOPICS = 3

runner = CliRunner()


def poisson_counts() -> np.ndarray:
    return (
        np.random
        .default_rng(0)
        .poisson(3.0, size=(N_CELLS, N_FEATURES))
        .astype(np.float32)
    )


def write_input(tmp_path: Path, X: np.ndarray | sparse.spmatrix) -> Path:
    path = tmp_path / "counts.h5ad"
    adata = ad.AnnData(X)
    adata.obs_names = [f"cell{index}" for index in range(N_CELLS)]
    adata.write_h5ad(path)
    return path


def run(input_path: Path, out_path: Path, *options: str) -> Result:
    return runner.invoke(
        app,
        [
            "-i",
            str(input_path),
            "-o",
            str(out_path),
            "-n",
            str(N_TOPICS),
            "--iterations",
            "20",
            *options,
        ],
    )


@pytest.mark.parametrize(
    "matrix_type",
    [np.asarray, sparse.csr_matrix, sparse.csc_matrix],
    ids=lambda matrix_type: matrix_type.__name__,
)
def test_cis_lda_writes_the_results_to_the_output_file(
    matrix_type: Callable[[np.ndarray], Any], tmp_path: Path
) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, matrix_type(poisson_counts())), out_path)

    assert result.exit_code == 0, result.output
    adata = ad.read_h5ad(out_path)
    assert adata.obsm["X_lda"].shape == (N_CELLS, N_TOPICS)
    np.testing.assert_allclose(adata.obsm["X_lda"].sum(axis=1), 1.0, rtol=1e-5)
    assert adata.varm["LDA_topics"].shape == (N_FEATURES, N_TOPICS)
    report = adata.uns["LDA"]
    assert report["method"] == "collapsed_gibbs"
    assert report["params"]["n_topics"] == N_TOPICS
    assert report["params"]["iterations"] == 20
    assert report["params"]["alpha"] == "symmetric"
    assert report["params"]["eta"] == "symmetric"
    assert report["eta"] == pytest.approx(1 / N_TOPICS)
    assert len(report["log_likelihood"]) == 2
    dense = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    np.testing.assert_array_equal(dense, poisson_counts())
    assert "X_umap" not in adata.obsm


def test_cis_lda_results_equal_the_library_results(tmp_path: Path) -> None:
    input_path = write_input(tmp_path, sparse.csr_matrix(poisson_counts()))
    out_path = tmp_path / "out.h5ad"

    result = run(input_path, out_path)

    assert result.exit_code == 0, result.output
    model = GibbsLDA(
        N_TOPICS, iterations=20, n_jobs=AVAILABLE_PROCESSORS, random_state=SEED
    )
    theta = model.fit_transform(poisson_counts())
    adata = ad.read_h5ad(out_path)
    np.testing.assert_array_equal(adata.obsm["X_lda"], theta)
    np.testing.assert_array_equal(adata.varm["LDA_topics"], model.topics_.T)


def test_cis_lda_passes_every_option_to_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models: list[GibbsLDA] = []
    original = GibbsLDA.fit_transform

    def recording_fit_transform(self: GibbsLDA, X: object) -> np.ndarray:
        models.append(self)
        return original(self, X)

    monkeypatch.setattr(GibbsLDA, "fit_transform", recording_fit_transform)
    out_path = tmp_path / "out.h5ad"

    result = run(
        write_input(tmp_path, poisson_counts()),
        out_path,
        "--alpha",
        "2.5",
        "--eta",
        "0.1",
        "--binarize",
        "--chunkSize",
        "8",
        "-p",
        "1",
    )

    assert result.exit_code == 0, result.output
    (model,) = models
    assert model.n_topics == N_TOPICS
    assert model.alpha == 2.5
    assert model.eta == 0.1
    assert model.binarize
    assert model.iterations == 20
    assert model.chunk_size == 8
    assert model.n_jobs == 1
    params = ad.read_h5ad(out_path).uns["LDA"]["params"]
    assert params["alpha"] == "2.5"
    assert params["eta"] == "0.1"


def test_cis_lda_plots_the_umap_with_leiden_clusters(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"
    plot_path = tmp_path / "umap.png"

    result = run(
        write_input(tmp_path, poisson_counts()),
        out_path,
        "-op",
        str(plot_path),
        "-nk",
        "5",
        "--dpi",
        "50",
    )

    assert result.exit_code == 0, result.output
    assert plot_path.read_bytes().startswith(b"\x89PNG")
    adata = ad.read_h5ad(out_path)
    assert adata.obsm["X_umap"].shape == (N_CELLS, 2)
    table = pd.read_csv(plot_path.with_suffix(".tsv"), sep="\t", dtype={"cluster": str})
    assert table["cluster"].tolist() == adata.obs["leiden"].astype(str).tolist()


def test_cis_lda_reports_errors_without_a_traceback(tmp_path: Path) -> None:
    out_path = tmp_path / "out.h5ad"

    result = run(write_input(tmp_path, poisson_counts() * 0.5), out_path)

    assert result.exit_code == 1
    assert "integer counts" in result.output
    assert not out_path.exists()


def test_parse_eta() -> None:
    assert parse_eta("symmetric") == "symmetric"
    assert parse_eta("0.1") == 0.1


def test_cis_lda_refuses_eta_auto(tmp_path: Path) -> None:
    result = run(
        write_input(tmp_path, poisson_counts()), tmp_path / "out.h5ad", "--eta", "auto"
    )

    assert result.exit_code == 2


def test_cis_lda_prints_the_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"cis-lda {version('cis-tomoto-pic')}"


def test_cis_lda_prints_the_help() -> None:
    result = runner.invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "--iterations" in result.output
    assert "collapsed Gibbs" in result.output


def test_cis_lda_uses_every_processor_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models: list[GibbsLDA] = []
    original = GibbsLDA.fit_transform

    def recording_fit_transform(self: GibbsLDA, X: object) -> np.ndarray:
        models.append(self)
        return original(self, X)

    monkeypatch.setattr(GibbsLDA, "fit_transform", recording_fit_transform)

    result = run(write_input(tmp_path, poisson_counts()), tmp_path / "out.h5ad")

    assert result.exit_code == 0, result.output
    (model,) = models
    assert model.n_jobs == AVAILABLE_PROCESSORS


def write_region_input(tmp_path: Path) -> Path:
    path = tmp_path / "counts.h5ad"
    adata = ad.AnnData(sparse.csr_matrix(poisson_counts()))
    adata.obs_names = [f"cell{index}" for index in range(N_CELLS)]
    adata.var_names = [
        f"chr1_{index * 100}_{index * 100 + 100}" for index in range(N_FEATURES)
    ]
    adata.write_h5ad(path)
    return path


def test_cis_lda_writes_a_cistopic_object(tmp_path: Path) -> None:
    pytest.importorskip("pycisTopic")
    out_path = tmp_path / "out.pkl"
    plot_path = tmp_path / "umap.png"

    result = run(
        write_region_input(tmp_path),
        out_path,
        "--cistopic",
        "--project",
        "test",
        "-op",
        str(plot_path),
        "-nk",
        "5",
        "--dpi",
        "50",
    )

    assert result.exit_code == 0, result.output
    with out_path.open("rb") as file:
        cistopic = pickle.load(file)
    assert cistopic.project == "test"
    model = GibbsLDA(
        N_TOPICS, iterations=20, n_jobs=AVAILABLE_PROCESSORS, random_state=SEED
    )
    theta = model.fit_transform(poisson_counts())
    np.testing.assert_array_equal(
        cistopic.selected_model.cell_topic.to_numpy(), theta.T
    )
    table = pd.read_csv(plot_path.with_suffix(".tsv"), sep="\t", dtype={"cluster": str})
    assert (
        table["cluster"].tolist() == cistopic.cell_data["leiden"].astype(str).tolist()
    )
    np.testing.assert_allclose(
        cistopic.projections["cell"]["UMAP"].to_numpy(),
        table[["UMAP1", "UMAP2"]].to_numpy(),
    )


def test_cis_lda_refuses_a_cistopic_object_of_features_that_are_not_regions(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pycisTopic")
    out_path = tmp_path / "out.pkl"

    result = run(write_input(tmp_path, poisson_counts()), out_path, "--cistopic")

    assert result.exit_code == 1
    assert "not a region" in result.output
    assert not out_path.exists()

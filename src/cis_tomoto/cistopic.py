from __future__ import annotations

import re
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
from anndata.abc import CSCDataset, CSRDataset
from scipy import sparse

if TYPE_CHECKING:
    import anndata as ad
    from pycisTopic.cistopic_class import CistopicObject
    from pycisTopic.lda_models import CistopicLDAModel

    from cis_tomoto.gibbs import GibbsLDA

TOP_TOPICS_COHERENCE = 5
LOCUS = re.compile(r"^(?P<chrom>.+?)[:_](?P<start>\d+)[-_](?P<end>\d+)(?:::.*)?$")


def require_pycistopic() -> None:
    try:
        import pycisTopic.cistopic_class  # noqa: PLC0415
        import pycisTopic.lda_models  # noqa: F401, PLC0415
        import tmtoolkit  # noqa: F401, PLC0415
    except ImportError as exc:
        msg = (
            "A cistopic object needs pycisTopic "
            "(https://github.com/aertslab/pycisTopic) and tmtoolkit: "
            f"{exc}"
        )
        raise ImportError(msg) from exc


def cistopic_region_names(features: pd.Index) -> list[str]:
    """The features as pycisTopic regions, "chrom:start-end".

    It accepts "chrom:start-end" and the "chrom_start_end" or "chrom_start_end::name"
    names of sincei.
    """
    names = []
    for feature in features.astype(str):
        match = LOCUS.match(feature)
        if match is None:
            msg = (
                f"The feature {feature!r} is not a region; pycisTopic needs features "
                "named chrom:start-end or chrom_start_end."
            )
            raise ValueError(msg)
        names.append(f"{match['chrom']}:{match['start']}-{match['end']}")
    if len(set(names)) < len(names):
        msg = "Two or more features have the same region; pycisTopic needs unique ones."
        raise ValueError(msg)
    return names


def in_memory(matrix: object) -> sparse.csr_matrix:
    if isinstance(matrix, CSRDataset | CSCDataset):
        matrix = matrix.to_memory()
    elif isinstance(matrix, h5py.Dataset):
        matrix = matrix[()]
    return sparse.csr_matrix(matrix)


def lda_model(
    model: GibbsLDA,
    theta: np.ndarray,
    tokens: sparse.csr_matrix,
    cell_names: list[str],
    region_names: list[str],
    elapsed: float,
) -> CistopicLDAModel:
    """The fit as a CistopicLDAModel, with the metrics of pycisTopic's run_cgs_model.

    As pycisTopic removes the features without tokens before its fit, the topics are
    restricted to the others, and normalized again. The log-likelihood is the one of
    tomotopy, for all the tokens.
    """
    from pycisTopic.lda_models import CistopicLDAModel  # noqa: PLC0415
    from tmtoolkit.topicmod import evaluate, model_stats  # noqa: PLC0415

    used = tokens.getnnz(axis=0) > 0
    topic_word = model.topics_[:, used]
    topic_word /= topic_word.sum(axis=1, keepdims=True)
    tokens = tokens[:, used]
    lengths = np.asarray(tokens.sum(axis=1), dtype=np.float64).ravel()
    mimno = evaluate.metric_coherence_mimno_2011(
        topic_word,
        dtm=tokens,
        top_n=min(20, topic_word.shape[1]),
        eps=1e-12,
        normalize=True,
        return_mean=False,
    )
    top = (
        mimno
        if len(mimno) <= TOP_TOPICS_COHERENCE
        else mimno[
            np.argpartition(mimno, -TOP_TOPICS_COHERENCE)[-TOP_TOPICS_COHERENCE:]
        ]
    )
    metrics = pd.DataFrame(
        [
            evaluate.metric_arun_2010(topic_word, theta, lengths),
            evaluate.metric_cao_juan_2009(topic_word),
            np.mean(top),
            model.log_likelihood_[-1] * lengths.sum(),
        ],
        index=["Arun_2010", "Cao_Juan_2009", "Mimno_2011", "loglikelihood"],
        columns=["Metric"],
    ).transpose()

    topics = list(range(1, model.n_topics + 1))
    topic_names = [f"Topic{topic}" for topic in topics]
    marginal = model_stats.marginal_topic_distrib(theta, lengths)
    alpha = model.alpha_
    parameters = pd.DataFrame(
        [
            "tomotopy",
            model.n_topics,
            model.iterations,
            model.random_state,
            float(alpha[0]) if np.all(alpha == alpha[0]) else alpha.tolist(),
            False,
            model.eta_,
            False,
            TOP_TOPICS_COHERENCE,
            elapsed,
            model.binarize,
            model.n_jobs,
        ],
        index=[
            "package",
            "n_topics",
            "n_iter",
            "random_state",
            "alpha",
            "alpha_by_topic",
            "eta",
            "eta_by_topic",
            "top_topics_coh",
            "time",
            "binarize",
            "n_jobs",
        ],
        columns=["Parameter"],
    )
    return CistopicLDAModel(
        metrics,
        pd.DataFrame([topics, mimno], index=["Topic", "Mimno_2011"]).transpose(),
        pd.DataFrame([topics, marginal], index=["Topic", "Marg_Topic"]).transpose(),
        pd.DataFrame.from_records(
            [topics, model.topic_counts_], index=["Topic", "Assignments"]
        ).transpose(),
        pd.DataFrame(theta.T, index=topic_names, columns=cell_names),
        pd.DataFrame(
            topic_word.T, index=np.asarray(region_names)[used], columns=topic_names
        ),
        parameters,
    )


def to_cistopic(
    adata: ad.AnnData,
    model: GibbsLDA,
    theta: np.ndarray,
    project: str = "cisTopic",
    elapsed: float = 0.0,
    embedding: np.ndarray | None = None,
    clusters: pd.Categorical | None = None,
) -> CistopicObject:
    """A CistopicObject of the counts of `adata`, with the fit as its selected model.

    The matrix of `adata` is loaded into memory. The cell names are kept, without the
    project tag; the columns of ``adata.obs`` are added to the cell data, and the
    feature names to the column ``var_name`` of the region data. Features without
    counts are removed, as pycisTopic does. The UMAP goes to
    ``projections["cell"]["UMAP"]`` and the clusters to ``cell_data["leiden"]``.
    """
    require_pycistopic()
    from pycisTopic.cistopic_class import create_cistopic_object  # noqa: PLC0415

    region_names = cistopic_region_names(adata.var_names)
    cell_names = adata.obs_names.astype(str).tolist()
    counts = in_memory(adata.X)
    cistopic = create_cistopic_object(
        sparse.csr_matrix(counts.T),
        cell_names=cell_names,
        region_names=region_names,
        project=project,
        tag_cells=False,
    )
    feature_names = pd.Series(adata.var_names.astype(str), index=region_names)
    cistopic.region_data["var_name"] = feature_names.loc[
        cistopic.region_names
    ].to_numpy()
    if adata.obs.shape[1]:
        cistopic.add_cell_data(adata.obs.set_axis(cell_names))

    tokens = counts.copy()
    tokens.sum_duplicates()
    tokens.eliminate_zeros()
    if model.binarize:
        tokens.data[:] = 1
    cistopic.add_LDA_model(
        lda_model(model, theta, tokens, cell_names, region_names, elapsed)
    )

    if embedding is not None:
        cistopic.projections["cell"]["UMAP"] = pd.DataFrame(
            embedding, index=cell_names, columns=["UMAP_1", "UMAP_2"]
        )
    if clusters is not None:
        cistopic.cell_data["leiden"] = pd.Series(clusters, index=cell_names)
    return cistopic

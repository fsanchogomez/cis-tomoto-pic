from __future__ import annotations

import logging

import numpy as np
import tomotopy as tp

from cis_tomoto.inputs import CellChunks, Matrix

DEFAULT_ITERATIONS = 150
DEFAULT_CHUNK_SIZE = 4096
LOG_INTERVAL = 10
OPTIMIZE_INTERVAL = 10
ALPHA_CHOICES = ("symmetric", "asymmetric", "auto")


def initial_alpha(alpha: str | float | np.ndarray, n_topics: int) -> np.ndarray:
    if isinstance(alpha, str):
        if alpha in ("symmetric", "auto"):
            return np.full(n_topics, 1.0 / n_topics, dtype=np.float32)
        if alpha == "asymmetric":
            prior = 1.0 / (np.arange(n_topics) + np.sqrt(n_topics))
            return (prior / prior.sum()).astype(np.float32)
        msg = (
            f"alpha must be one of {ALPHA_CHOICES}, a number or an array, not "
            f"{alpha!r}."
        )
        raise ValueError(msg)
    prior = np.broadcast_to(np.asarray(alpha, dtype=np.float32), (n_topics,)).copy()
    if (prior <= 0).any():
        msg = "alpha must be positive."
        raise ValueError(msg)
    return prior


class GibbsLDA:
    r"""Latent Dirichlet allocation of a cell-by-feature count matrix, fitted by
    collapsed Gibbs sampling with tomotopy.

    Each cell is a document and each count of a feature is one token of that feature.
    Every token holds a topic assignment, and each iteration samples every assignment
    again from the counts of the others (Griffiths and Steyvers, 2004), as pycisTopic
    does. The cells are read in chunks, but all the tokens are held by tomotopy during
    the fit.

    The results are those of the last sample: the topic proportions of a cell are
    :math:`(n_{dk} + \alpha_k) / (n_d + \sum_j \alpha_j)` and the feature
    distribution of a topic is :math:`(n_{kw} + \eta) / (n_k + V \eta)`, over every
    feature of the input.

    Parameters
    ----------
    n_topics
        Number of topics K.
    alpha
        Prior of the topic proportions of a cell: "symmetric" (1/K), "asymmetric"
        (1/(k + sqrt(K)), normalized), "auto" (learned every 10 iterations by
        tomotopy, starting at 1/K), a number, or an array of K values. pycisTopic
        uses 50/K.
    eta
        Prior of the feature weights of a topic: "symmetric" or None (1/K), or a
        number. pycisTopic uses 0.1.
    iterations
        Number of Gibbs sampling iterations over all the tokens.
    binarize
        Count every non-zero entry as 1.
    chunk_size
        Number of cells read at a time.
    n_jobs
        Number of threads of tomotopy. The result of a seed depends on it.
    random_state
        Seed of the sampler.
    """

    def __init__(
        self,
        n_topics: int = 20,
        alpha: str | float | np.ndarray = "symmetric",
        eta: str | float | None = None,
        iterations: int = DEFAULT_ITERATIONS,
        binarize: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        n_jobs: int = 1,
        random_state: int = 42,
    ) -> None:
        for name, value in (
            ("n_topics", n_topics),
            ("iterations", iterations),
            ("chunk_size", chunk_size),
            ("n_jobs", n_jobs),
        ):
            if value < 1:
                msg = f"{name} must be at least 1, not {value}."
                raise ValueError(msg)
        if isinstance(eta, str):
            if eta != "symmetric":
                msg = f"eta must be 'symmetric', a number or None, not {eta!r}."
                raise ValueError(msg)
        elif eta is not None and eta <= 0:
            msg = "eta must be positive."
            raise ValueError(msg)
        self.n_topics = n_topics
        self.alpha = alpha
        self.eta = eta
        self.iterations = iterations
        self.binarize = binarize
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._alpha = initial_alpha(alpha, n_topics)
        self._eta = 1.0 / n_topics if eta is None or isinstance(eta, str) else eta
        self._topics: np.ndarray | None = None
        self.log_likelihood_: list[float] = []
        self.topic_counts_: np.ndarray | None = None

    @property
    def topics_(self) -> np.ndarray:
        """E[beta]: the feature distribution of every topic, topics x features."""
        if self._topics is None:
            msg = "The model is not fitted yet. Call fit_transform() first."
            raise ValueError(msg)
        return self._topics

    @property
    def alpha_(self) -> np.ndarray:
        return self._alpha

    @property
    def eta_(self) -> float:
        return float(self._eta)

    def fit_transform(self, X: Matrix) -> np.ndarray:
        """Fits the model, and returns the topic proportions of the cells."""
        data = CellChunks(X, self.chunk_size, self.binarize)
        model = tp.LDAModel(
            tw=tp.TermWeight.ONE,
            min_cf=0,
            min_df=0,
            rm_top=0,
            k=self.n_topics,
            alpha=self._alpha.tolist(),
            eta=float(self._eta),
            seed=self.random_state,
        )
        names = np.array([str(feature) for feature in range(data.n_features)])
        modelled = []
        for rows, block in data.chunks():
            if block.nnz and not np.array_equal(block.data, np.round(block.data)):
                msg = (
                    "Collapsed Gibbs sampling needs integer counts; use binarize "
                    "or round the matrix."
                )
                raise ValueError(msg)
            counts = block.data.astype(np.int64)
            for row in range(block.shape[0]):
                low, high = block.indptr[row], block.indptr[row + 1]
                if high > low:
                    tokens = np.repeat(names[block.indices[low:high]], counts[low:high])
                    model.add_doc(tokens.tolist())
                    modelled.append(rows[row])

        model.optim_interval = (
            OPTIMIZE_INTERVAL
            if isinstance(self.alpha, str) and self.alpha == "auto"
            else 0
        )
        self.log_likelihood_ = []
        done = 0
        while done < self.iterations:
            step = min(LOG_INTERVAL, self.iterations - done)
            model.train(step, workers=self.n_jobs)
            done += step
            self.log_likelihood_.append(float(model.ll_per_word))
            logging.info(
                "Iteration %d: log-likelihood per token %.6f", done, model.ll_per_word
            )

        self._alpha = np.asarray(model.alpha, dtype=np.float32)
        theta = np.tile(self._alpha / self._alpha.sum(), (data.n_cells, 1))
        theta[np.asarray(modelled, dtype=np.int64)] = np.array(
            [document.get_topic_dist() for document in model.docs], dtype=np.float32
        )
        self._topics = self._full_topics(model, data.n_features)
        self.topic_counts_ = np.asarray(model.get_count_by_topics(), dtype=np.int64)
        return theta

    def _full_topics(self, model: tp.LDAModel, n_features: int) -> np.ndarray:
        """(n_kw + eta) / (n_k + V eta) over every feature, with or without tokens."""
        features = np.array([int(word) for word in model.used_vocabs])
        counts = np.zeros((self.n_topics, n_features))
        for topic in range(self.n_topics):
            raw = np.asarray(model.get_topic_word_dist(topic, normalize=False))
            counts[topic, features] = raw - self._eta
        weights = np.clip(counts, 0.0, None) + self._eta
        return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)

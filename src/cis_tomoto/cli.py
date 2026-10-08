from __future__ import annotations

import logging
import os
import pickle
import shutil
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import typer
from anndata.abc import CSCDataset
from anndata.io import write_elem

from cis_tomoto.api import fit, lda_elements
from cis_tomoto.cistopic import cistopic_region_names, require_pycistopic, to_cistopic
from cis_tomoto.clustering import (
    SEED,
    PlotFileFormat,
    leiden_categories,
    plot_umap,
    umap_leiden,
)
from cis_tomoto.gibbs import ALPHA_CHOICES, DEFAULT_CHUNK_SIZE, DEFAULT_ITERATIONS

DESCRIPTION = (
    "Find the topics of a cell-by-feature count matrix with latent Dirichlet "
    "allocation (LDA), fitted by collapsed Gibbs sampling (tomotopy).\n\n"
    "``cis-lda`` fits LDA to the counts in ``.X`` of an h5ad file, with cells in rows "
    "and features in columns, by collapsed Gibbs sampling, as pycisTopic does. The "
    "results are those of the last sample. The output is a copy of the input file "
    "with:\n"
    '* ``obsm["X_lda"]``: the topic proportions of the cells.\n'
    '* ``varm["LDA_topics"]``: the feature distribution of the topics.\n'
    '* ``uns["LDA"]``: the parameters of the fit, the priors and the log-likelihood '
    "per token every 10 iterations.\n\n"
    "With ``--cistopic``, the output is instead a pickle file of a pycisTopic "
    "CistopicObject, with the fit as its selected model.\n\n"
    "If ``--outFileUMAP`` is given, ``cis-lda`` also computes a 2D projection "
    "(UMAP) and Leiden clusters of the cells, from the square root of their topic "
    'proportions, and stores them in ``obsm["X_umap"]`` and ``obs["leiden"]`` (or '
    'in ``projections["cell"]["UMAP"]`` and ``cell_data["leiden"]`` of the '
    "CistopicObject), with a plot file and a .tsv file."
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help=DESCRIPTION,
    context_settings={"help_option_names": []},
)

_IO = "Input / Output options"
_LDA = "LDA options"
_SAMPLING = "Sampling options"
_CLUSTERING = "Clustering options"
_PLOT = "Plot options"
_OTHER = "Other options"

AVAILABLE_PROCESSORS = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else os.cpu_count() or 1
)


def normalize_processors(value: str | int) -> int:
    if value == "max/2":
        return max(AVAILABLE_PROCESSORS // 2, 1)
    if value == "max":
        return AVAILABLE_PROCESSORS

    try:
        number_of_processors = int(value)
    except ValueError as exc:
        msg = f"{value} is not a valid number of processors"
        raise typer.BadParameter(msg) from exc

    if number_of_processors < 1:
        msg = f"{value} is not a valid number of processors"
        raise typer.BadParameter(msg)
    return min(number_of_processors, AVAILABLE_PROCESSORS)


def parse_prior(value: str, choices: tuple[str, ...]) -> str | float:
    if value in choices:
        return value
    try:
        number = float(value)
    except ValueError as exc:
        msg = f"{value} is not one of {', '.join(choices)} or a number"
        raise typer.BadParameter(msg) from exc
    if number <= 0:
        msg = f"{value} is not a positive number"
        raise typer.BadParameter(msg)
    return number


def parse_alpha(value: str) -> str | float:
    return parse_prior(value, ALPHA_CHOICES)


def parse_eta(value: str) -> str | float:
    return parse_prior(value, ("symmetric",))


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"cis-lda {version('cis-tomoto-pic')}")
        raise typer.Exit()


def help_callback(ctx: typer.Context, value: bool) -> None:
    if value:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def log_parameters(**parameters: object) -> None:
    for name, value in parameters.items():
        logging.info("%s: %s", name, value)


def configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def fail(message: str) -> typer.Exit:
    sys.stderr.write(message + "\n")
    return typer.Exit(code=1)


def write_results(
    input: str,
    out_file: str,
    elements: dict[str, dict[str, object]],
    frames: dict[str, pd.DataFrame],
) -> None:
    """Copies the input file to `out_file`, and adds `elements` and `frames` to it.

    `frames` replaces the obs or var tables. The matrix is copied as a file and never
    read, so it costs no memory.
    """
    shutil.copyfile(input, out_file)
    with h5py.File(out_file, "r+") as file:
        for group, values in elements.items():
            if group not in file:
                write_elem(file, group, {})
            for key, value in values.items():
                if key in file[group]:
                    del file[group][key]
                write_elem(file[group], key, value)
        for name, frame in frames.items():
            del file[name]
            write_elem(file, name, frame)


def open_input(input: str, out_file: str) -> tuple[ad.AnnData, object]:
    """The backed input file and its matrix; a CSC matrix is loaded into memory."""
    if Path(input).resolve() == Path(out_file).resolve():
        msg = "The output file must not be the input file."
        raise fail(msg)
    adata = ad.read_h5ad(input, backed="r")
    matrix = adata.X
    if matrix is None:
        msg = f"'{input}' has no matrix in .X."
        raise fail(msg)
    if isinstance(matrix, CSCDataset):
        logging.warning(
            "The matrix of '%s' is stored as CSC, from which rows cannot be read "
            "efficiently; it is loaded into memory.",
            input,
        )
        matrix = matrix.to_memory()
    return adata, matrix


@app.callback(invoke_without_command=True)
def main(
    # Input / Output options
    input: Annotated[
        str,
        typer.Option(
            "-i",
            "--input",
            metavar=".h5ad",
            rich_help_panel=_IO,
            help=(
                "Input file in .h5ad format, with cells in rows and features in "
                "columns."
            ),
        ),
    ],
    out_file: Annotated[
        str,
        typer.Option(
            "-o",
            "--outFile",
            metavar="FILE",
            rich_help_panel=_IO,
            help=(
                "The output .h5ad file: the input with the LDA results added. With "
                "``--cistopic``, the output pickle (.pkl) file of the CistopicObject."
            ),
        ),
    ],
    out_file_umap: Annotated[
        str | None,
        typer.Option(
            "-op",
            "--outFileUMAP",
            rich_help_panel=_IO,
            help=(
                "The output plot file (for UMAP). If specified, the UMAP coordinates "
                'and Leiden clusters are added to the .h5ad file as ``obsm["X_umap"]`` '
                'and ``obs["leiden"]``, and a 4-column .tsv file with the same prefix '
                "is also created with the cell IDs, raw UMAP coordinates (UMAP1 and "
                "UMAP2) and Leiden cluster number."
            ),
        ),
    ] = None,
    cistopic: Annotated[
        bool,
        typer.Option(
            "--cistopic",
            rich_help_panel=_IO,
            help=(
                "Write a pycisTopic CistopicObject, with the counts and the fit as "
                "its selected model, as a pickle file, instead of an .h5ad file. It "
                "needs pycisTopic, and features named as regions (chrom:start-end or "
                "chrom_start_end). The matrix is loaded into memory."
            ),
        ),
    ] = False,
    project: Annotated[
        str,
        typer.Option(
            "--project",
            rich_help_panel=_IO,
            help="Project name of the CistopicObject. Only used with ``--cistopic``.",
        ),
    ] = "cisTopic",
    # LDA options
    n_topics: Annotated[
        int,
        typer.Option(
            "-n",
            "--nTopics",
            min=1,
            rich_help_panel=_LDA,
            help=(
                "Number of topics. Use a higher number for samples with more expected "
                "heterogeneity."
            ),
        ),
    ] = 20,
    alpha: Annotated[
        str,
        typer.Option(
            "--alpha",
            parser=parse_alpha,
            metavar="ALPHA",
            rich_help_panel=_LDA,
            help=(
                "Prior of the topic proportions of a cell. "
                "[bold yellow]symmetric[/bold yellow] is 1/nTopics for every topic; "
                "[bold yellow]asymmetric[/bold yellow] is 1/(k + sqrt(nTopics)) for "
                "topic k, normalized; [bold yellow]auto[/bold yellow] learns it from "
                "the data every 10 iterations. A number sets the same value for every "
                "topic; pycisTopic uses 50/nTopics."
            ),
        ),
    ] = "symmetric",
    eta: Annotated[
        str | None,
        typer.Option(
            "--eta",
            parser=parse_eta,
            metavar="ETA",
            show_default="symmetric",
            rich_help_panel=_LDA,
            help=(
                "Prior of the feature weights of a topic. "
                "[bold yellow]symmetric[/bold yellow] is 1/nTopics for every "
                "feature. A number sets the same value for every feature; pycisTopic "
                "uses 0.1."
            ),
        ),
    ] = None,
    binarize: Annotated[
        bool,
        typer.Option(
            "--binarize",
            rich_help_panel=_LDA,
            help=(
                "Count every non-zero entry as 1, off by default. Without it, the "
                "counts must be integers."
            ),
        ),
    ] = False,
    # Sampling options
    iterations: Annotated[
        int,
        typer.Option(
            "--iterations",
            min=1,
            rich_help_panel=_SAMPLING,
            help="Number of Gibbs sampling iterations over all the tokens.",
        ),
    ] = DEFAULT_ITERATIONS,
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunkSize",
            min=1,
            rich_help_panel=_SAMPLING,
            help="Number of cells read from the input at a time.",
        ),
    ] = DEFAULT_CHUNK_SIZE,
    # Clustering options
    n_neighbors: Annotated[
        int,
        typer.Option(
            "-nk",
            "--nNeighbors",
            rich_help_panel=_CLUSTERING,
            help=(
                "Number of nearest neighbours to consider for clustering and UMAP. "
                "Only used with ``--outFileUMAP``."
            ),
        ),
    ] = 15,
    cluster_resolution: Annotated[
        float,
        typer.Option(
            "-cr",
            "--clusterResolution",
            rich_help_panel=_CLUSTERING,
            help=(
                "Resolution parameter for Leiden clustering. Values lower than 1.0 "
                "result in fewer clusters. Only used with ``--outFileUMAP``."
            ),
        ),
    ] = 1.0,
    # Plot options
    plot_width: Annotated[
        float,
        typer.Option(
            "--plotWidth",
            metavar="FLOAT",
            rich_help_panel=_PLOT,
            help="Output plot width (in cm).",
        ),
    ] = 25.0,
    plot_height: Annotated[
        float,
        typer.Option(
            "--plotHeight",
            metavar="FLOAT",
            rich_help_panel=_PLOT,
            help="Output plot height (in cm).",
        ),
    ] = 25.0,
    plot_file_format: Annotated[
        PlotFileFormat,
        typer.Option(
            "--plotFileFormat",
            metavar="FORMAT",
            rich_help_panel=_PLOT,
            help=(
                "Image format type of the plot file.\n\n"
                "One of: [bold yellow]png[/bold yellow], "
                "[bold yellow]jpg[/bold yellow], "
                "[bold yellow]svg[/bold yellow], "
                "[bold yellow]pdf[/bold yellow]."
            ),
        ),
    ] = PlotFileFormat.png,
    dpi: Annotated[
        int,
        typer.Option(
            "--dpi",
            metavar="INT",
            rich_help_panel=_PLOT,
            help="Resolution of the output image, in dots per inch.",
        ),
    ] = 300,
    # Other options
    number_of_processors: Annotated[
        int,
        typer.Option(
            "-p",
            "--numberOfProcessors",
            parser=normalize_processors,
            metavar="INT",
            show_default="max",
            rich_help_panel=_OTHER,
            help=(
                "Number of sampling threads. The result of the fixed seed depends on "
                'it. You can also type "max/2" to use half the maximum number of '
                'processors or "max" to use all available processors.'
            ),
        ),
    ] = AVAILABLE_PROCESSORS,
    verbose: Annotated[
        bool,
        typer.Option(
            "-v",
            "--verbose",
            rich_help_panel=_OTHER,
            help="Set to see processing messages.",
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "-V",
            "--version",
            is_eager=True,
            expose_value=False,
            callback=version_callback,
            rich_help_panel=_OTHER,
            help="Print the program version and exit.",
        ),
    ] = False,
    help: Annotated[
        bool,
        typer.Option(
            "-h",
            "--help",
            is_eager=True,
            expose_value=False,
            callback=help_callback,
            rich_help_panel=_OTHER,
            help="Show this message and exit.",
        ),
    ] = False,
) -> int:
    params = {
        "n_topics": n_topics,
        "alpha": alpha,
        "eta": eta,
        "iterations": iterations,
        "binarize": binarize,
        "chunk_size": chunk_size,
    }
    if verbose:
        log_parameters(
            input=input,
            out_file=out_file,
            **params,
            out_file_umap=out_file_umap,
            cistopic=cistopic,
            project=project,
            n_neighbors=n_neighbors,
            cluster_resolution=cluster_resolution,
            plot_width=plot_width,
            plot_height=plot_height,
            plot_file_format=plot_file_format,
            dpi=dpi,
            number_of_processors=number_of_processors,
        )
    else:
        warnings.filterwarnings("ignore")

    adata, matrix = open_input(input, out_file)
    try:
        if cistopic:
            require_pycistopic()
            cistopic_region_names(adata.var_names)
        model, theta, elapsed = fit(
            matrix, n_jobs=number_of_processors, random_state=SEED, **params
        )
    except (ImportError, ValueError) as exc:
        raise fail(str(exc)) from exc

    embedding = clusters = None
    if out_file_umap is not None:
        embedding, labels = umap_leiden(np.sqrt(theta), n_neighbors, cluster_resolution)
        clusters = leiden_categories(labels)
        plot_umap(
            embedding,
            labels,
            out_file_umap,
            plot_width,
            plot_height,
            plot_file_format,
            dpi,
        )
        table = pd.DataFrame(
            embedding,
            index=adata.obs_names,
            columns=pd.Index(["UMAP1", "UMAP2"]),
        )
        table["cluster"] = clusters
        table.to_csv(
            Path(out_file_umap).with_suffix(".tsv"), sep="\t", index_label="Cell_ID"
        )

    if cistopic:
        cistopic_object = to_cistopic(
            adata, model, theta, project, elapsed, embedding, clusters
        )
        adata.file.close()
        with Path(out_file).open("wb") as file:
            pickle.dump(cistopic_object, file)
        return 0

    elements = lda_elements(model, theta)
    frames: dict[str, pd.DataFrame] = {}
    if embedding is not None:
        elements["obsm"]["X_umap"] = embedding
        frames["obs"] = adata.obs.copy()
        frames["obs"]["leiden"] = clusters
    adata.file.close()
    write_results(input, out_file, elements, frames)
    return 0


def cli() -> None:
    configure_logging()
    app()


if __name__ == "__main__":
    cli()

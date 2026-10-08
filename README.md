# cis-tomoto-pic

Latent Dirichlet allocation (LDA) of single-cell count matrices, fitted by collapsed
Gibbs sampling with [tomotopy](https://github.com/bab2min/tomotopy), as pycisTopic does.
The results go into an AnnData object or into a pycisTopic `CistopicObject`.

## Installation

### Requirements

* Python 3.11, `pycistopic` only works for this version.
* tomotopy, anndata (0.11 or newer), umap-learn, igraph and matplotlib. pip installs
  them.

### Basic installation

Install the package from the directory of the repository:

```bash
cd cis-tomoto-pic
pip install .
```

For development, install it in editable mode, with the test dependencies:

```bash
pip install -e ".[test]"
```

### Installation with the CistopicObject output

To write `CistopicObject`s, install the `cistopic` extra in a Python 3.11 environment:

```bash
conda create -n cis-tomoto python=3.11
conda activate cis-tomoto
pip install ".[cistopic]"
```

With uv:

```bash
uv venv -p 3.11
uv pip install ".[cistopic]"
```

pycisTopic is not on PyPI; the `cistopic` extra installs it from GitHub, together with
tmtoolkit. pycisTopic pins pandas 1.5, so the extra also pins:

* `numpy<2`: pandas 1.5 is built for numpy 1.
* `scipy<1.15`: anndata 0.11, the last release for pandas 1.5, cannot read backed CSR
  matrices with newer scipy.
* `setuptools<81`: pycisTopic imports `pkg_resources`.

### Check the installation

```bash
cis-lda --version
python -c "from cis_tomoto import cis_lda"
```

To check the `CistopicObject` output, run the tests. Without pycisTopic, the
`CistopicObject` tests are skipped.

```bash
pip install -e ".[cistopic,test]"
pytest
```

## Usage

### Input

The input is an .h5ad file (or an AnnData object) with the counts in `.X`, cells in
rows and features in columns. The matrix can be dense, CSR or CSC. The cells are read in
chunks, so a CSR matrix in a file is not loaded into memory.

Collapsed Gibbs sampling needs integer counts. For other values, use `--binarize`
(`binarize=True`), which counts every non-zero entry as 1.

### Command line

Fit 20 topics and write the results to a copy of the input:

```bash
cis-lda -i counts.h5ad -o counts_lda.h5ad -n 20 --binarize
```

Also compute a UMAP and Leiden clusters, and write a plot and a .tsv file:

```bash
cis-lda -i counts.h5ad -o counts_lda.h5ad -n 20 --binarize \
    -op umap.png -nk 15 -cr 1.0
```

Write a `CistopicObject` instead of an .h5ad file:

```bash
cis-lda -i counts.h5ad -o cistopic.pkl -n 20 --binarize --cistopic --project my_sample
```

Fit with the priors of pycisTopic (for 20 topics, alpha = 50/20 = 2.5), with 8 threads:

```bash
cis-lda -i counts.h5ad -o cistopic.pkl -n 20 --binarize --alpha 2.5 --eta 0.1 \
    --cistopic -p 8
```

#### Options

| Option | Default | Description |
| --- | --- | --- |
| `-i`, `--input` | required | Input .h5ad file. |
| `-o`, `--outFile` | required | Output .h5ad file, or .pkl file with `--cistopic`. It must not be the input file. |
| `-op`, `--outFileUMAP` | none | Plot file of the UMAP. It also writes a .tsv file with the same prefix. |
| `--cistopic` | off | Write a pickle file of a `CistopicObject`. |
| `--project` | `cisTopic` | Project name of the `CistopicObject`. |
| `-n`, `--nTopics` | 20 | Number of topics. |
| `--alpha` | `symmetric` | Prior of the topic proportions: `symmetric`, `asymmetric`, `auto` or a number. |
| `--eta` | `symmetric` | Prior of the feature weights: `symmetric` or a number. |
| `--binarize` | off | Count every non-zero entry as 1. |
| `--iterations` | 150 | Number of Gibbs sampling iterations. |
| `--chunkSize` | 4096 | Number of cells read at a time. |
| `-nk`, `--nNeighbors` | 15 | Number of nearest neighbors of the UMAP and the clusters. |
| `-cr`, `--clusterResolution` | 1.0 | Resolution of the Leiden clusters. |
| `--plotWidth`, `--plotHeight` | 25.0 | Size of the plot, in cm. |
| `--plotFileFormat` | `png` | `png`, `jpg`, `svg` or `pdf`. |
| `--dpi` | 300 | Resolution of the plot. |
| `-p`, `--numberOfProcessors` | `max` | Number of sampling threads: a number, `max/2` or `max`. |
| `-v`, `--verbose` | off | Also show the parameters and the warnings. |
| `-V`, `--version` | | Show the version. |
| `-h`, `--help` | | Show the help. |

The seed is fixed (42). The result also depends on the number of threads: use the same
`-p` to get the same result again.

#### Outputs

Without `--cistopic`, the output is a copy of the input with:

* `obsm["X_lda"]`: the topic proportions of the cells (cells × topics).
* `varm["LDA_topics"]`: the feature distribution of the topics (features × topics).
* `uns["LDA"]`: the parameters, the priors and the log-likelihood per token every 10
  iterations.
* With `-op`: `obsm["X_umap"]` and `obs["leiden"]`.

With `--cistopic`, the output is a pickle file of a `CistopicObject` (see below).

With `-op`, the plot shows the UMAP colored by the Leiden clusters. The .tsv file has 4
columns: `Cell_ID`, `UMAP1`, `UMAP2` and `cluster`. The UMAP and the clusters are
computed from the square root of the topic proportions.

#### Read the outputs

```python
import anndata as ad

adata = ad.read_h5ad("counts_lda.h5ad")
theta = adata.obsm["X_lda"]
topics = adata.varm["LDA_topics"]
```

```python
import pickle

with open("cistopic.pkl", "rb") as file:
    cistopic_obj = pickle.load(file)
cell_topic = cistopic_obj.selected_model.cell_topic
```

To read the pickle file, the environment must have pycisTopic.

### Python

```python
from cis_tomoto import cis_lda

# Read a file into memory, fit, and add the results to the AnnData object.
adata = cis_lda("counts.h5ad", n_topics=20, binarize=True, n_jobs=8)

# Also compute the UMAP and the Leiden clusters.
adata = cis_lda(adata, n_topics=20, binarize=True, n_jobs=8, umap=True)

# Return a CistopicObject instead.
cistopic_obj = cis_lda(
    "counts.h5ad", n_topics=20, binarize=True, n_jobs=8, return_cistopic=True
)
```

`cis_lda` takes an AnnData object (in memory or backed) or the path of an .h5ad file.
An AnnData object gets the results and is returned; it is not copied. A path is read
into memory. For a large file, give a backed AnnData object:

```python
import anndata as ad

adata = cis_lda(ad.read_h5ad("counts.h5ad", backed="r"), n_topics=20, binarize=True)
```

The parameters are the options of the command line:

| Parameter | Default | Command-line option |
| --- | --- | --- |
| `n_topics` | 20 | `--nTopics` |
| `alpha` | `"symmetric"` | `--alpha`; also an array of `n_topics` values |
| `eta` | `None` (symmetric) | `--eta` |
| `iterations` | 150 | `--iterations` |
| `binarize` | `False` | `--binarize` |
| `chunk_size` | 4096 | `--chunkSize` |
| `n_jobs` | 1 | `--numberOfProcessors` (default `max` there) |
| `random_state` | 42 | fixed at 42 |
| `umap` | `False` | `--outFileUMAP` (without the plot) |
| `n_neighbors` | 15 | `--nNeighbors` |
| `resolution` | 1.0 | `--clusterResolution` |
| `return_cistopic` | `False` | `--cistopic` |
| `project` | `"cisTopic"` | `--project` |

For more control, use the model class:

```python
from cis_tomoto import GibbsLDA, to_cistopic

model = GibbsLDA(n_topics=20, binarize=True, n_jobs=8)
theta = model.fit_transform(adata)  # cells x topics
topics = model.topics_  # topics x features
cistopic_obj = to_cistopic(adata, model, theta, project="my_sample")
```

### Continue in pycisTopic

The `CistopicObject` works with the pycisTopic functions that read the selected model,
for example:

```python
from pycisTopic.topic_binarization import binarize_topics
from pycisTopic.topic_qc import compute_topic_metrics

region_sets = binarize_topics(cistopic_obj, method="otsu", plot=False)
topic_qc = compute_topic_metrics(cistopic_obj)
```

## The CistopicObject

* The features must be regions: `chrom:start-end`, or the `chrom_start_end` and
  `chrom_start_end::name` names of sincei. They become `chrom:start-end`; the original
  names are in `region_data["var_name"]`.
* The cell names are kept, without the project tag. The columns of `obs` are added to
  `cell_data`.
* Regions without counts are removed, as `create_cistopic_object` does. The topics of
  the model are restricted to the other regions and normalized again.
* The selected model is a `CistopicLDAModel` with the metrics of pycisTopic's
  `run_cgs_model` (Arun 2010, Cao Juan 2009, Mimno 2011, from tmtoolkit). The
  log-likelihood is the one of tomotopy, not the one of pycisTopic.
* With a UMAP, it is in `projections["cell"]["UMAP"]` and the clusters are in
  `cell_data["leiden"]`.
* The count matrix is loaded into memory.

## Defaults

The defaults are those of `scGibbsLDA` (scLDA), not those of pycisTopic. To fit as
pycisTopic does, use `--binarize --alpha <50/nTopics> --eta 0.1`.

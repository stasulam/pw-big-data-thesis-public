# pw-big-data-thesis

## Introduction.

**Thesis in the field of Big Data: Processing and Analysis of Large Data Sets organized by the Warsaw University of Technology.**

The aim of this thesis is to create a system that enables the storage and analysis of large volumes of text documents (e.g., scientific articles). As part of the project, a solution was developed to enable:

- Performing ad-hoc analyses using MongoDB
- Batch data processing using Apache Spark
- Applying modern natural language processing methods to text data
- Preparing a platform for building a Retrieval Augmented Generation (RAG) system

## Structure.

- `data/` (ignored from version control) should be stored in a `data/` directory. Data can be obtained from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv),
- `notebooks/` stores the Notebook used for demonstration purposes,
- `src/` contains the Python package which will be used to glue all of the services,
- `tex/` contains the text of the thesis written in `LaTeX`,
- `docker-compose.yml` implements the infrastructure used by the system.

## Docs.

### Initialize the Data Lake.

To initialize the dataset, you need to:

1. Install the `rag` Python package.
2. Run the command: `rag-init-dataset` (see `--help` for more information about the parameters), e.g., `rag-init-dataset --path-to-raw-data data/arxiv-metadata-oai-snapshot-sample.json`.

## Tasks.

- [x] Add Python dependencies to Spark workers: https://stackoverflow.com/a/77308322/5404616.
- [x] Add model to Spark workers: https://dataking.hashnode.dev/making-predictions-on-a-pyspark-dataframe-with-a-scikit-learn-model-ckzzyrudn01lv25nv41i2ajjh.
- Preprocessing: implementacja jako skrypt .py, zapisany w: src/processor/preprocessing.py:
    - setup sesji Spark (z konfiguracją odczyt z MinIO, zapis: MongoDB),
    - podział `abstract` na zdania: `sentences`,
    - zapis do `database`: `arxiv`, `collection`: `sentences` (obsłużyć tworzenie tego programistycznie w MongoDB),

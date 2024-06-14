"""Command line interface for the RAG package."""

import click


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--path-to-raw-data",
    required=True,
    default="data/raw/arxiv-metadata-oai-snapshot-sample.json",
)
@click.option("--bucket", required=False, default="papers")
@click.option("--processes", required=False, default=8)
@click.option("--path-to-env", required=False, default=".dev-env")
def minio(
    path_to_raw_data: str,
    bucket: str,
    processes: int,
    path_to_env: str,
):
    """
    Populate Minio with data.

    Args:
        path_to_raw_data (str): The path to the raw data.
        bucket (str): The name of the Minio bucket.
        processes (int): The number of processes to use for populating Minio.
        path_to_env (str): The path to the environment file.

    Returns:
        The result of the `main` function from `rag.datasets.populate_minio`.

    """
    from rag.datasets.populate_minio import main

    return main(
        path_to_raw_data=path_to_raw_data,
        bucket=bucket,
        processes=processes,
        path_to_env=path_to_env,
    )


@cli.command()
@click.option("--path-to-data", required=True, default="s3a://papers/*.json")
@click.option("--path-to-env", required=False, default=".dev-env")
def preprocessing(path_to_data: str, path_to_env: str):
    """
    Preprocess data.

    This function performs data preprocessing using the
    specified data and environment paths.

    Args:
        path_to_data (str): The path to the data.
        path_to_env (str): The path to the environment.

    Returns:
        The result of the preprocessing.

    """
    from rag.processor.preprocessing import main

    return main(path_to_data=path_to_data, path_to_env=path_to_env)


@cli.command()
@click.option(
    "--model", required=False, default="sentence-transformers/all-MiniLM-L6-v2"
)
@click.option("--source", required=False, default=None)
@click.option("--path-to-env", required=False, default=".dev-env")
def embeddings(path_to_env: str, source: str, model: str):
    """
    Calculate embeddings.

    This function calculates embeddings using the specified model.

    Args:
        path_to_env (str): The path to the environment.
        model (str): The name of the model to use.

    Returns:
        The result of the `main` function from the `embeddings` module.

    """
    from rag.processor.embeddings import main

    return main(model=model, source=source, path_to_env=path_to_env)


@cli.command()
@click.option("--text", required=True)
@click.option(
    "--model", required=False, default="sentence-transformers/all-MiniLM-L6-v2"
)
@click.option("--num-docs", required=False, default=3)
@click.option("--query", required=False, default=None)
@click.option("--path-to-env", required=False, default=".dev-env")
def most_similar_docs(
    text: str, model: str, num_docs: int, query: str | None, path_to_env: str
):
    """Get most similar documents.

    Args:
        text (str): The input text.
        model (str): The name of the model to use.
        num_docs (int): The number of most similar documents to retrieve.
        query (str): The query to use for helping to retrieve the most similar documents.
        path_to_env (str): The path to the environment.

    Returns:
        List[str]: A list of most similar documents.
    """
    from rag.processor.most_similar_docs import main

    return main(
        text=text, model=model, num_docs=num_docs, query=query, path_to_env=path_to_env
    )


@cli.command()
@click.option("--text", required=True)
@click.option(
    "--embeddings-model",
    required=False,
    default="sentence-transformers/all-MiniLM-L6-v2",
)
@click.option(
    "--qa-model",
    required=False,
    default="distilbert/distilbert-base-cased-distilled-squad",
)
@click.option("--num-docs", required=False, default=1)
@click.option("--query", required=False, default=None)
@click.option("--path-to-env", required=False, default=".dev-env")
def qa(
    text: str, embeddings_model: str, qa_model: str, num_docs: int, query: str | None, path_to_env: str
):
    """
    Question & Answers.

    This function takes in a text, embeddings model, QA model, number of documents, and path to environment.
    It uses the `get_most_similar_docs` function to retrieve the most similar document IDs based on the input text.
    Then, it uses the `main` function from the `qa` module to generate an answer and context based on the input text and retrieved document IDs.
    Finally, it prints the answer, document IDs, certainty score, and full text for debugging purposes.

    Args:
        text (str): The input text containing the question.
        embeddings_model (str): The path to the embeddings model.
        qa_model (str): The path to the QA model.
        query (str): The query to use for helping to retrieve the most similar documents.
        num_docs (int): The number of most similar documents to retrieve.
        path_to_env (str): The path to the environment.

    Returns:
        dict: A dictionary containing the answer, document IDs, and certainty score.

    """
    from rag.processor.most_similar_docs import main as get_most_similar_docs
    from rag.processor.qa import main

    ids = get_most_similar_docs(
        text=text,
        model=embeddings_model,
        num_docs=num_docs,
        query=query,
        path_to_env=path_to_env,
    )
    answer, context = main(
        text=text,
        ids=ids,
        qa_model=qa_model,
        path_to_env=path_to_env,
    )
    click.echo(
        f"Answer: {answer['answer']}.\n"
        f"Document ID(s): {ids}.\n"
        f"Certainty: {answer['score']:.2f}\n\n\n"
        f"Full Text (for debugging): {context}"
    )
    return answer


if __name__ == "__main__":
    cli()

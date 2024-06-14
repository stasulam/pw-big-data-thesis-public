"""Embeddings."""

import logging

import pandas as pd
import pyspark.sql.functions as F
from dotenv import dotenv_values
from transformers import pipeline

from rag.clients import setup_spark_session


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def setup_feature_extractor(
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> pipeline:
    """
    Setup feature extractor.

    Args:
        model (str): The name or path of the pre-trained model to use.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        pipeline: The initialized feature extractor pipeline.

    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model)


def define_calculate_embeddings_udf(
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Defines a pandas UDF that calculates embeddings for a given input series of sentences.

    Args:
        model (str): The name of the pre-trained model to use for feature extraction.
            Default is "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        pandas UDF: A pandas UDF that takes a series of sentences as input
            and returns a DataFrame with calculated embeddings.
    """
    feature_extractor = setup_feature_extractor(model=model)

    @F.pandas_udf("array<array<double>>")
    def calculate_embeddings(inputs: pd.Series) -> pd.DataFrame:
        return inputs.apply(
            lambda sentences: [
                feature_extractor.encode(sentence) for sentence in sentences
            ]
        )

    return calculate_embeddings


def main(
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    source: str | None = None,
    path_to_env: str = ".dev-env",
) -> None:
    """
    Main function for processing and calculating embeddings.

    Args:
        path_to_env (str): Path to the environment file. Defaults to ".env".
        model (str): Name of the pre-trained model to use for embeddings.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    """
    try:
        ENV = dotenv_values(path_to_env)
    except Exception:
        LOGGER.warn("Using the default environment variables.")
        ENV = dict()  # the default values will be used.
    LOGGER.info("Setting up Spark...")
    spark = setup_spark_session(path_to_env=path_to_env)
    LOGGER.info("Reading data from MongoDB...")
    data = (
        spark.read.format("mongodb")
        .option("database", ENV.get("MONGO_DATABASE", "arxiv"))
        .option("collection", ENV.get("MONGO_COLLECTION", "sentences"))
    )
    if source:
        data = data.option(
            "aggregation.pipeline", f"{{ $match: {{ source: '{source}' }} }}"
        ).load()
    else:
        data = data.load()
    LOGGER.info("Calculating embeddings...")
    calculate_embeddings = define_calculate_embeddings_udf(model=model)
    data = data.withColumn("embeddings", calculate_embeddings(F.col("sentences")))
    (
        data.write.format("mongodb")
        .option("database", ENV.get("MONGO_DATABASE", "arxiv"))
        .option("collection", ENV.get("MONGO_COLLECTION", "sentences"))
        .option("replaceDocument", "false")
        .mode("append")
        .save()
    )

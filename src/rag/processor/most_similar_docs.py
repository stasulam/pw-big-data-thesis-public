"""Get similar documents to a given document."""

import logging
from typing import Union

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from dotenv import dotenv_values
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType

from rag.clients import setup_spark_session
from rag.processor.embeddings import setup_feature_extractor


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def get_basic_db_info(data: DataFrame) -> dict[str, Union[int, pd.DataFrame]]:
    """
    Retrieves basic information about the database.

    Args:
        data (DataFrame): The input DataFrame representing the database.

    Returns:
        dict[str, Union[int, pd.DataFrame]]: A dictionary containing the following information:
            - 'number_of_docs_in_db' (int): The total number of documents in the database.
            - 'example_docs' (pd.DataFrame): A DataFrame containing a sample of example documents
              from the database, including the '_id' and 'full_text' columns.
    """
    example_rows = data.limit(10).collect()
    return {
        "number_of_docs_in_db": data.count(),
        "example_docs": pd.DataFrame([x.asDict() for x in example_rows])[
            ["_id", "full_text"]
        ],
    }


def process_query(
    text: str, model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Process a query text using a specified model and return the encoded features.

    Args:
        text (str): The query text to be processed.
        model (str, optional): The name of the model to be used for feature extraction.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        np.ndarray: The encoded features of the query text.

    """
    feature_extractor = setup_feature_extractor(model=model)
    return feature_extractor.encode(text)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.

    Args:
        v1 (np.ndarray): The first vector.
        v2 (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.

    """
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def define_udf_cosine_similarity(query: np.ndarray) -> F.udf:
    """Cosine similarity UDF.

    Defines a user-defined function (UDF) for calculating
    cosine similarity between a vector and a query.

    Args:
        query (np.ndarray): The query vector.

    Returns:
        F.udf: The UDF that calculates cosine similarity between a vector and the query.
    """
    return F.udf(lambda v: cosine_similarity(v, query), FloatType())


def get_most_similar_documents(
    text: str,
    data: DataFrame,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    num_docs: int = 3,
) -> list[str]:
    """
    Retrieves the most similar documents based on a given text query.

    Args:
        text (str): The query text.
        data (DataFrame): The DataFrame containing the documents.
        model (str, optional): The name of the pre-trained model
            to use for embedding. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        num_docs (int, optional): The number of most similar documents to retrieve.
            Defaults to 3.

    Returns:
        list[str]: A list of document IDs representing the most similar documents.

    """
    LOGGER.info("Processing query...")
    query = process_query(text, model=model)
    LOGGER.info("Calculating cosine similarity...")
    data = data.withColumn(
        "cosine_similarity", define_udf_cosine_similarity(query)(F.col("embeddings"))
    )
    LOGGER.info("Getting most similar documents...")
    most_similar = (
        data.groupby(["_id"])
        .agg(F.max("cosine_similarity").alias("max_cosine_similarity"))
        .sort(F.col("max_cosine_similarity").desc())
        .select("_id")
        .limit(num_docs)
        .collect()
    )
    LOGGER.info("Most similar documents:")
    LOGGER.info([row["_id"] for row in most_similar])
    return [row["_id"] for row in most_similar]


def main(
    text: str,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    num_docs: int = 3,
    path_to_env: str = ".dev-env",
) -> list[str]:
    """
    Process the most similar documents based on the given text.

    Args:
        text (str): The input text to find similar documents for.
        model (str, optional): The name or path of the pre-trained model to use for sentence embeddings.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        num_docs (int, optional): The number of most similar documents to retrieve.
            Defaults to 3.
        path_to_env (str, optional): The path to the environment file.
            Defaults to ".env".

    Returns:
        list[str]: A list of most similar documents based on the given text.

    """
    try:
        ENV = dotenv_values(path_to_env)
    except Exception:
        LOGGER.warn("Using the default environment variables.")
        ENV = dict()  # the default values will be used.
    LOGGER.info("Setting up Spark session...")
    spark = setup_spark_session()
    LOGGER.info("Reading data from MongoDB")
    data = (
        spark.read.format("mongodb")
        .option("database", ENV.get("MONGO_DATABASE", "arxiv"))
        .option("collection", ENV.get("MONGO_COLLECTION", "sentences"))
        .load()
    )
    data = data.select(F.col("_id"), F.col("embeddings")).withColumn(
        "embeddings", F.explode(F.col("embeddings"))
    )
    return get_most_similar_documents(
        text=text, data=data, model=model, num_docs=num_docs
    )

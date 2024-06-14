"""Preprocessing."""

import logging
import re
from pathlib import Path

import pyspark.sql.functions as F
from dotenv import dotenv_values
from pyspark.sql.types import ArrayType, StringType

from rag.clients import setup_spark_session


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def split_to_senteces(text: str) -> list[str]:
    """Split text into sentences.

    Args:
        text (str): The input text to be split into sentences.

    Returns:
        list[str]: A list of sentences extracted from the input text.

    """
    sentences = list(map(str.strip, re.split(r"(?<=[.!?]) +", text)))
    # Remove new lines inside sentences.
    return [sentence.replace(r"\n", " ").replace(r"\r", " ") for sentence in sentences]


udf_split_to_sentences = F.udf(split_to_senteces, ArrayType(StringType()))


def main(path_to_data: str, path_to_env: str) -> None:
    """
    Main function for preprocessing.

    Args:
        path_to_data (str): Path to the data file.
        path_to_env (str): Path to the environment file.

    """
    try:
        ENV = dotenv_values(path_to_env)
    except Exception:
        LOGGER.warn("Using the default environment variables.")
        ENV = dict()  # Use default values
    LOGGER.info("Setting up Spark...")
    spark = setup_spark_session(path_to_env)
    LOGGER.info("Preprocessing...")
    LOGGER.info("Reading data from Minio...")
    data = spark.read.json(path_to_data)
    LOGGER.info("Splitting text to sentences...")
    data = data.withColumn("sentences", udf_split_to_sentences(F.col("abstract")))
    data = data.withColumn("source", F.lit(str(Path(path_to_data).parent)))
    data = data.withColumnRenamed("id", "_id")
    LOGGER.info("Writing data to MongoDB...")
    (
        data.select(
            F.col("_id"),
            F.col("abstract").alias("full_text"),
            F.col("sentences"),
            F.col("source"),
            F.col("timestamp"),
        )
        .write.format("mongodb")
        .option("database", ENV.get("MONGO_DATABASE", "arxiv"))
        .option("collection", ENV.get("MONGO_COLLECTION", "sentences"))
        .mode("append")
        .save()
    )

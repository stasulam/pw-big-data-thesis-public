"""Clients."""

from dotenv import dotenv_values
from minio import Minio
from pymongo import MongoClient
from pyspark.sql import SparkSession


def setup_minio_client(path_to_env: str = ".dev-env") -> Minio:
    env = dotenv_values(path_to_env)
    return Minio(
        env["MINIO_ENDPOINT"],
        access_key=env["MINIO_ACCESS_KEY"],
        secret_key=env["MINIO_SECRET_KEY"],
        secure=False,
    )


def setup_mongo_client(path_to_env: str = ".dev-env") -> MongoClient:
    env = dotenv_values(path_to_env)
    return MongoClient(env["MONGO_URI"])


def setup_spark_session(path_to_env: str = ".dev-env") -> SparkSession:
    """Build Spark session."""
    # TODO(lukasz): This config will be moved to the configuration file.
    spark = (
        SparkSession.builder.master("spark://spark:7077")
        .appName("Preprocessing")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.mongodb.spark:mongo-spark-connector_2.12:10.3.0",
        )
        .config("spark.hadoop.fs.s3a.access.key", "minio")
        .config("spark.hadoop.fs.s3a.secret.key", "miniominio")
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "1")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
        .config("spark.hadoop.fs.s3a.connection.timeout", "50000")
        .config("spark.mongodb.read.connection.uri", "mongodb://mongo:27017")
        .config("spark.mongodb.write.connection.uri", "mongodb://mongo:27017")
        .config(
            "spark.eventLog.gcMetrics.youngGenerationGarbageCollectors",
            "G1 Young Generation, ParNew",
        )
        .config(
            "spark.eventLog.gcMetrics.oldGenerationGarbageCollectors",
            "G1 Old Generation, ConcurrentMarkSweep",
        )
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "1g")
        .config("spark.executor.instances", 2)
        .config("spark.executor.cores", 1)
        .config("spark.dynamicAllocation.minExecutors", 2)
        .config("spark.dynamicAllocation.maxExecutors", 4)
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "1g")
        .getOrCreate()
    )
    return spark

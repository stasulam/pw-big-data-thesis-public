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
    env = dotenv_values(path_to_env)
    spark = (
        SparkSession.builder.master(env.get("SPARK_MASTER", "spark://spark:7077"))
        .appName("RAG")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.mongodb.spark:mongo-spark-connector_2.12:10.3.0",
        )
        .config("spark.hadoop.fs.s3a.access.key", env.get("MINIO_ACCESS_KEY", "minio"))
        .config(
            "spark.hadoop.fs.s3a.secret.key", env.get("MINIO_SECRET_KEY", "miniominio")
        )
        .config(
            "spark.hadoop.fs.s3a.endpoint",
            env.get("MINIO_ENDPOINT", "http://minio:9000"),
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "1")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
        .config("spark.hadoop.fs.s3a.connection.timeout", "50000")
        .config(
            "spark.mongodb.read.connection.uri",
            env.get("MONGO_URI", "mongodb://mongo:27017"),
        )
        .config(
            "spark.mongodb.write.connection.uri",
            env.get("MONGO_URI", "mongodb://mongo:27017"),
        )
        .config(
            "spark.eventLog.gcMetrics.youngGenerationGarbageCollectors",
            "G1 Young Generation, ParNew",
        )
        .config(
            "spark.eventLog.gcMetrics.oldGenerationGarbageCollectors",
            "G1 Old Generation, ConcurrentMarkSweep",
        )
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.driver.memory", env.get("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.executor.memory", env.get("SPARK.EXECUTOR_MEMORY", "1g"))
        .config("spark.executor.instances", env.get("SPARK_EXECUTOR_INSTANCES", 2))
        .config("spark.executor.cores", env.get("SPARK_EXECUTOR_CORES", 1))
        .config(
            "spark.dynamicAllocation.minExecutors", env.get("SPARK.MIN.EXECUTORS", 2)
        )
        .config(
            "spark.dynamicAllocation.maxExecutors", env.get("SPARK.MAX.EXECUTORS", 4)
        )
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", env.get("SPARK.OFFHEAP.SIZE", "1g"))
        .getOrCreate()
    )
    return spark

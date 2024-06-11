import streamlit as st
from pyspark.sql import functions as F

from rag.clients import setup_spark_session


st.set_page_config(
    page_title="KYD",
    page_icon="👋",
)

st.write("# K-Y-D: Know Your Database! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    K-Y-D: Know Your Database.
"""
)

with st.spinner("Setting up the application..."):
    spark = setup_spark_session()
    full_df = (
        spark.read.format("mongodb")
        .option("database", "arxiv")
        .option("collection", "sentences")
        .load()
    )
    data = full_df.select(F.col("_id"), F.col("embeddings")).withColumn(
        "embeddings", F.explode(F.col("embeddings"))
    )

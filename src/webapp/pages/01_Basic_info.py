"""Basic Info."""

import streamlit as st

from rag.processor.most_similar_docs import get_basic_db_info
from webapp.Hello import full_df

# App.
st.title("K-Y-D: Basic Info")

with st.spinner("Getting the basic info..."):
    st.header("Basic Db info")
    st.write("There you can find some basic information regarding your database")
    collected_info = get_basic_db_info(data=full_df)

    st.write("Number of documents of in your database")
    st.write(collected_info["number_of_docs_in_db"])

    st.write("Here are some sample documents")
    st.dataframe(collected_info["example_docs"])

"""LLM."""

import time

import streamlit as st
from llama_cpp import Llama
from pyspark.sql import functions as F

from webapp.Hello import data, full_df


MODEL_PATH = "../../models/Phi-3-mini-4k-instruct-q4.gguf"

MODEL = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=64,  # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
)

PROMPT = """
Context: ```
{context}
```
Given the context inside ``` solve the following taks: {task}.
If the context is not enough, try to solve the task with the
knowledge you have. But inform the user that the context is not
enough to solve the task.
"""


def get_most_similar_docs(text: str, num_docs: int) -> str:
    """Get most similar documents."""
    from rag.processor.most_similar_docs import get_most_similar_documents

    ids = get_most_similar_documents(
        text=text,
        data=data,
        num_docs=num_docs,
    )
    texts = full_df.filter(F.col("_id").isin(ids)).select("full_text").collect()
    context = "\n".join([text["full_text"] for text in texts])
    return ids, context


def qa(text: str, num_docs: int = 3) -> str:
    """Question & Answers."""
    ids, context = get_most_similar_docs(text=text, num_docs=num_docs)
    prompt = PROMPT.format(context=context, task=text)
    output = MODEL(
        f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
        max_tokens=256,  # Generate up to 256 tokens
        stop=["<|end|>"],
        echo=False,  # Whether to echo the prompt
    )
    return ids, output


# App.
st.title("K-Y-D: LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        ids, output = qa(text=prompt)

        def stream(output):
            for token in output["choices"][0]["text"].split(" "):
                yield token + " "
                time.sleep(0.02)

        st.write_stream(stream(output))
        st.markdown(f"Document ID(s): {ids}")
        st.session_state.messages.append(
            {"role": "assistant", "content": " ".join(stream(output))}
        )

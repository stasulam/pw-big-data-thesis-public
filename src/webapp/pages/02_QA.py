"""Chat."""

import streamlit as st


def qa(text: str) -> str:
    """Question & Answers."""
    from rag.processor.most_similar_docs import main as get_most_similar_docs
    from rag.processor.qa import main

    ids = get_most_similar_docs(text=text, num_docs=1, path_to_env=".app-env")
    answer, full_text = main(text=text, ids=ids, path_to_env=".app-env")
    if (score := answer["score"]) < 0.5:
        return (
            "Sorry, I cannot answer this question, becasue I am not certain enough.\n"
            "This would be the answer if I was more certain:\n"
            f"{answer['answer']}\n (certainty: {score:.2f}, docs: {ids})\n"
        )
    answer = (
        "### Answer\n"
        "The answer to your question is:\n"
        f"{answer['answer']}\n\n\n"
        "### Metadata\n"
        "Metadata of the answer:\n"
        f"  - Document ID(s): {ids}\n"
        f"  - Certainty: {score:.2f}\n"
        "### Context\n"
        "Here is the context which was used to find the answer:\n"
        f"{full_text}"
    )
    return answer


# App.
st.title("K-Y-D: Chat")

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
        answer = qa(text=prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

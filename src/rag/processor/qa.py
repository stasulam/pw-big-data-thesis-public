"""Question & Answers."""

from typing import Any

from transformers import pipeline

from rag.clients import setup_mongo_client


def setup_qa_model(model: str = "distilbert/distilbert-base-cased-distilled-squad"):
    """
    Setup QA model.

    Args:
        model (str): The name or path of the pre-trained QA model to use.
            Defaults to "distilbert/distilbert-base-cased-distilled-squad".

    Returns:
        QA model: The initialized question-answering model.

    """
    return pipeline("question-answering", model=model)


def main(
    text: str,
    ids: list[dict[str, Any]],
    qa_model: str = "distilbert/distilbert-base-cased-distilled-squad",
    path_to_env: str = ".dev-env",
) -> tuple[dict[str, Any], str]:
    """Main function for performing question answering.

    Args:
        text (str): The question to be answered.
        ids (list[dict[str, Any]]): List of document IDs to retrieve the context from.
        qa_model (str, optional): The name or path of the question answering model to use.
            Defaults to "distilbert/distilbert-base-cased-distilled-squad".
        path_to_env (str, optional): The path to the environment file. Defaults to ".dev-env".

    Returns:
        tuple[dict[str, Any], str]: A tuple containing the answer and the context.

    """
    client = setup_mongo_client(path_to_env)
    db = client.arxiv
    collection = db.sentences
    context = " ".join(
        [
            doc.get("full_text", "").strip()
            for doc in collection.find({"_id": {"$in": ids}})
        ]
    )
    qa_model_pipeline = setup_qa_model(model=qa_model)
    return qa_model_pipeline(question=text, context=context), context

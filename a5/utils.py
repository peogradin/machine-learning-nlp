from typing import Any
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState


class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.last_doc_ids: list[int] | None = None
        self.last_docs: list[Document] | None = None

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1] # get the user input query
        retrieved_docs = self.vector_store.similarity_search(last_message.text)  # search for documents

        # docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)  
        unique = {}
        for doc in retrieved_docs:
            doc_id = doc.metadata.get("id")
            if doc_id not in unique:
                unique[doc_id] = doc
        unique_docs = list(unique.values())
        docs_content = "\n\n".join(doc.page_content for doc in unique_docs)

        self.last_doc_ids = [doc.metadata.get("id") for doc in unique_docs]
        self.last_docs = unique_docs

        augmented_message_content = (
            f"question:\n"
            f"{last_message.text}\n\n"
            "Use the following context to answer:\n"
            f"{docs_content}\n\n"
            "You are grading this output as yes or no.\n\n"
            "RULES:\n"
            "- Output must be exactly one word: 'yes' or 'no'\n"
            "- Do not add any explanation\n"
            "- Do not add any punctuation\n"
            "- If you add anything else, the answer will be considered wrong.\n\n"
            "Answer (yes/no):"
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }
# %%
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from transformers import pipeline
from typing import Any
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents import create_agent


# %%
tmp_data = pd.read_json("ori_pqal.json").T
# some labels have been defined as "maybe", only keep the yes/no answers
tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

documents = pd.DataFrame({"abstract": tmp_data.apply(lambda row: (" ").join(row.CONTEXTS+[row.LONG_ANSWER]), axis=1),
             "year": tmp_data.YEAR})
questions = pd.DataFrame({"question": tmp_data.QUESTION,
             "year": tmp_data.YEAR,
             "gold_label": tmp_data.final_decision,
             "gold_context": tmp_data.LONG_ANSWER,
             "gold_document_id": documents.index})

questions.iloc[0].question

# %%
documents.iloc[0].abstract
# %%
model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
print(f"Loading model {model_id}...")

model = HuggingFacePipeline.from_model_id(
    model_id,
    task="text-generation",
    pipeline_kwargs={"return_full_text": False},
)
prompt = "What is the capital of France?"
print("Prompt:", prompt)
answer = model.invoke(prompt)
print("Answer:", answer)

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)

metadatas = [{"id": idx} for idx in documents.index]
texts = text_splitter.create_documents(texts=documents.abstract.tolist(), metadatas=metadatas)

splits = text_splitter.split_documents(texts)

# %%
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
len(embeddings.embed_query("Hello world"))
# %%

# Initialize the Vector Store
vector_store = Chroma(
    collection_name="assignment5",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add splits
document_ids = vector_store.add_documents(documents=splits[:50])

print(f"Success! Added {len(document_ids)} chunks")

results = vector_store.similarity_search_with_score(
    "What is programmed cell death?", k=3
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

# %%

class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        retrieved_docs = vector_store.similarity_search(last_message.text)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        augmented_message_content = (
            f"{last_message.text}\n\n"
            "Use the following context to answer the query. Only answer with yes or no!\n"
            f"{docs_content}"
            "\n\n Answer (yes/no):"
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }

agent = create_agent(
    model,
    tools=[],
    middleware=[RetrieveDocumentsMiddleware()],
)
# %%
for step in agent.stream(
    {"messages": [{"role": "user", "content": "Is programmed cell death the regulated death of cells?" }]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

# %%

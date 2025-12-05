# %%
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from transformers import pipeline

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
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
messages = [
    {"role": "user", "content": "What is the capital of France?"},
]
pipe(messages)
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
embeddings.embed_query("Hello world").shape
# %%

# Initialize the Vector Store
vector_store = Chroma(
    collection_name="assignment5",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add splits
document_ids = vector_store.add_documents(documents=splits[:50])

print(f"Success! Added {len(document_ids)} chunks using Gemini Embeddings.")

results = vector_store.similarity_search_with_score(
    "What is programmed cell death?", k=3
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

# %%

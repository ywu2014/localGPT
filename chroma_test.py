from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS
)

embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})

vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
)

query = "什么是工作日历?"
docs = vectorstore.similarity_search(query, k=3)
for doc in docs:
    # print(doc.page_content)
    print(doc)
    print('-'*100)
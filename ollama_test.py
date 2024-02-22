from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from langchain.llms import Ollama

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import textwrap
from langchain import PromptTemplate
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

ollama_host="localhost"
ollama_port="11434"
ollama_model="llama2-chinese:latest"

def simple():
    llm = Ollama(base_url=f"http://{ollama_host}:{ollama_port}", 
                 model=ollama_model,
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),)
    
    while True:
        query = input("\n\nEnter a Query: ")
        llm(query)

def load_pdf_data(file_path):
    # Creating a PyMuPDFLoader object with file_path
    loader = PyMuPDFLoader(file_path=file_path)
    
    # loading the PDF file
    docs = loader.load()
    
    # returning the loaded document
    return docs


# Responsible for splitting the documents into several chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    
    # Initializing the RecursiveCharacterTextSplitter with
    # chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(documents=documents)
    
    # returning the document chunks
    return chunks


# function for loading the embedding model
def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'}, # here we will run the model with CPU only
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding # keep True to compute cosine similarity
        }
    )

'''
# Function for creating embeddings using FAISS
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    # Creating the embeddings using FAISS
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    
    # Saving the model in current directory
    vectorstore.save_local(storing_path)
    
    # returning the vectorstore
    return vectorstore
'''

def create_embeddings(embedding_model):
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
        client_settings=CHROMA_SETTINGS
    )
    # retriever = db.as_retriever()
    return db

# Creating the chain for Question Answering
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True, # including source documents in output
        chain_type_kwargs={'prompt': prompt}, # customizing the prompt
        # callbacks=[StdOutCallbackHandler()],
        verbose=True
    )
    
def get_response(query, chain):
    # Getting response from chain
    response = chain({'query': query})
    
    # Wrapping the text for better output in Jupyter Notebook
    wrapped_text = textwrap.fill(response['result'], width=100)
    print(wrapped_text)

def complex():
    # Loading orca-mini from Ollama
    llm = Ollama(base_url=f"http://{ollama_host}:{ollama_port}", 
                 model=ollama_model,
                 verbose=True,
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),)
    
    # Loading the Embedding Model
    # embed = load_embedding_model(model_path=EMBEDDING_MODEL_NAME)
    embed = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})

    # loading and splitting the documents
    # docs = load_pdf_data(file_path="/Users/ywu/Downloads/2017 雪佛兰全新迈锐宝保修及保养手册.pdf")
    # documents = split_docs(documents=docs)
    # print(documents)
    
    # creating vectorstore
    vectorstore = create_embeddings(embed)

    # converting vectorstore to a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    template = """
    ### System:
    You are an respectful and honest assistant. You have to answer the user's questions using only the context \
    provided to you. If you don't know the answer, just say you don't know. Don't try to make up an answer.

    ### Context:
    {context}

    ### User:
    {question}

    ### Response:
    """

    prompt = PromptTemplate.from_template(template)

    # Creating the prompt from the template which we created before
    prompt = PromptTemplate.from_template(template)

    # Creating the chain
    chain = load_qa_chain(retriever, llm, prompt)

    while True:
        query = input("\n\nEnter a Query: ")
        get_response(query, chain)
    

if __name__ == "__main__":
    # simple()

    complex()
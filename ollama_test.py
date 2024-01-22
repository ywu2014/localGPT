from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

ollama_host="localhost"
ollama_port="11434"
ollama_model="llama2-chinese:latest"

if __name__ == "__main__":
    llm = Ollama(base_url=f"http://{ollama_host}:{ollama_port}", 
                 model=ollama_model,
                 callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),)
    
    while True:
        query = input("\n\nEnter a Query: ")
        llm(query)
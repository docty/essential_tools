from llama_index.core import Settings
from llama_index.llms.mistralai import MistralAI 
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)


class LLMContextAware:

    def __init__(self):
        self.llm = MistralAI(model="mistral-large-latest")
        self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en")
        setDefaultSettings()


    def setDefaultSettings(self):
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 64
        
    def llm(self):
        return self.llm
        

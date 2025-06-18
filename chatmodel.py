from llama_index.core import Settings
from llama_index.llms.mistralai import MistralAI 
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgent
import os

class LLMChat:

    def __init__(self, vector_store='storage'):
        self.llm = MistralAI(model="mistral-small-latest")
        self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en")
        self.setDefaultSettings()
        self.vector_store = vector_store


    def setDefaultSettings(self):
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 64
        
    def llms(self):
        return self.llm

    def build_index(self, files):
        documents = SimpleDirectoryReader(input_files=[files]).load_data()
        if isinstance(self.vector_store, str):
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(self.vector_store)
            return index
        else:
             index = VectorStoreIndex.from_documents(documents, storage_context=self.vector_store)
             return index
        
    def load_index(self):
        if isinstance(self.vector_store, str): 
            storage_context = StorageContext.from_defaults(persist_dir=self.vector_store)
            return load_index_from_storage(storage_context)
        else:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        return load_index_from_storage(storage_context)
    
    def load_data(self, files=None):
        if not os.path.exists(self.vector_store):
            #print("Building new index \n===========================================")
            index = self.build_index(files)
        else:
            #print("Loading existing index \n===========================================")
            index = self.load_index()
        return index
        
        
    def query_engine(self, docs):
        query_engine = docs.as_query_engine(similarity_top_k=5)
        return query_engine

     
    
    def build_agent(self, query_engine):
        query_engine_tools = [
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                name="cv",
                description="Provides information about my cv",
                ),
            ),
        ]

        agent = FunctionCallingAgent.from_tools(
            query_engine_tools,
            llm=self.llm,
            verbose=True,
            allow_parallel_tool_calls=False,
        )
        return agent
        

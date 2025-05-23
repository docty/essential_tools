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


class LLMContextAware:

    def __init__(self):
        self.llm = MistralAI(model="mistral-small-latest")
        self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en")
        self.setDefaultSettings()


    def setDefaultSettings(self):
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 64
        
    def llms(self):
        return self.llm

    def load_data(self, files):
        uber_docs = SimpleDirectoryReader(input_files=[files]).load_data()
        return uber_docs
        
    def query_engine(self, uber_docs):
        uber_index = VectorStoreIndex.from_documents(uber_docs)
        uber_query_engine = uber_index.as_query_engine(similarity_top_k=5)
        return uber_query_engine

     
    
    def build_agent(self, query_engine):
        query_engine_tools = [
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                name="uber_10k",
                description="Provides information about Uber financials for year 2021",
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
        

from langchain.memory import VectorStoreRetrieverMemory
from langchain.tools import BaseTool
from typing import Any
from utils import vect_db_init


class KnowledgeBaseTool(BaseTool):
    name: str = "knowledge_base"
    description: str = "Useful for when you need to answer questions about UAE cities, attractions, and cultural tips."

    get_tool: Any = None
    vect_db: Any = None

    def __init__(self, get_tool):
        super().__init__()
        self.get_tool = get_tool
        self.vect_db = VectorStoreRetrieverMemory(
            retriever=vect_db_init(get_tool),
            memory_key="base_knowledge",
            return_messages=True
        )
    
    def _run(self, user_input: str) -> str:
        try:
            return self.vect_db.load_memory_variables({"input": user_input})['base_knowledge']
        except Exception as e:
            return f"Error retrieving knowledge base information: {str(e)}"
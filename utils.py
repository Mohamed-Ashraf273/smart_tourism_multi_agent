import json

from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def json_chunker(docs):
    if isinstance(docs, str):
        try:
            data = json.loads(docs)
        except json.JSONDecodeError:
            return [Document(page_content=docs)]

    chunks = []
    for section_name, section_data in data.items():
        section_text = json.dumps(section_data, indent=2)
        chunks.append(Document(
            page_content=f"{section_name.upper()}:\n{section_text}",
            metadata={"section": section_name}
        ))
    return chunks

def vect_db_init(get_tool):
    docs = get_tool.run('https://sprintscdn-fnh2cugtb8a4deba.z02.azurefd.net/production/files/1751977720.json')
    chunks = json_chunker(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever_memory = vector_store.as_retriever(search_kwargs={"k": 2})
    return retriever_memory
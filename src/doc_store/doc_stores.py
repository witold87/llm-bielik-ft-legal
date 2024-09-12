import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


class BaseDocStore:

    def __init__(self, text: list[str]):
        self.text = text

    def save(self):
        ...


class FaissDocStore(BaseDocStore):
    def __init__(self, text: list[str]):
        super().__init__(text)
        self.index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))


    def save(self):

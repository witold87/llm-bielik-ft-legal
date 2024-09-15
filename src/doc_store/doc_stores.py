import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import uuid
from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings


embedding_model = SentenceTransformer('ipipan/silver-retriever-base-v1.1')
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

class BaseDocStore:

    def __init__(self, text: list[str]):
        self.text = text

    def save(self):
        ...


class FaissDocStore(BaseDocStore):
    def __init__(self, text: list[str]):
        super().__init__(text)
        self.embeddings = self._create_embeddings()
        self.index = self.create_index()# faiss.IndexFlatL2(len(self.embeddings))
        self.vector_store = None #self._setup_vector_store()


    def create_index(self):
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        print(self.embeddings.shape)
        index.add(self.embeddings)
        print(f'Elements in index: {index.ntotal}')
        #faiss.write_index(index, 'index_test')
        #index = faiss.read_index('index_test')
        return index

    # def _setup_vector_store(self):
    #     vector_store = FAISS(
    #         embedding_function=embedding_model,
    #         index=self.index,
    #         docstore=InMemoryDocstore(),
    #         index_to_docstore_id={},
    #     )
    #     return vector_store

    def _create_embeddings(self):
        embeddings = embedding_model.encode(self.text)
        return embeddings

    def save(self) -> None:

        docs = []
        for item in self.text:
            doc = Document(page_content=item, metadata={})
            docs.append(doc)

        uuids = [str(uuid.uuid4()) for _ in range(len(docs))]
        self.vector_store.add_documents(documents=docs, ids=uuids)

    def index_search(self, query:str, k: int=3):
        query_vector = embedding_model.encode([query])
        top_k = self.index.search(query_vector, k)  # top3 only
        return [self.text[_id] for _id in top_k[1].tolist()[0]]
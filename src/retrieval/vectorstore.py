from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List
from src.retrieval.embeddings import get_embeddings_model

class ChromaVectorStore:
    def __init__(self, persist_directory: str = "data/processed/chroma_db"):
        self.persist_directory = persist_directory
        # Usamos o mesmo modelo de embedding da OpenAI
        self.embeddings_model = get_embeddings_model()

    def create_vectorstore(self, chunks: List[Document]):
        """Cria o banco vetorial a partir dos chunks e o salva no disco."""
        print("Criando e populando o banco de dados vetorial...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"Banco vetorial salvo em: {self.persist_directory}")
        return vectorstore

    def load_vectorstore(self):
        """Carrega o banco vetorial existente do disco."""
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return vectorstore
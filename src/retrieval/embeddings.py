from langchain_openai import OpenAIEmbeddings
import os

def get_embeddings_model():
    """
    Factory function que fornece a Fonte Única de Verdade para os embeddings.
    Garante que a Ingestão (Chunking), o VectorStore e a Aplicação Final 
    usem exatamente a mesma matemática para vetorização.
    """
    
    return OpenAIEmbeddings(model="text-embedding-3-small")
import os
import sys
from dotenv import load_dotenv, find_dotenv

# 1. CARREGAMENTO DE CREDENCIAIS
print("Procurando variáveis de ambiente...")
load_dotenv(find_dotenv())
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("Erro: OPENAI_API_KEY não encontrada.")
print("Variáveis de ambiente carregadas com sucesso!")

# 2. IMPORTS
from src.data_ingestion.download_data import download_pdf
from src.data_ingestion.parser import MeditacoesParser
from src.data_ingestion.chunking import AphorismChunker
from src.retrieval.vectorstore import ChromaVectorStore

def main():
    print("\nIniciando Pipeline de Ingestão de Dados...")
    
    url_livro = "https://masculinistaopressoroficial.wordpress.com/wp-content/uploads/2017/06/meditac3a7c3b5es-marco-aurc3a9lio.pdf"
    caminho_pdf = "data/raw/meditacoes.pdf"
    download_pdf(url_livro, caminho_pdf)
    
    print("\nExtraindo e limpando o texto do PDF...")
    parser = MeditacoesParser(caminho_pdf)
    documento_limpo = parser.get_cleaned_text()
    
    print("\nIniciando Chunking (Estrutural + Semântico)...")
    print("Aviso: Esta etapa se conectará à OpenAI e pode levar alguns segundos.")
    chunker = AphorismChunker()
    chunks = chunker.split_text(documento_limpo)
    
    print(f"Sucesso! O livro foi dividido em {len(chunks)} fragmentos semânticos.")
    
    print("\nConfigurando Banco Vetorial...")
    vector_store_manager = ChromaVectorStore()
    vector_store_manager.create_vectorstore(chunks)

if __name__ == "__main__":
    main()
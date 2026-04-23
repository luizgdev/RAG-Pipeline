from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

def get_reranker_compressor(top_n=4):
    """
    Cria um compressor de Reranking usando um modelo Cross-Encoder.
    Ele vai pegar N documentos do banco vetorial e retornar apenas os 'top_n' melhores,
    lendo o contexto da pergunta e do documento juntos.
    """
    print("Inicializando o modelo de Reranking Multilíngue (BGE-M3)...")
    
    # Modelo da BAAI, fluente em português
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    
    # O compressor é a "capa" que o LangChain precisa para plugar no sistema
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    
    return compressor
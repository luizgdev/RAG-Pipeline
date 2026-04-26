from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.generation.llm_factory import get_llm
from src.retrieval.vectorstore import ChromaVectorStore
from langchain_classic.retrievers import ContextualCompressionRetriever
from src.retrieval.reranker import get_reranker_compressor
from utils.prompts import REFORMULATION_PROMPT, INTENT_PROMPT, SCHOLAR_PROMPT


# FUNÇÕES AUXILIARES DE FORMATAÇÃO
def format_docs_with_metadata(docs):
    """
    Transforma os chunks do ChromaDB em texto formatado com [N] 
    e informações de Livro/Lição nos metadados.
    """
    context = ""
    for i, doc in enumerate(docs, 1):
        secao = doc.metadata.get('secao', 'Desconhecido')
        licao = doc.metadata.get('licao', '?')
        
        context += f"[{i}] ({secao} | Lição {licao}): {doc.page_content}\n\n"
    return context

class MeditacoesRAG:
    def __init__(self):
        self.llm = get_llm()
        self.vector_store = ChromaVectorStore().load_vectorstore()
        
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        compressor = get_reranker_compressor(top_n=3)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        self.reformulation_chain = REFORMULATION_PROMPT | self.llm | StrOutputParser()
        self.intent_chain = INTENT_PROMPT | self.llm | StrOutputParser()

    def _get_standalone_question(self, inputs):
        if not inputs.get("history") or len(inputs["history"]) == 0:
            return inputs["question"]
        return self.reformulation_chain.invoke(inputs)

    def _get_raw_docs(self, inputs):
        pergunta = inputs["standalone_question"]
        
        # 1. Pede para a LLM classificar a intenção
        intencao = self.intent_chain.invoke({"question": pergunta}).strip().upper()
        
        # 2. Se for só um bate-papo, não faz busca vetorial! Devolve vazio.
        if "CHAT" in intencao:
            return []
            
        # 3. Se for sobre o livro, faz o fluxo normal pesado (ChromaDB + BGE-M3)
        return self.retriever.invoke(pergunta)

    def get_chain(self):
        chain = (
            RunnablePassthrough.assign(
                standalone_question=RunnableLambda(self._get_standalone_question)
            )
            | RunnablePassthrough.assign(
                raw_docs=RunnableLambda(self._get_raw_docs), 
            )
            | RunnablePassthrough.assign(
                context=lambda x: format_docs_with_metadata(x["raw_docs"]),
            )
            | RunnablePassthrough.assign(
                answer=(SCHOLAR_PROMPT | self.llm | StrOutputParser())
            )
        )
        return chain
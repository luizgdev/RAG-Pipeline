from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.generation.llm_factory import get_llm
from src.retrieval.vectorstore import ChromaVectorStore

# PROMPTS
REFORMULATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Dada a conversa abaixo e uma nova pergunta, reformule a pergunta para que ela seja "
               "independente e completa por si só, contendo todo o contexto necessário para uma busca. "
               "NÃO responda à pergunta, apenas reformule-a."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

SCHOLAR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Você é um assistente conversacional amigável, educado e um grande especialista na obra "Meditações" de Marco Aurélio.

    DIRETRIZES DE COMPORTAMENTO:
    1. Tom e Conversa: Seja caloroso e humano. Sinta-se livre para dar bom dia, boa tarde, se apresentar e interagir naturalmente com o usuário se ele puxar assunto.
    2. Limite de Escopo (Guarda-corpo): Se o usuário fizer perguntas fora do universo do Estoicismo ou da obra "Meditações" (ex: receitas, programação, política atual), recuse de forma gentil. Diga algo como: "Como um assistente focado em Filosofia Estoica, meu conhecimento se concentra na sabedoria de Marco Aurélio. Como posso te ajudar com o livro hoje?".
    3. Fonte da Verdade: Para responder dúvidas sobre os ensinamentos, sua ÚNICA fonte de verdade são os [Documentos Numerados] fornecidos abaixo. NUNCA invente ensinamentos que não estejam no contexto. Se a resposta não estiver lá, diga explicitamente: "Marco Aurélio não aborda esse assunto específico nos trechos que encontrei".
    4. Citações Obrigatórias: Sempre que você fizer uma afirmação baseada no livro, você DEVE ancorá-la colocando a referência no formato [N] ao final da frase.

    [Documentos Numerados]:
    {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

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

# CLASSE PRINCIPAL DO MOTOR RAG
class MeditacoesRAG:
    def __init__(self):
        # Inicializa componentes usando as factories
        self.llm = get_llm()
        self.vector_store = ChromaVectorStore().load_vectorstore()
        
        # Configuramos o retriever com o threshold
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.3}
        )
        
        # Corrente de reformulação
        self.reformulation_chain = REFORMULATION_PROMPT | self.llm | StrOutputParser()

    def _get_standalone_question(self, inputs):
        # Se não houver histórico, usa a pergunta original
        if not inputs.get("history") or len(inputs["history"]) == 0:
            return inputs["question"]
        return self.reformulation_chain.invoke(inputs)

    def get_chain(self):
        """
        Monta a corrente LCEL final que será invocada pelo Gradio.
        """
        # 1. Reformula a pergunta baseada no histórico
        # 2. Busca documentos
        # 3. Formata contexto
        # 4. Gera resposta
        
        chain = (
            RunnablePassthrough.assign(
                standalone_question=RunnableLambda(self._get_standalone_question)
            )
            | RunnablePassthrough.assign(
                raw_docs=lambda x: self.retriever.invoke(x["standalone_question"]),
            )
            | RunnablePassthrough.assign(
                context=lambda x: format_docs_with_metadata(x["raw_docs"]),
            )
            | RunnablePassthrough.assign(
                answer=(SCHOLAR_PROMPT | self.llm | StrOutputParser())
            )
        )
        return chain
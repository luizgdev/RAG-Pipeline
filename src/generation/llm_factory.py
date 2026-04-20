import os
from langchain_openai import ChatOpenAI

def get_llm(temperature: float = 0.3):
    """
    Factory function que instancia o modelo de linguagem principal.
    Centraliza a configuração do modelo, provedor e hiperparâmetros.
    
    Args:
        temperature (float): Criatividade do modelo. Para RAG, mantemos 
                             baixo (0.0 a 0.3) para evitar alucinações.
    """
    # A chave OPENAI_API_KEY já é puxada automaticamente pelo LangChain do ambiente
    
    default_model = "gpt-4o-mini"
    
    print(f"Inicializando LLM: {default_model} (Temperatura: {temperature})")
    
    return ChatOpenAI(
        model=default_model,
        temperature=temperature
    )
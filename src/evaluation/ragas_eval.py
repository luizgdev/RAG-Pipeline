import os
import sys
import warnings
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Silencia os avisos do RAGAS
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.generation.rag_chain import MeditacoesRAG
from src.generation.llm_factory import get_llm
from src.retrieval.embeddings import get_embeddings_model
from src.evaluation.metrics import perguntas_teste, ground_truths

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_entities_recall import ContextEntityRecall
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics._noise_sensitivity import NoiseSensitivity
from ragas import RunConfig

def collect_rag_predictions():
    print("🏛️ Despertando o Motor RAG para Avaliação Rápida (Dev Set)...")
    rag_engine = MeditacoesRAG()
    chain = rag_engine.get_chain()
    
    respostas_geradas = []
    contextos_recuperados = []
    
    total = len(perguntas_teste)
    print(f"\n⏳ Gerando as {total} respostas (Enviando rastros para o LangSmith)...")
    for i, pergunta in enumerate(perguntas_teste, 1):
        resultado = chain.invoke({"question": pergunta, "history": []})
        respostas_geradas.append(resultado["answer"])
        docs = [doc.page_content for doc in resultado["raw_docs"]]
        contextos_recuperados.append(docs)
        
        print(f"  [{i}/{total}] ✅ Respondida: {pergunta[:45]}...")
        
    return respostas_geradas, contextos_recuperados

def run_ragas_evaluation(respostas, contextos):
    print("\n🧪 Preparando dataset para as 8 métricas do RAGAS...")
    
    samples = []
    for q, a, c, gt in zip(perguntas_teste, respostas, contextos, ground_truths):
        samples.append(SingleTurnSample(user_input=q, response=a, retrieved_contexts=c, reference=gt))
    dataset_ragas = EvaluationDataset(samples=samples)
    
    meu_llm_base = get_llm(temperature=0.0) 
    ragas_llm = LangchainLLMWrapper(meu_llm_base)
    
    meu_embedding_base = get_embeddings_model()
    ragas_embeddings = LangchainEmbeddingsWrapper(meu_embedding_base)
    
    metricas = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextRecall(llm=ragas_llm),
        LLMContextPrecisionWithReference(llm=ragas_llm),
        ContextEntityRecall(llm=ragas_llm),
        AnswerSimilarity(embeddings=ragas_embeddings),
        FactualCorrectness(llm=ragas_llm),
        NoiseSensitivity(llm=ragas_llm)
    ]
    
    print("⏳ Iniciando o LLM Juiz do RAGAS (Ciclo Rápido)...")
    
    run_config = RunConfig(timeout=600, max_workers=1, max_retries=5)
    
    resultado = evaluate(
        dataset=dataset_ragas,
        metrics=metricas,
        run_config=run_config,
        raise_exceptions=False,
    )
    
    return resultado.to_pandas()

if __name__ == "__main__":
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("⚠️ AVISO: Chave do LangSmith não encontrada. A avaliação rodará sem rastreamento na nuvem.\n")
        
    respostas, contextos = collect_rag_predictions()
    df_resultados = run_ragas_evaluation(respostas, contextos)
    
    print("\n📊 RESULTADOS FINAIS (As 8 Métricas - Dev Set):")
    print("=" * 60)
    
    colunas_ignoradas = ['user_input', 'response', 'retrieved_contexts', 'reference']
    colunas_metricas = [col for col in df_resultados.columns if col not in colunas_ignoradas]
    
    for col in colunas_metricas:
        media = df_resultados[col].mean()
        
        if pd.isna(media):
            print(f" ❓ {col:<38} [Falhou na avaliação]")
            continue

        if col == "llm_context_precision_with_reference":
            emoji = "✅" if media >= 0.85 else ("⚠️" if media >= 0.5 else "❌")
        else:
            emoji = "✅" if media >= 0.7 else ("⚠️" if media >= 0.4 else "❌")
            
        barra = "█" * int(media * 20) + "░" * (20 - int(media * 20))
        print(f" {emoji} {col:<38} {barra} {media:.4f}")
        
    print("=" * 60)
    
    os.makedirs("src/evaluation", exist_ok=True)
    caminho_csv = "src/evaluation/resultado_reranker.csv"
    
    df_resultados.to_csv(caminho_csv, index=False)
    print(f"📝 Resultados detalhados salvos em '{caminho_csv}'!")
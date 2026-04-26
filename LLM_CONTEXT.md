# CONTEXTO DO PROJETO PARA LLMs (AI_CONTEXT.md)

Este documento foi formatado especificamente para ser lido por outras IAs (LLMs). Ele descreve a arquitetura, o domínio, o fluxo de dados e as tecnologias do projeto. Use este documento como "System Context" antes de analisar ou sugerir modificações no código.

## 1. Visão Geral do Projeto
**Nome**: Oráculo de Marco Aurélio (RAG Pipeline)
**Objetivo**: Um assistente conversacional (chatbot) educado e especializado na obra "Meditações" do imperador romano Marco Aurélio.
**Paradigma Principal**: Retrieval-Augmented Generation (RAG) utilizando chunking semântico, reranking multilíngue e roteamento de intenção.
**Interface**: Gradio (Web UI) com suporte a respostas em áudio (ElevenLabs).
**Deploy**: Conteinerizado com Docker (Debian-based), orquestrado via `docker-compose` com suporte a NVIDIA GPU pass-through para aceleração de modelos locais (Reranker).

## 2. Stack Tecnológica
*   **Linguagem**: Python 3.11
*   **Orquestração de IA**: LangChain (`langchain-core`, `langchain-openai`, `langchain-community`, `langchain-classic`)
*   **LLM (Geração)**: OpenAI `gpt-4o-mini`
*   **Embeddings**: OpenAI `text-embedding-3-small` (ou similar)
*   **Vector Database**: ChromaDB local (persistido em disco)
*   **Reranking**: HuggingFace CrossEncoder (`BAAI/bge-reranker-m3`) usando `sentence-transformers` e `torch`.
*   **Voz (TTS)**: ElevenLabs API
*   **Frontend**: Gradio 6.12+
*   **Observabilidade**: LangSmith (`LANGCHAIN_TRACING_V2=true`)
*   **Infraestrutura**: Docker (`python:3.11-slim`), `docker-compose`.

## 3. Estrutura de Diretórios
O repositório segue o padrão modularizado de MLOps:

```text
/
├── app.py                     # Entrypoint do frontend (Gradio). Orquestra UI e eventos.
├── run_ingestion.py           # Script para executar o pipeline de ETL (Ingestão do PDF).
├── Dockerfile                 # Imagem de produção Debian-based (com FFmpeg para áudio).
├── docker-compose.yml         # Orquestrador com volume binding e GPU pass-through.
├── requirements.txt           # Dependências fixadas do ambiente.
├── .env.example               # Template seguro de variáveis de ambiente.
├── data/
│   ├── raw/                   # PDF original bruto (Meditações).
│   └── processed/chroma_db/   # Banco vetorial local persistido.
├── src/
│   ├── data_ingestion/        # ETL: Download (requests), Parser (PyMuPDF) e Chunking Semântico.
│   ├── retrieval/             # Básico do RAG: Embeddings (OpenAI), VectorStore (Chroma), Reranker (BGE-M3).
│   ├── generation/            # Lógica: LLM Factory, Audio (ElevenLabs TTS), RAG Chain (LangChain).
│   └── evaluation/            # Scripts para avaliar a qualidade das respostas (ex: RAGAS).
├── utils/
│   └── prompts.py             # Repositório centralizado de todos os System Prompts e textos da UI.
└── tests/                     # Suíte de testes unitários (pytest) e scripts de calibração.
```

## 4. Padrões de Projeto e Arquitetura Avançada

### 4.1 Roteamento Semântico ("Segurança da Porta")
Para otimizar o custo de tokens e tempo de resposta, o fluxo (`src/generation/rag_chain.py`) utiliza um "Roteador de Intenção" executado via LLM. 
*   Se a intenção do usuário for `CHAT` (saudações, perguntas curtas sem relação com o livro), a busca vetorial é ignorada e a string vazia de contexto é passada.
*   Se a intenção for `LIVRO`, o fluxo pesado de `ContextualCompressionRetriever` é ativado.

### 4.2 Compressão Contextual e Reranking
O pipeline aplica a estratégia de *Retriever-Reranker*:
1.  **Base Retriever**: O ChromaDB retorna os top-K (ex: 15) chunks mais próximos utilizando *Cosine Similarity* dos embeddings da OpenAI.
2.  **Compressor (Reranker)**: Os documentos retornados passam pelo `BAAI/bge-reranker-m3` local (CrossEncoder), que cruza a query e cada chunk individualmente para reordenar a relevância. Apenas os `top_n=3` são efetivamente enviados para o LLM.

### 4.3 Histórico de Chat e Reformulação de Pergunta
Como o RAG é stateless por natureza, nós passamos o histórico do Gradio. A RAG Chain inclui um `REFORMULATION_PROMPT` que pega a última mensagem e o histórico para gerar uma `standalone_question` (pergunta independente). Isso garante que o Vector Store sempre receba consultas completas e ricas em contexto (Ex: "Qual a visão dele sobre isso?" -> "Qual a visão de Marco Aurélio sobre a morte?").

### 4.4 Prompting Acadêmico
O `SCHOLAR_PROMPT` no `utils/prompts.py` obriga o modelo a agir estritamente baseado no contexto retornado, fazendo citações obrigatórias no formato `[N]` no final das frases. O Gradio exibe um painel lateral de transparência onde as metadados (`seção` e `lição`) do chunk original são mapeadas junto ao número da citação.

### 4.5 Blindagem de Áudio Multimodal
O Gradio na versão 4+ introduziu mensagens em formato de Dicionário (para multimodais). A função `gerar_audio_manual` no `app.py` blinda a integração com a ElevenLabs para extrair o texto limpo, não importa se o Gradio envia tuplas, dicionários ou strings simples.

## 5. Regras para o LLM Assistente (Você)
Se você for solicitado a alterar este projeto:
1.  **Mantenha a Modularização**: Nunca polua o `app.py` com lógicas de negócio pesadas. Coloque abstrações na pasta `src/`.
2.  **Prompts no Utils**: Qualquer novo prompt do LangChain ou string estática grande deve ir para `utils/prompts.py`.
3.  **Ambiente Docker**: Se adicionar novas dependências do sistema (pacotes OS) ou portas, modifique o `Dockerfile` e o `docker-compose.yml`. Use sempre `0.0.0.0` como bind de host no código Python para garantir que o container não fique isolado.

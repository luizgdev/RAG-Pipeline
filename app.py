import gradio as gr
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.generation.rag_chain import MeditacoesRAG

# Carrega variáveis de ambiente
load_dotenv()

# Inicializa o motor RAG
print("🏛️ Despertando o Oráculo de Marco Aurélio...")
rag_engine = MeditacoesRAG()
rag_chain = rag_engine.get_chain()

def format_history_safe(gr_history):
    """
    Lê o histórico no padrão nativo do Gradio 5+ (Dicionários ou Objetos OpenAI)
    e converte para o formato rigoroso do LangChain.
    """
    lc_history = []
    for msg in gr_history:
        # Se o Gradio entregar como um dicionário
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
        # Se o Gradio entregar como um objeto interno (ChatMessage)
        elif hasattr(msg, "role"):
            role = msg.role
            content = msg.content
        else:
            continue
            
        # Instancia as classes do LangChain
        if role == "user":
            lc_history.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            lc_history.append(AIMessage(content=str(content)))
            
    return lc_history

def chat_interface(message, history):
    """
    Função principal de processamento.
    Retorna a resposta do bot e atualiza o painel de referências.
    """
    inputs = {
        "question": message,
        "history": format_history_safe(history)
    }
    
    try:
        result = rag_chain.invoke(inputs)
        answer = result["answer"]
        raw_docs = result["raw_docs"]
        
        refs_text = ""
        if not raw_docs:
            refs_text = "Nenhum documento relevante foi encontrado para esta pergunta."
        else:
            for i, doc in enumerate(raw_docs, 1):
                secao = doc.metadata.get('secao', 'Desconhecido')
                licao = doc.metadata.get('licao', '?')
                refs_text += f"### [{i}] {secao} | Lição {licao}\n"
                refs_text += f"{doc.page_content}\n\n---\n"
        
        return answer, refs_text
    
    except Exception as e:
        error_msg = f"Ocorreu um erro ao consultar o oráculo: {str(e)}"
        return error_msg, ""

# --- CONSTRUÇÃO DA INTERFACE ---
with gr.Blocks(title="Oráculo de Marco Aurélio") as demo:
    gr.Markdown("""
    # 🏛️ Oráculo de Marco Aurélio
    *Consulte a sabedoria das Meditações através de uma inteligência artificial acadêmica.*
    """)
    
    with gr.Row():
        # COLUNA DA ESQUERDA: Chat e Voz
        with gr.Column(scale=3):
            # Deixamos o Chatbot limpo, sem forçar 'type'
            chatbot = gr.Chatbot(show_label=False, height=500)
            msg = gr.Textbox(
                placeholder="Pergunte algo a Marco Aurélio...",
                label="Sua pergunta",
                container=False
            )
            
            with gr.Accordion("🎧 Resposta em Áudio (Beta)", open=False):
                audio_output = gr.Audio(label="Voz do Imperador", interactive=False)
                gr.Markdown("*Integração com ElevenLabs em breve.*")
            
            submit_btn = gr.Button("Consultar Oráculo", variant="primary")
            clear = gr.ClearButton([msg, chatbot], value="Limpar Conversa")

        # COLUNA DA DIREITA: Auditoria RAG
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Transparência RAG")
            sources_output = gr.Markdown(
                value="As referências do livro aparecerão aqui após a resposta.",
                label="Fontes Consultadas"
            )

    def respond(message, chat_history):
        bot_message, refs = chat_interface(message, chat_history)
        
        # O PULO DO GATO: Entregando exatamente o que o erro pediu (dicionários com role e content)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        
        return "", chat_history, refs

    submit_btn.click(respond, [msg, chatbot], [msg, chatbot, sources_output])
    msg.submit(respond, [msg, chatbot], [msg, chatbot, sources_output])

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    server = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    
    demo.launch(server_name=server, server_port=port, theme=gr.themes.Soft())
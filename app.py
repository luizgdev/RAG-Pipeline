import gradio as gr
import os
import copy
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.generation.rag_chain import MeditacoesRAG
from src.generation.audio import gerador_tts
from utils.prompts import history_inicial

# Carrega variáveis de ambiente forçando a atualização do cache do Windows
load_dotenv(override=True)

# Inicializa o motor RAG
print("🏛️ Despertando o Oráculo de Marco Aurélio...")
rag_engine = MeditacoesRAG()
rag_chain = rag_engine.get_chain()

def format_history_safe(gr_history):
    """
    Converte o histórico do Gradio para o formato Human/AI do LangChain.
    """
    lc_history = []
    for msg in gr_history:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
        elif hasattr(msg, "role"):
            role = msg.role
            content = msg.content
        else:
            continue
            
        if role == "user":
            lc_history.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            lc_history.append(AIMessage(content=str(content)))
            
    return lc_history

def format_references(raw_docs):
    """
    Formata os metadados e conteúdo para o painel de transparência.
    """
    refs_text = ""
    if not raw_docs:
        return "Nenhum documento relevante foi encontrado para esta pergunta."
    
    for i, doc in enumerate(raw_docs, 1):
        secao = doc.metadata.get('secao', 'Desconhecido')
        licao = doc.metadata.get('licao', '?')
        refs_text += f"### [{i}] {secao} | Lição {licao}\n"
        refs_text += f"{doc.page_content}\n\n---\n"
    return refs_text

def processar_interacao(mensagem_texto, chat_history):
    """
    Fluxo de Texto: RAG Streaming.
    """
    input_final = mensagem_texto
    
    if not input_final or not input_final.strip():
        gr.Warning("Por favor, digite uma mensagem válida.")
        yield "", chat_history, gr.update()
        return

    inputs = {
        "question": input_final,
        "history": format_history_safe(chat_history)
    }
    
    chat_history.append({"role": "user", "content": input_final})
    chat_history.append({"role": "assistant", "content": ""})
    
    resposta_acumulada = ""
    referencias_finais = "Buscando referências no livro..."
    referencias_cacheadas = None
    
    # Exibe o estado inicial (limpa o texto)
    yield "", chat_history, referencias_finais

    try:
        for chunk in rag_chain.stream(inputs):
            if "raw_docs" in chunk and referencias_cacheadas is None:
                 referencias_cacheadas = format_references(chunk["raw_docs"])
                 referencias_finais = referencias_cacheadas
            
            if "answer" in chunk:
                resposta_acumulada += chunk["answer"]
                chat_history[-1]["content"] = resposta_acumulada
                yield "", chat_history, referencias_finais
                
    except Exception as e:
         chat_history[-1]["content"] = f"Erro no Oráculo: {str(e)}"
         yield "", chat_history, ""

def gerar_audio_manual(chat_history):
    """
    Acionada pelo botão: Gera áudio apenas da última resposta do chat.
    Blindada contra TODOS os formatos multimodais do Gradio 5+.
    """
    if not chat_history:
        return None
    
    ultima_msg = chat_history[-1]
    texto_resposta = ""
    
    # 1. Se o Gradio enviar no formato Dicionário
    if isinstance(ultima_msg, dict):
        if ultima_msg.get("role") == "assistant":
            texto_resposta = ultima_msg.get("content", "")
            
    # 2. Se o Gradio enviar no formato Clássico (Lista: [user, bot])
    elif isinstance(ultima_msg, (list, tuple)) and len(ultima_msg) == 2:
        texto_resposta = ultima_msg[1]
        
    # --- A BLINDAGEM MULTIMODAL CORRIGIDA ---
    if isinstance(texto_resposta, (list, tuple)):
        pedacos = []
        for item in texto_resposta:
            # Se for texto puro
            if isinstance(item, str):
                pedacos.append(item)
            # Se for o novo formato de dicionário do Gradio [{'text': '...', 'type': 'text'}]
            elif isinstance(item, dict) and "text" in item:
                pedacos.append(item["text"])
                
        texto_resposta = " ".join(pedacos)
        
    # Garante que temos uma string limpa no final
    texto_resposta = str(texto_resposta).strip()

    # 3. Dispara para a ElevenLabs
    if texto_resposta:
        try:
            audio_path_gerado = gerador_tts(texto_resposta, "imperador")
            return audio_path_gerado
        except Exception as e:
            gr.Warning(f"Erro na ElevenLabs: {str(e)}")
            return None
            
    return None


# --- INTERFACE GRADIO ---
with gr.Blocks(title="Oráculo de Marco Aurélio") as demo:
    gr.Markdown("# 🏛️ Oráculo de Marco Aurélio")
    gr.Markdown("*Consulte a sabedoria das Meditações através de uma IA acadêmica.*")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(value=history_inicial, show_label=False, height=450)
            
            with gr.Group():
                msg = gr.Textbox(placeholder="Digite sua pergunta...", show_label=False, container=False)
            
            with gr.Row():
                submit_btn = gr.Button("Consultar Oráculo", variant="primary")
                clear = gr.Button("Limpar Conversa", variant="secondary")

            with gr.Accordion("🎧 Voz do Imperador", open=True):
                btn_audio = gr.Button("🔊 Ouvir Última Resposta", variant="secondary")
                audio_output = gr.Audio(label="Player de Áudio", interactive=False, autoplay=True)

        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Transparência RAG")
            sources_output = gr.Markdown(value="As referências aparecerão aqui.", label="Fontes")

    # --- EVENTOS DE CHAT ---
    submit_event = {
        "fn": processar_interacao, 
        "inputs": [msg, chatbot], 
        "outputs": [msg, chatbot, sources_output] 
    }
    
    # 1. Envio via botão azul
    submit_btn.click(**submit_event)
    
    # 2. Envio via "Enter" no teclado
    msg.submit(**submit_event)
    
    # 3. Limpar Conversa (Garante que tudo seja resetado)
    clear.click(
        fn=lambda: ("", copy.deepcopy(history_inicial), "As referências aparecerão aqui.", None),
        inputs=None,
        outputs=[msg, chatbot, sources_output, audio_output]
    )
    
    # --- Evento de Áudio Manual da ElevenLabs ---
    btn_audio.click(
        fn=gerar_audio_manual,
        inputs=[chatbot],
        outputs=[audio_output]
    )

if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    
    print("\n" + "="*60)
    print("🏛️  O ORÁCULO DE MARCO AURÉLIO ESTÁ ONLINE!")
    print("="*60)
    print("👉 ACESSE PELO SEU NAVEGADOR CLICANDO NO LINK ABAIXO:")
    print(f"🔗 http://localhost:{port}")
    print("="*60 + "\n")
    
    # Para contêineres Docker, precisamos forçar o binding em 0.0.0.0
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft(), quiet=True)
import os
import re
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv(override=True)

# Inicializa os clientes
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# IDs de voz reais que você configurou
VOICES = {
    "imperador": "N2lVS1w4EtoT3dr4eOWO",
}

def gerador_tts(texto, voice_key="imperador"):
    """ElevenLabs: Text-to-Speech (Resposta do RAG -> Voz do Imperador)"""
    # Limpa referências como [1], [2] para o áudio ficar natural
    texto_limpo = re.sub(r'\[\d+\]', '', texto)
    
    try:
        # CORREÇÃO: O método correto na v1+ é text_to_speech.convert
        response = client_eleven.text_to_speech.convert(
            voice_id=VOICES.get(voice_key, VOICES["imperador"]),
            text=texto_limpo,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        
        # A resposta é um gerador de bytes, precisamos salvar manualmente
        output_path = "resposta_imperador.mp3"
        with open(output_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)
                    
        return output_path
    except Exception as e:
        raise

def gerador_s2s(audio_path_usuario, voice_key="imperador"):
    """ElevenLabs: Speech-to-Speech (Voz do usuário -> Voz do Imperador diretamente)"""
    with open(audio_path_usuario, "rb") as audio_file:
        response = client_eleven.speech_to_speech.convert(
            voice_id=VOICES.get(voice_key, VOICES["imperador"]),
            audio=audio_file,
            model_id="eleven_english_sts_v2",
            output_format="mp3_44100_128",
        )
        
        output_path = "s2s_imperador.mp3"
        with open(output_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)
    
    return output_path
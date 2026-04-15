import os
import requests
from pathlib import Path

def download_pdf(url: str, output_path: str) -> str:
    """
    Baixa um arquivo PDF de uma URL e o salva no caminho especificado.
    Se o arquivo já existir no destino, pula o download.
    """
    # Converte a string do caminho para um objeto Path (Resolve problemas do Windows)
    dest_path = Path(output_path)
    
    # Garante que as pastas pai (ex: data/raw) existam antes de salvar
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        print(f"O arquivo já existe em: {dest_path}")
        print("Pulando o download...")
        return str(dest_path)
        
    print(f"Iniciando o download do livro de: {url}")
    
    try:
        # Timeout de 30s é uma boa prática em engenharia de software para evitar travamentos eternos
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status() 

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"Download concluído com sucesso! Salvo em: {dest_path}")
        return str(dest_path)

    except requests.exceptions.RequestException as e:
        print(f"Erro de rede ou HTTP ao baixar o arquivo: {e}")
        raise # Interrompe a execução do pipeline, pois sem dados não há RAG
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        raise
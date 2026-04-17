import re
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from src.retrieval.embeddings import get_embeddings_model

class AphorismChunker:
    """
    Fatia o texto do livro Meditações identificando Números e Letras opcionais.
    Remove automaticamente marcadores de notas de rodapé e seus glossários.
    """
    def __init__(self, threshold: int = 85):
        self.embeddings_model = get_embeddings_model()
        self.semantic_splitter = SemanticChunker(
            embeddings=self.embeddings_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold
        )

    def _clean_footnotes(self, text: str) -> str:
        """
        Remove o bloco de notas de rodapé no final do texto e os marcadores inline.
        """
        # Passo 1: Guilhotina (Corta o glossário final)
        # Regex: Lookbehind para pontuação (?<=[.!?"]), espaços \s+, Lookahead para nota (?=\[\s*\d+\s*\]\s+[A-Z])
        parts = re.split(r'(?<=[.!?"])\s+(?=\[\s*\d+\s*\]\s+[A-Z])', text)
        main_text = parts[0] # Pegamos apenas a parte antes do primeiro corte
        
        # Passo 2: Pinça (Remove os marcadores ex: "[2]" ou "[ 1 ]" do meio do texto)
        clean_text = re.sub(r'\[\s*\d+\s*\]', '', main_text)
        
        # Passo 3: Limpeza de múltiplos espaços deixados pela remoção
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text

    def split_text(self, text: str) -> List[Document]:
        # Estágio 1: Chunking Estrutural
        pattern = r'\n(?=(Introdução|Livro\s[IVX]+)\n)'
        splits = re.split(pattern, text)
        docs_splitted = []
        
        i = 1
        while i < len(splits):
            title = splits[i].strip()
            content = splits[i+1].strip()
            
            if content.startswith(title):
                content = content[len(title):].strip()
                
            if title.lower() == "introdução":
                i += 2
                continue 
                
            pattern_licao = r'(?:^|\n)\s*(\d+)([a-z]?)\.\s+'
            splits_licoes = re.split(pattern_licao, content)
            
            j = 1
            while j < len(splits_licoes):
                num_licao = splits_licoes[j]
                text_licao = splits_licoes[j+2].strip()
                
                # Limpa notas antes de criar o Documento
                clean_licao = self._clean_footnotes(text_licao)
                
                # Prevenção: Só avança se sobrou conteúdo válido (maior que 10 caracteres)
                if len(clean_licao) > 10:
                    doc = Document(
                        page_content=clean_licao, 
                        metadata={
                            "origem": "meditacoes_marco_aurelio.pdf",
                            "secao": title,
                            "licao": num_licao, 
                            "autor": "Marco Aurélio",
                            "estilo": "Estoicismo"
                        }
                    )
                    docs_splitted.append(doc)
                
                j += 3
                
            i += 2
            
        # Estágio 2: Chunking Semântico
        semantic_docs = self.semantic_splitter.split_documents(docs_splitted)
            
        return semantic_docs
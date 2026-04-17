import re
from langchain_community.document_loaders import PyMuPDFLoader

class MeditacoesParser:
    """
    Carrega o PDF, remove páginas irrelevantes e corrige as quebras 
    de linha falsas geradas pelo formato PDF.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def _clean_text(self, text: str) -> str:
        def substitution_rule(match):
            line = match.group(1)
            # Se a linha termina em pontuação ou é curta, mantemos o \n
            if re.search(r'[.!?]$', line) or len(line) < 25:
                return line + '\n'
            # Caso contrário, transforma o \n em espaço
            return line + ' '
        
        # O regex pega cada linha individualmente
        return re.sub(r'^(.+)\n', substitution_rule, text, flags=re.MULTILINE)

    def get_cleaned_text(self) -> str:
        """Retorna o texto do livro inteiro limpo como uma única string."""
        loader = PyMuPDFLoader(self.file_path)
        content = loader.load()
        
        # Ignora as 2 primeiras e 2 últimas páginas
        complete_text = "\n".join([doc.page_content for doc in content[2:-2]])
        
        return self._clean_text(complete_text)
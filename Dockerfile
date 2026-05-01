FROM python:3.11-slim

# Evitar prompts interativos no apt-get
ENV DEBIAN_FRONTEND=noninteractive


# Configurar diretório de trabalho
WORKDIR /app

# Copiar requirements primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Atualizar pip e instalar dependências
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código
COPY . .

# Porta padrão do Gradio
EXPOSE 7860

# Variáveis de ambiente padrão para garantir que o Gradio seja exposto para fora do contêiner
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONUNBUFFERED=1

# Comando de inicialização
CMD ["python", "app.py"]

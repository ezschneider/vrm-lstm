# Use uma imagem base com Python 3.9
FROM python:3.9-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de requirements para o container
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Instala o Jupyter Notebook
RUN pip install jupyter

# Expõe a porta 8888 (porta padrão do Jupyter Notebook)
EXPOSE 8888

# Comando para rodar o Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
# **Projeto de Análise de Séries Temporais com Poetry**

## **Descrição do Projeto**
Este projeto realiza a análise de dados de vibração ao longo do tempo utilizando bibliotecas do ecossistema Python. Foi implementado para processar séries temporais, realizar decomposição sazonal e detectar padrões com base em dados históricos. O gerenciamento de dependências e ambientes é feito utilizando **Poetry**.

---

## **Pré-requisitos**

Antes de rodar o projeto, certifique-se de que os seguintes requisitos estão instalados:

1. **Python** (>= 3.12)
2. **Poetry** (>= 1.1.0)

### **Instalação do Poetry**
Caso o Poetry não esteja instalado, você pode instalá-lo com o seguinte comando:

```bash
pipx install poetry
```

### **Verificar se foi instalado**

```bash
poetry --version
```

---

### **Configuração do Ambiente**

#### 1. **Clone o repositório:**
Baixe o código-fonte do projeto para o seu computador usando git:

```bash
git clone <URL_DO_REPOSITÓRIO>
cd <NOME_DA_PASTA_DO_PROJETO>
```

#### 2. **Instale as dependências:**
Utilize o Poetry para instalar todas as dependências necessárias para rodar o projeto:

```bash
poetry install
```

Este comando cria um ambiente virtual isolado e baixa todas as bibliotecas listadas no pyproject.toml.

---

### **Execução do Programa**

Para rodar o programa, utilize o Poetry no terminal da seguinte forma:

```bash
poetry run python <nome_do_arquivo_principal>.py

# Substitua <nome_do_arquivo_principal> pelo nome do script principal do projeto.
```

**Executando no Jupyter Notebook:**

Caso o projeto utilize Jupyter Notebook e você deseje rodar as análises no ambiente interativo:

1. Ative o kernel do Poetry:
```bash
poetry run jupyter notebook
```
2. Abra o arquivo `.ipynb` no Jupyter e execute as células.

---

### Observasões

> [!WARNING]
> Dados pré processados não disponibilizados por questões de Licença

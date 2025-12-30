# Minicurso: IA para Pesquisa - Introdução a Embeddings

Este projeto contém notebooks educacionais para aprender sobre **embeddings** e **busca semântica** usando a API da OpenAI.

## 📚 Conteúdo

| Notebook | Descrição |
|----------|-----------|
| [01_embeddings_palavras.ipynb](01_embeddings_palavras.ipynb) | Introdução a embeddings usando palavras isoladas. Aprenda os conceitos de similaridade de cosseno e visualização com t-SNE. |
| [02_embeddings_frases.ipynb](02_embeddings_frases.ipynb) | Embeddings de frases e busca semântica. Base para entender RAG (Retrieval Augmented Generation). |
| [03_busca_semantica_wikipedia.ipynb](03_busca_semantica_wikipedia.ipynb) | Busca semântica com conteúdo real da Wikipedia. Aprenda sobre chunking e busca em documentos. |
| [04_rag_completo.ipynb](04_rag_completo.ipynb) | RAG completo (Retrieval Augmented Generation). Combina busca semântica com LLM para respostas contextualizadas. |

## 🔧 Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes Python)
- Chave de API da OpenAI

## 🚀 Configuração do Ambiente

### 1. Instalar o uv (se ainda não tiver)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Criar e ativar o ambiente virtual

```bash
# Criar o ambiente virtual
uv venv .venv

# Ativar o ambiente (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Ativar o ambiente (Linux/macOS)
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
uv pip install -r requirements.txt
```

### 4. Configurar a API Key da OpenAI

Crie um arquivo `.env` na raiz do projeto com sua chave de API:

```env
OPENAI_API_KEY=sk-sua-chave-aqui
```

> ⚠️ **Importante**: Nunca compartilhe sua chave de API ou a commit no repositório. O arquivo `.env` já está no `.gitignore`.

Para obter uma chave de API:
1. Acesse [platform.openai.com](https://platform.openai.com/)
2. Crie uma conta ou faça login
3. Vá em **API Keys** e crie uma nova chave
4. Copie a chave e cole no arquivo `.env`

## 📦 Estrutura do Projeto

```
mc_ia_para_pesquisa/
├── .env                          # Variáveis de ambiente (API keys)
├── .venv/                        # Ambiente virtual Python
├── requirements.txt              # Dependências do projeto
├── embedding_utils.py            # Funções utilitárias para embeddings
├── 01_embeddings_palavras.ipynb  # Notebook 1: Embeddings de palavras
├── 02_embeddings_frases.ipynb    # Notebook 2: Busca semântica com frases
├── 03_busca_semantica_wikipedia.ipynb  # Notebook 3: Busca na Wikipedia
├── 04_rag_completo.ipynb         # Notebook 4: Sistema RAG completo
├── embeddings_chunks.npy         # Embeddings salvos (gerado pelo notebook 3)
├── metadados_chunks.json         # Metadados dos chunks (gerado pelo notebook 3)
├── chunks.json                   # Chunks de texto (gerado pelo notebook 3)
└── README.md                     # Este arquivo
```

## 💡 Conceitos Abordados

- **Embeddings**: Representações vetoriais de texto em espaços de alta dimensão
- **Similaridade de Cosseno**: Métrica para medir proximidade entre vetores
- **t-SNE**: Técnica de redução de dimensionalidade para visualização
- **Busca Semântica**: Encontrar informações relevantes baseado no significado, não em palavras-chave
- **RAG (Retrieval Augmented Generation)**: Base para sistemas que combinam busca com LLMs

## 📝 Licença

Este projeto é para fins educacionais.

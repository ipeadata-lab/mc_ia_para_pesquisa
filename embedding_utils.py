"""
Funções utilitárias para trabalhar com embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
import requests
from bs4 import BeautifulSoup


def get_embedding(client, text: str, model: str = "text-embedding-3-large") -> list:
    """
    Obtém o embedding de um texto usando a API da OpenAI.

    Args:
        client: Cliente OpenAI configurado
        text: O texto para gerar o embedding
        model: O modelo de embedding a ser usado

    Returns:
        Uma lista de floats representando o embedding
    """
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calcula a similaridade de cosseno entre dois vetores.
    Retorna um valor entre 0 e 1, onde 1 = idênticos.
    """
    return 1 - cosine(vec1, vec2)


def plot_similarity_matrix(
    palavras, embeddings, group_separators=None, figsize=(12, 10)
):
    """
    Cria uma matriz visual mostrando a similaridade entre todas as palavras.

    Args:
        palavras: Lista de palavras a serem comparadas
        embeddings: Dicionário com os embeddings de cada palavra
        group_separators: Lista de posições para adicionar linhas separadoras (opcional)
        figsize: Tamanho da figura (largura, altura)

    Returns:
        matriz_similaridade: Matriz numpy com os valores de similaridade
    """
    n = len(palavras)
    matriz_similaridade = np.zeros((n, n))

    for i, palavra1 in enumerate(palavras):
        for j, palavra2 in enumerate(palavras):
            matriz_similaridade[i, j] = cosine_similarity(
                embeddings[palavra1]["embedding"], embeddings[palavra2]["embedding"]
            )

    # Visualizando a matriz
    plt.figure(figsize=figsize)
    plt.imshow(matriz_similaridade, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(label="Similaridade de Cosseno")
    plt.xticks(range(n), palavras, rotation=45, ha="right")
    plt.yticks(range(n), palavras)
    plt.title(
        "Matriz de Similaridade entre Palavras\n(Verde = Alta Similaridade, Vermelho = Baixa Similaridade)",
        fontsize=14,
    )

    # Adicionando linhas para separar os grupos
    if group_separators:
        for sep in group_separators:
            plt.axhline(y=sep, color="black", linewidth=2)
            plt.axvline(x=sep, color="black", linewidth=2)

    plt.tight_layout()
    plt.show()

    return matriz_similaridade


def plot_embeddings_2d(
    palavras,
    embeddings_list,
    cores,
    legenda_config=None,
    perplexity=5,
    figsize=(12, 8),
    random_state=42,
):
    """
    Reduz os embeddings para 2 dimensões usando t-SNE e plota a visualização.

    Args:
        palavras: Lista de palavras correspondentes aos embeddings
        embeddings_list: Lista de vetores de embeddings
        cores: Lista de cores para cada palavra
        legenda_config: Lista de dicionários com 'color' e 'label' para a legenda (opcional)
        perplexity: Parâmetro perplexity do t-SNE (default: 5)
        figsize: Tamanho da figura (largura, altura)
        random_state: Seed para reprodutibilidade

    Returns:
        embeddings_2d: Array com as coordenadas 2D dos embeddings
    """
    # Reduzindo dimensionalidade com t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(np.array(embeddings_list))

    # Plotando
    plt.figure(figsize=figsize)

    for i, palavra in enumerate(palavras):
        plt.scatter(
            embeddings_2d[i, 0], embeddings_2d[i, 1], c=cores[i], s=200, alpha=0.7
        )
        plt.annotate(
            palavra,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=12,
            ha="center",
            va="bottom",
            xytext=(0, 10),
            textcoords="offset points",
        )

    # Legenda
    if legenda_config:
        legenda = [
            Patch(facecolor=item["color"], label=item["label"])
            for item in legenda_config
        ]
        plt.legend(handles=legenda, loc="upper right", fontsize=12)

    plt.title(
        "Visualização 2D dos Embeddings (t-SNE)\nPalavras relacionadas ficam próximas!",
        fontsize=14,
    )
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return embeddings_2d


def comparar_palavras(client, palavra1: str, palavra2: str):
    """
    Compara duas palavras e mostra sua similaridade.

    Args:
        client: Cliente OpenAI configurado
        palavra1: Primeira palavra
        palavra2: Segunda palavra

    Returns:
        sim: Valor de similaridade entre as palavras
    """
    emb1 = get_embedding(client, palavra1)
    emb2 = get_embedding(client, palavra2)
    sim = cosine_similarity(emb1, emb2)

    print(f"\n{'═' * 40}")
    print(f"Comparando: '{palavra1}' vs '{palavra2}'")
    print(f"{'═' * 40}")
    print(f"Similaridade: {sim:.4f} ({sim*100:.1f}%)")

    # Interpretação
    if sim > 0.8:
        print("📊 Interpretação: Muito similares! Provavelmente relacionadas.")
    elif sim > 0.5:
        print("📊 Interpretação: Moderadamente similares.")
    else:
        print("📊 Interpretação: Pouco similares. Conceitos diferentes.")

    return sim


def buscar_afirmacao_mais_relevante(
    client, pergunta: str, afirmacoes: list, embeddings_afirmacoes: list, top_k: int = 3
):
    """
    Busca as afirmações mais relevantes para responder uma pergunta.

    Args:
        client: Cliente OpenAI configurado
        pergunta: A pergunta em linguagem natural
        afirmacoes: Lista de afirmações/fatos
        embeddings_afirmacoes: Lista de embeddings das afirmações
        top_k: Número de resultados a retornar

    Returns:
        Lista de tuplas (afirmação, similaridade)
    """
    # Gera embedding da pergunta
    emb_pergunta = get_embedding(client, pergunta)

    # Calcula similaridade com todas as afirmações
    similaridades = []
    for i, emb_afirmacao in enumerate(embeddings_afirmacoes):
        sim = cosine_similarity(emb_pergunta, emb_afirmacao)
        similaridades.append((afirmacoes[i], sim))

    # Ordena por similaridade (maior primeiro)
    similaridades.sort(key=lambda x: x[1], reverse=True)

    return similaridades[:top_k]


def mostrar_busca(client, pergunta: str, afirmacoes: list, embeddings_afirmacoes: list):
    """
    Mostra os resultados de busca de forma visual.

    Args:
        client: Cliente OpenAI configurado
        pergunta: A pergunta em linguagem natural
        afirmacoes: Lista de afirmações/fatos
        embeddings_afirmacoes: Lista de embeddings das afirmações
    """
    print(f"\n{'═' * 60}")
    print(f"❓ PERGUNTA: {pergunta}")
    print(f"{'═' * 60}")

    resultados = buscar_afirmacao_mais_relevante(
        client, pergunta, afirmacoes, embeddings_afirmacoes, top_k=3
    )

    print("\n📊 Afirmações mais relevantes:\n")
    for i, (afirmacao, sim) in enumerate(resultados, 1):
        # Barra de progresso visual
        barra = "█" * int(sim * 30) + "░" * (30 - int(sim * 30))
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"{emoji} [{barra}] {sim:.1%}")
        print(f"   → {afirmacao}\n")


def buscar_wikipedia(titulo: str) -> dict:
    """
    Busca o conteúdo de uma página da Wikipedia em português.

    Args:
        titulo: O título da página (ex: "Albert_Einstein")

    Returns:
        Dicionário com 'titulo', 'conteudo' e 'url'
    """
    url = f"https://pt.wikipedia.org/wiki/{titulo}"

    # Headers necessários para evitar bloqueio 403
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Erro ao buscar página: {response.status_code}")

    soup = BeautifulSoup(response.content, "html.parser")

    # Pega o conteúdo principal
    content_div = soup.find("div", {"id": "mw-content-text"})

    # Remove elementos indesejados
    for element in content_div.find_all(["table", "sup", "span", "style", "script"]):
        element.decompose()

    # Extrai parágrafos
    paragrafos = []
    for p in content_div.find_all("p"):
        texto = p.get_text().strip()
        # Filtra parágrafos muito curtos ou vazios
        if len(texto) > 50:
            paragrafos.append(texto)

    return {"titulo": titulo.replace("_", " "), "conteudo": paragrafos, "url": url}


def dividir_em_chunks(
    texto: str, titulo_pagina: str = None, tamanho_max: int = 1000, overlap: int = 100
) -> list:
    """
    Divide um texto em chunks menores com sobreposição (overlap).

    Args:
        texto: O texto a ser dividido
        titulo_pagina: Nome da página/documento para adicionar como contexto no início de cada chunk
        tamanho_max: Tamanho máximo de cada chunk em caracteres
        overlap: Número de caracteres de sobreposição entre chunks consecutivos

    Returns:
        Lista de chunks com overlap entre eles, cada um prefixado com o título da página
    """
    # Prefixo de contexto
    prefixo = f"[{titulo_pagina}] " if titulo_pagina else ""
    tamanho_prefixo = len(prefixo)
    tamanho_util = tamanho_max - tamanho_prefixo

    # Divide por sentenças
    sentencas = texto.replace("\n", " ").split(". ")

    # Reconstrói adicionando ponto final
    sentencas = [s.strip() + "." for s in sentencas if s.strip()]

    chunks = []
    chunk_atual = ""
    sentencas_do_chunk = []

    for sentenca in sentencas:
        if len(chunk_atual) + len(sentenca) + 1 < tamanho_util:
            chunk_atual += " " + sentenca if chunk_atual else sentenca
            sentencas_do_chunk.append(sentenca)
        else:
            if chunk_atual:
                chunks.append(prefixo + chunk_atual.strip())

            # Calcula overlap: pega sentenças do final do chunk anterior
            overlap_text = ""
            overlap_sentencas = []
            for s in reversed(sentencas_do_chunk):
                if len(overlap_text) + len(s) + 1 <= overlap:
                    overlap_text = s + " " + overlap_text if overlap_text else s
                    overlap_sentencas.insert(0, s)
                else:
                    break

            # Inicia novo chunk com overlap + sentença atual
            chunk_atual = overlap_text + " " + sentenca if overlap_text else sentenca
            sentencas_do_chunk = overlap_sentencas + [sentenca]

    if chunk_atual:
        chunks.append(prefixo + chunk_atual.strip())

    return chunks

"""
Analisador de Repertório de Aberturas de Xadrez
=================================================

Aplicação Streamlit que recebe um arquivo PGN (podendo conter várias partidas)
e um nome de jogador, e mostra estatísticas de quais aberturas esse jogador
usa (separadas por cor), com frequência e taxa de vitória/empate/derrota.

Como rodar:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
from collections import defaultdict

import chess
import chess.pgn
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Tabela de fallback: mapeia sequências iniciais de lances (em SAN, separados
# por espaço) para o nome da abertura correspondente. Usada apenas quando a
# partida não possui as tags ECO/Opening no cabeçalho do PGN.
# As chaves mais longas (mais lances) são verificadas primeiro, para que
# variações mais específicas tenham prioridade sobre a abertura genérica.
# ---------------------------------------------------------------------------
OPENING_FALLBACK_TABLE = {
    "e4 e5 Nf3 Nc6 Bb5": "Ruy Lopez (Espanhola)",
    "e4 e5 Nf3 Nc6 Bc4": "Abertura Italiana",
    "e4 e5 Nf3 Nc6 d4": "Gambito Escocês",
    "e4 e5 Nf3 Nc6": "Abertura do Cavalo do Rei",
    "e4 e5 Bc4": "Gambito Bispo do Rei",
    "e4 e5 f4": "Gambito do Rei",
    "e4 e5": "Abertura Aberta (1.e4 e5)",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3": "Defesa Siciliana Clássica",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3": "Defesa Siciliana Clássica",
    "e4 c5 Nf3 d6": "Defesa Siciliana Najdorf/Scheveningen",
    "e4 c5 Nc3": "Defesa Siciliana Fechada",
    "e4 c5 c3": "Defesa Siciliana Alapin",
    "e4 c5": "Defesa Siciliana",
    "e4 e6 d4 d5 Nc3": "Defesa Francesa Clássica",
    "e4 e6 d4 d5 Nd2": "Defesa Francesa Tarrasch",
    "e4 e6": "Defesa Francesa",
    "e4 c6 d4 d5": "Defesa Caro-Kann",
    "e4 c6": "Defesa Caro-Kann",
    "e4 d5": "Defesa Escandinava",
    "e4 Nf6": "Defesa Alekhine",
    "e4 g6": "Defesa Moderna",
    "e4 d6 d4 Nf6 Nc3 g6": "Defesa Pirc",
    "e4 d6": "Defesa Pirc/Filidor",
    "d4 d5 c4 e6": "Gambito da Dama Recusado",
    "d4 d5 c4 c6": "Defesa Eslava",
    "d4 d5 c4 dxc4": "Gambito da Dama Aceito",
    "d4 d5 Nf3 Nf6 c4": "Gambito da Dama",
    "d4 d5": "Abertura da Dama Fechada (1.d4 d5)",
    "d4 Nf6 c4 g6 Nc3 Bg7": "Defesa Grünfeld",
    "d4 Nf6 c4 e6 Nc3 Bb4": "Defesa Nimzo-Índia",
    "d4 Nf6 c4 g6": "Defesa Índia do Rei",
    "d4 Nf6 c4 e6 Nf3 b6": "Defesa Índia da Dama",
    "d4 Nf6 Nf3 g6": "Sistema Fianchetto / Índia do Rei",
    "d4 Nf6 Bg5": "Ataque Torre/London (via Nf6)",
    "d4 Nf6": "Defesa Índia (genérica)",
    "d4 f5": "Defesa Holandesa",
    "d4 g6": "Defesa Moderna (via d4)",
    "d4 e6": "Defesa Francesa (via d4)",
    "d4 d6": "Defesa Pirc (via d4)",
    "d4": "Abertura da Dama",
    "Nf3 d5 g3": "Ataque Réti / Fianchetto",
    "Nf3 Nf6 c4": "Abertura Inglesa (via Nf3)",
    "Nf3 d5 c4": "Abertura Réti",
    "Nf3": "Abertura Réti",
    "c4 e5": "Abertura Inglesa Reversa",
    "c4 Nf6": "Abertura Inglesa",
    "c4 c5": "Abertura Inglesa Simétrica",
    "c4": "Abertura Inglesa",
    "g3": "Abertura Benko / Fianchetto do Rei",
    "b3": "Abertura Larsen/Nimzowitsch",
    "f4": "Abertura Bird",
    "Nc3": "Abertura Van Geet",
    "b4": "Abertura Orangutango (Sokolsky)",
}
# Ordena as chaves da mais longa (mais específica) para a mais curta,
# para que a busca por prefixo encontre sempre a correspondência mais precisa.
_SORTED_FALLBACK_KEYS = sorted(
    OPENING_FALLBACK_TABLE.keys(), key=lambda k: len(k.split()), reverse=True
)


def identify_opening_by_moves(moves_san):
    """
    Identifica o nome de uma abertura a partir da lista de lances (SAN)
    de uma partida, usando a tabela de fallback embutida.

    Procura o prefixo mais longo dos lances da partida que bate com alguma
    entrada da tabela. Se nenhuma entrada for encontrada, retorna um nome
    genérico baseado nos primeiros lances.
    """
    moves_str = " ".join(moves_san)
    for key in _SORTED_FALLBACK_KEYS:
        if moves_str.startswith(key):
            return OPENING_FALLBACK_TABLE[key]
    # Nenhuma abertura conhecida bateu: usa os 2 primeiros lances como rótulo
    if len(moves_san) >= 2:
        return f"Abertura não catalogada ({moves_san[0]} {moves_san[1]})"
    elif len(moves_san) == 1:
        return f"Abertura não catalogada ({moves_san[0]})"
    return "Abertura desconhecida"


def parse_pgn(pgn_bytes, player_name):
    """
    Faz o parsing de um arquivo PGN (em bytes) contendo uma ou mais partidas,
    filtrando apenas as partidas em que `player_name` participou (correspondência
    exata com a tag White ou Black).

    Retorna uma tupla (registros, partidas_ignoradas, total_partidas_lidas), onde:
      - registros: lista de dicts, um por partida válida do jogador, contendo
        cor, abertura, resultado e fonte da abertura (tag ou fallback)
      - partidas_ignoradas: dict com contadores de motivos de partidas ignoradas
      - total_partidas_lidas: número total de partidas encontradas no arquivo
    """
    pgn_text = pgn_bytes.decode("utf-8", errors="replace")
    pgn_io = io.StringIO(pgn_text)

    registros = []
    ignoradas = defaultdict(int)
    total_lidas = 0

    while True:
        try:
            game = chess.pgn.read_game(pgn_io)
        except Exception:
            # Partida malformada que o parser não conseguiu processar
            ignoradas["malformada"] += 1
            continue

        if game is None:
            break  # fim do arquivo

        total_lidas += 1
        headers = game.headers

        white = headers.get("White", "")
        black = headers.get("Black", "")

        # Correspondência exata (case-sensitive) com o nome informado
        if player_name == white:
            color = "Brancas"
        elif player_name == black:
            color = "Pretas"
        else:
            continue  # partida não é do jogador analisado, ignora silenciosamente

        # Determina o resultado do ponto de vista do jogador analisado
        result_tag = headers.get("Result", "*")
        if result_tag == "1-0":
            outcome = "Vitória" if color == "Brancas" else "Derrota"
        elif result_tag == "0-1":
            outcome = "Vitória" if color == "Pretas" else "Derrota"
        elif result_tag == "1/2-1/2":
            outcome = "Empate"
        else:
            ignoradas["sem_resultado"] += 1
            continue

        # Extrai a lista de lances em SAN, necessária para o fallback de abertura
        try:
            board = game.board()
            moves_san = []
            for move in game.mainline_moves():
                moves_san.append(board.san(move))
                board.push(move)
        except Exception:
            ignoradas["malformada"] += 1
            continue

        # Prioriza a tag ECO+Opening do PGN; se ausente, usa o fallback por lances
        eco = headers.get("ECO", "").strip()
        opening_tag = headers.get("Opening", "").strip()

        if opening_tag:
            opening_name = f"{opening_tag}" + (f" ({eco})" if eco else "")
        elif eco:
            opening_name = f"ECO {eco}"
        elif moves_san:
            opening_name = identify_opening_by_moves(moves_san)
        else:
            ignoradas["sem_abertura_e_sem_lances"] += 1
            continue

        registros.append(
            {
                "cor": color,
                "abertura": opening_name,
                "resultado": outcome,
            }
        )

    return registros, ignoradas, total_lidas


def compute_stats(registros, color):
    """
    Calcula as estatísticas de abertura para uma cor específica ("Brancas" ou
    "Pretas") a partir da lista de registros de partidas.

    Retorna um DataFrame ordenado por número de partidas (decrescente), com
    colunas: Abertura, Nº de partidas, % do total, Vitórias, Empates,
    Derrotas, Taxa de vitória (%).
    """
    filtrados = [r for r in registros if r["cor"] == color]
    total = len(filtrados)

    if total == 0:
        return pd.DataFrame(
            columns=[
                "Abertura",
                "Nº de partidas",
                "% do total",
                "Vitórias",
                "Empates",
                "Derrotas",
                "Taxa de vitória (%)",
            ]
        ), 0

    contagem = defaultdict(lambda: {"Vitória": 0, "Empate": 0, "Derrota": 0})
    for r in filtrados:
        contagem[r["abertura"]][r["resultado"]] += 1

    linhas = []
    for abertura, res in contagem.items():
        n_partidas = res["Vitória"] + res["Empate"] + res["Derrota"]
        taxa_vitoria = (res["Vitória"] / n_partidas * 100) if n_partidas else 0.0
        linhas.append(
            {
                "Abertura": abertura,
                "Nº de partidas": n_partidas,
                "% do total": round(n_partidas / total * 100, 1),
                "Vitórias": res["Vitória"],
                "Empates": res["Empate"],
                "Derrotas": res["Derrota"],
                "Taxa de vitória (%)": round(taxa_vitoria, 1),
            }
        )

    df = pd.DataFrame(linhas).sort_values("Nº de partidas", ascending=False).reset_index(drop=True)
    return df, total


def render_color_section(color, df, total):
    """Renderiza a seção de estatísticas (tabela + gráficos) para uma cor."""
    emoji = "⚪" if color == "Brancas" else "⚫"
    st.subheader(f"{emoji} Jogando de {color}")

    if total == 0:
        st.info(f"Nenhuma partida encontrada com o jogador de {color.lower()}.")
        return

    st.caption(f"Total de {total} partida(s) analisada(s).")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Gráfico de barras: partidas por abertura (top 10)
    top_freq = df.head(10).set_index("Abertura")["Nº de partidas"]
    st.markdown("**Aberturas mais usadas**")
    st.bar_chart(top_freq)

    # Gráfico de barras: taxa de vitória por abertura (top 10, apenas com >=1 partida)
    top_taxa = df.head(10).set_index("Abertura")["Taxa de vitória (%)"]
    st.markdown("**Taxa de vitória por abertura**")
    st.bar_chart(top_taxa)


def main():
    st.set_page_config(page_title="Analisador de Aberturas de Xadrez", page_icon="♟️", layout="wide")
    st.title("♟️ Analisador de Repertório de Aberturas")
    st.write(
        "Envie um arquivo PGN com uma ou mais partidas e informe o nome exato "
        "do jogador (como aparece nas tags White/Black do PGN) para ver as "
        "aberturas mais usadas e o desempenho com cada uma, separado por cor."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("Arquivo PGN", type=["pgn"])
    with col2:
        player_name = st.text_input("Nome do jogador (exato, como no PGN)")

    if uploaded_file is None or not player_name:
        st.info("Envie um arquivo PGN e informe o nome do jogador para começar.")
        return

    with st.spinner("Analisando partidas..."):
        pgn_bytes = uploaded_file.read()
        registros, ignoradas, total_lidas = parse_pgn(pgn_bytes, player_name)

    if total_lidas == 0:
        st.error("Não foi possível ler nenhuma partida do arquivo. Verifique se o PGN está correto.")
        return

    if not registros:
        st.warning(
            f"O arquivo tem {total_lidas} partida(s), mas nenhuma corresponde exatamente "
            f"ao nome de jogador informado ('{player_name}'). Verifique se o nome está "
            "escrito exatamente como aparece no PGN (tags White/Black)."
        )
        return

    st.success(f"{len(registros)} de {total_lidas} partida(s) do arquivo pertencem a '{player_name}'.")

    # Informa partidas ignoradas, se houver
    total_ignoradas = sum(ignoradas.values())
    if total_ignoradas:
        motivos = {
            "malformada": "partida(s) malformada(s) ou não puderam ser processadas",
            "sem_resultado": "partida(s) sem resultado válido (tag Result)",
            "sem_abertura_e_sem_lances": "partida(s) sem informação de abertura e sem lances para identificar",
        }
        detalhes = "; ".join(
            f"{count} {motivos.get(motivo, motivo)}" for motivo, count in ignoradas.items()
        )
        st.warning(f"⚠️ {total_ignoradas} partida(s) foram ignoradas: {detalhes}.")

    st.divider()

    df_brancas, total_brancas = compute_stats(registros, "Brancas")
    df_pretas, total_pretas = compute_stats(registros, "Pretas")

    render_color_section("Brancas", df_brancas, total_brancas)
    st.divider()
    render_color_section("Pretas", df_pretas, total_pretas)


if __name__ == "__main__":
    main()

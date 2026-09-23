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
from collections import Counter, defaultdict

import chess
import chess.pgn
import html
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paleta "neon" usada nos gráficos de rosca (donut) de V/E/D.
# ---------------------------------------------------------------------------
COR_VITORIA = "#39FF14"   # verde neon
COR_EMPATE = "#D9D9D9"    # cinza claro (neutro, contrasta bem no fundo escuro)
COR_DERROTA = "#FF1B4C"   # vermelho/rosa neon
COR_FUNDO = "#0E1117"     # combina com o tema escuro padrão do Streamlit

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

# ---------------------------------------------------------------------------
# Tradução de código ECO -> nome da abertura, usada quando o PGN tem a tag
# ECO mas não tem a tag Opening (comum em exports do Chess.com).
# Primeiro tenta um código exato bem conhecido; se não achar, cai para uma
# faixa de códigos (ex: C60-C99 = Ruy Lopez) que cobre TODO o espectro do ECO
# com um nome genérico razoável. Isso evita mostrar só "ECO C50" ao usuário.
#
# Se você (usuário) tiver nomes mais específicos para algum código que caiu
# na faixa genérica, é só me passar o código ECO + nome que eu adiciono aqui.
# ---------------------------------------------------------------------------
ECO_EXACT_NAMES = {
    "A01": "Ataque Nimzowitsch-Larsen",
    "A02": "Abertura Bird",
    "A03": "Abertura Bird",
    "A04": "Abertura Réti",
    "A05": "Abertura Réti",
    "A07": "Réti com fianchetto do rei",
    "A10": "Abertura Inglesa",
    "A80": "Defesa Holandesa",
    "B01": "Defesa Escandinava",
    "B02": "Defesa Alekhine",
    "B06": "Defesa Moderna",
    "B07": "Defesa Pirc",
    "B10": "Defesa Caro-Kann",
    "B20": "Defesa Siciliana",
    "B90": "Defesa Siciliana Najdorf",
    "B92": "Defesa Siciliana Najdorf (variante Zagreb)",
    "C00": "Defesa Francesa",
    "C20": "Abertura do Peão do Rei",
    "C25": "Abertura Viena",
    "C30": "Gambito do Rei",
    "C42": "Defesa Petrov",
    "C44": "Abertura Escocesa",
    "C45": "Gambito Escocês",
    "C50": "Abertura Italiana",
    "C60": "Ruy Lopez (Espanhola)",
    "D00": "Abertura da Dama",
    "D06": "Gambito da Dama",
    "D10": "Defesa Eslava",
    "D20": "Gambito da Dama Aceito",
    "D30": "Gambito da Dama Recusado",
    "D70": "Defesa Grünfeld",
    "E00": "Abertura Catalã",
    "E10": "Defesa Índia da Dama",
    "E20": "Defesa Nimzo-Índia",
    "E60": "Defesa Índia do Rei",
}

# Faixas de códigos ECO (letra + intervalo numérico) -> nome genérico.
# Cobre o código A00-E99 inteiro, garantindo que sempre haja um nome.
ECO_RANGE_NAMES = [
    ("A", 0, 9, "Abertura Irregular / de Flanco"),
    ("A", 10, 39, "Abertura Inglesa"),
    ("A", 40, 44, "Abertura de Peão de Dama (irregular)"),
    ("A", 45, 49, "Defesa Índia (genérica)"),
    ("A", 50, 79, "Defesa Benoni / Índia"),
    ("A", 80, 99, "Defesa Holandesa"),
    ("B", 0, 0, "Abertura de Peão de Rei (irregular)"),
    ("B", 1, 5, "Defesa Escandinava / Alekhine"),
    ("B", 6, 9, "Defesa Pirc / Moderna"),
    ("B", 10, 19, "Defesa Caro-Kann"),
    ("B", 20, 99, "Defesa Siciliana"),
    ("C", 0, 19, "Defesa Francesa"),
    ("C", 20, 29, "Abertura do Peão do Rei / Viena"),
    ("C", 30, 39, "Gambito do Rei"),
    ("C", 40, 49, "Abertura dos Cavalos do Rei"),
    ("C", 50, 59, "Abertura Italiana"),
    ("C", 60, 99, "Ruy Lopez (Espanhola)"),
    ("D", 0, 5, "Abertura da Dama"),
    ("D", 6, 9, "Gambito da Dama (linhas variadas)"),
    ("D", 10, 19, "Defesa Eslava"),
    ("D", 20, 29, "Gambito da Dama Aceito"),
    ("D", 30, 69, "Gambito da Dama Recusado"),
    ("D", 70, 99, "Defesa Grünfeld"),
    ("E", 0, 9, "Abertura Catalã"),
    ("E", 10, 19, "Defesa Índia da Dama"),
    ("E", 20, 59, "Defesa Nimzo-Índia"),
    ("E", 60, 99, "Defesa Índia do Rei"),
]


def eco_to_name(eco):
    """
    Converte um código ECO (ex: 'C50') no nome da abertura correspondente.
    Retorna None se o código não tiver o formato esperado (letra A-E + 2 dígitos).
    """
    if not eco or len(eco) < 3 or eco[0] not in "ABCDE":
        return None
    try:
        num = int(eco[1:3])
    except ValueError:
        return None

    if eco in ECO_EXACT_NAMES:
        return ECO_EXACT_NAMES[eco]

    for letra, lo, hi, nome in ECO_RANGE_NAMES:
        if eco[0] == letra and lo <= num <= hi:
            return nome
    return None


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


def format_move_sequence(moves_san, max_plies=8):
    """
    Formata os primeiros lances de uma partida no estilo PGN legível,
    ex: "1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5" — usado para mostrar ao lado do nome
    da abertura os lances que efetivamente a definem naquela partida.
    """
    trimmed = moves_san[:max_plies]
    partes = []
    for i, lance in enumerate(trimmed):
        if i % 2 == 0:
            numero = i // 2 + 1
            partes.append(f"{numero}.{lance}")
        else:
            partes.append(lance)
    return " ".join(partes)


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
            # PGN tem o código ECO mas não o nome (comum em exports do Chess.com):
            # traduz o código para um nome de abertura em vez de mostrar só "ECO C50"
            nome_eco = eco_to_name(eco)
            opening_name = f"{nome_eco} ({eco})" if nome_eco else f"ECO {eco}"
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
                "lances": format_move_sequence(moves_san),
            }
        )

    return registros, ignoradas, total_lidas


# ---------------------------------------------------------------------------
# Ajuste estatístico (suavização de Dirichlet) para as taxas de V/E/D.
#
# Aberturas com poucas partidas podem mostrar taxas extremas (ex: 100% de
# derrota em 1 partida) que não são estatisticamente confiáveis. O ajuste
# abaixo "puxa" cada percentual em direção a uma base neutra (33,3% para
# cada resultado) com força K — quanto mais partidas a abertura tiver, menos
# esse puxão pesa, e o percentual ajustado se aproxima do percentual real.
#
# Fórmula (uma categoria i, entre V/E/D):
#   % ajustado_i = (Nº de partidas com resultado i + K/3) / (Nº total + K)
#
# Isso garante que % Vitórias ajustado + % Empates ajustado + % Derrotas
# ajustado somem sempre 100%, ao contrário de ajustar cada taxa de forma
# independente.
# ---------------------------------------------------------------------------
K_SUAVIZACAO = 5


def ajustar_percentual(n_resultado, n_total, k=K_SUAVIZACAO):
    """Calcula o percentual ajustado (suavizado) de um resultado específico."""
    return round((n_resultado + k / 3) / (n_total + k) * 100, 1)


def compute_stats(registros, color):
    """
    Calcula as estatísticas de abertura para uma cor específica ("Brancas" ou
    "Pretas") a partir da lista de registros de partidas.

    Retorna um DataFrame ordenado por número de partidas (decrescente), com
    colunas: Abertura, Lances, Nº de partidas, % do total, Vitórias,
    % Vitórias, % Vitórias (ajustado), Empates, % Empates,
    % Empates (ajustado), Derrotas, % Derrotas, % Derrotas (ajustado).
    O percentual "ajustado" pondera a taxa pelo tamanho da amostra (veja
    ajustar_percentual) para que aberturas com poucas partidas não pareçam
    artificialmente melhores/piores do que aberturas com muitas partidas.
    """
    filtrados = [r for r in registros if r["cor"] == color]
    total = len(filtrados)

    colunas = [
        "Abertura",
        "Lances",
        "Nº de partidas",
        "% do total",
        "Vitórias",
        "% Vitórias",
        "% Vitórias (ajustado)",
        "Empates",
        "% Empates",
        "% Empates (ajustado)",
        "Derrotas",
        "% Derrotas",
        "% Derrotas (ajustado)",
    ]

    if total == 0:
        return pd.DataFrame(columns=colunas), 0

    contagem = defaultdict(lambda: {"Vitória": 0, "Empate": 0, "Derrota": 0})
    lances_por_abertura = defaultdict(list)
    for r in filtrados:
        contagem[r["abertura"]][r["resultado"]] += 1
        lances_por_abertura[r["abertura"]].append(r.get("lances", ""))

    linhas = []
    for abertura, res in contagem.items():
        n_partidas = res["Vitória"] + res["Empate"] + res["Derrota"]
        pct = lambda n: round(n / n_partidas * 100, 1) if n_partidas else 0.0

        # Usa a sequência de lances mais frequente entre as partidas dessa
        # abertura como representativa (a "linha principal" jogada por ela)
        lances_validos = [l for l in lances_por_abertura[abertura] if l]
        lances_repr = Counter(lances_validos).most_common(1)[0][0] if lances_validos else ""

        linhas.append(
            {
                "Abertura": abertura,
                "Lances": lances_repr,
                "Nº de partidas": n_partidas,
                "% do total": round(n_partidas / total * 100, 1),
                "Vitórias": res["Vitória"],
                "% Vitórias": pct(res["Vitória"]),
                "% Vitórias (ajustado)": ajustar_percentual(res["Vitória"], n_partidas),
                "Empates": res["Empate"],
                "% Empates": pct(res["Empate"]),
                "% Empates (ajustado)": ajustar_percentual(res["Empate"], n_partidas),
                "Derrotas": res["Derrota"],
                "% Derrotas": pct(res["Derrota"]),
                "% Derrotas (ajustado)": ajustar_percentual(res["Derrota"], n_partidas),
            }
        )

    df = pd.DataFrame(linhas).sort_values("Nº de partidas", ascending=False).reset_index(drop=True)
    return df, total


def render_donut_grid(df, max_aberturas=6):
    """
    Desenha uma grade de gráficos de rosca (donut) em CSS puro (conic-gradient),
    em estilo "neon" sobre fundo escuro — um por abertura (as mais jogadas
    primeiro), mostrando a proporção de Vitórias (verde), Empates (cinza) e
    Derrotas (vermelho), com percentuais e os lances que definem a abertura.

    Todo o HTML é montado SEM indentação nas linhas (cada linha começa na
    coluna 0). Isso é necessário porque o parser de Markdown do Streamlit
    trata qualquer linha com 4+ espaços de indentação como um bloco de código
    literal, mesmo com unsafe_allow_html=True — foi isso que causava o bug
    dos gráficos aparecendo como texto/código na tela.
    """
    df_top = df.head(max_aberturas)
    if df_top.empty:
        return

    cards_html = []
    for _, row in df_top.iterrows():
        v, e, d = row["Vitórias"], row["Empates"], row["Derrotas"]
        n_partidas = v + e + d
        if n_partidas == 0:
            continue

        pv = v / n_partidas * 100
        pe = e / n_partidas * 100
        # pd_ (percentual de derrotas) fecha os 100% restantes do círculo
        pd_ = 100 - pv - pe

        # conic-gradient desenha o círculo repartindo os graus proporcionalmente
        # a cada faixa de percentual acumulado: verde -> cinza -> vermelho
        fim_verde = pv
        fim_cinza = pv + pe
        gradiente = (
            f"conic-gradient({COR_VITORIA} 0% {fim_verde:.2f}%, "
            f"{COR_EMPATE} {fim_verde:.2f}% {fim_cinza:.2f}%, "
            f"{COR_DERROTA} {fim_cinza:.2f}% 100%)"
        )

        titulo = html.escape(str(row["Abertura"]))
        lances = html.escape(str(row.get("Lances", "")) or "—")
        rotulo_partidas = "partida" if n_partidas == 1 else "partidas"

        # Cada linha do card começa na coluna 0 de propósito (ver docstring)
        card_lines = [
            '<div class="donut-card">',
            f'<div class="donut-circle" style="background:{gradiente};">',
            '<div class="donut-hole">',
            f'<span class="donut-total">{int(n_partidas)}</span>',
            f'<span class="donut-total-label">{rotulo_partidas}</span>',
            "</div>",
            "</div>",
            f'<div class="donut-title">{titulo}</div>',
            f'<div class="donut-moves">{lances}</div>',
            '<div class="donut-legend">',
            f'<span style="color:{COR_VITORIA}">&#9679; {pv:.0f}% V</span>',
            f'<span style="color:{COR_EMPATE}">&#9679; {pe:.0f}% E</span>',
            f'<span style="color:{COR_DERROTA}">&#9679; {pd_:.0f}% D</span>',
            "</div>",
            "</div>",
        ]
        cards_html.append("".join(card_lines))

    if not cards_html:
        return

    estilo_linhas = [
        "<style>",
        ".donut-grid { display: flex; flex-wrap: wrap; gap: 18px; margin: 8px 0 24px 0; }",
        f'.donut-card {{ background: {COR_FUNDO}; border: 1px solid #2a2e39; border-radius: 14px; '
        "padding: 16px; width: 190px; display: flex; flex-direction: column; "
        "align-items: center; box-shadow: 0 0 16px rgba(57, 255, 20, 0.06); }",
        ".donut-circle { width: 128px; height: 128px; border-radius: 50%; "
        "display: flex; align-items: center; justify-content: center; "
        "box-shadow: 0 0 14px 1px rgba(255, 255, 255, 0.08); }",
        f'.donut-hole {{ width: 76px; height: 76px; background: {COR_FUNDO}; border-radius: 50%; '
        "display: flex; flex-direction: column; align-items: center; justify-content: center; }",
        ".donut-total { color: white; font-weight: 700; font-size: 18px; line-height: 1.1; }",
        ".donut-total-label { color: #9aa0a6; font-size: 10px; }",
        ".donut-title { color: white; font-size: 13px; font-weight: 600; text-align: center; "
        "margin-top: 10px; min-height: 34px; }",
        '.donut-moves { color: #9aa0a6; font-size: 11px; font-family: "Courier New", monospace; '
        "text-align: center; margin-top: 4px; min-height: 30px; }",
        ".donut-legend { display: flex; gap: 8px; margin-top: 8px; font-size: 11px; color: #ddd; "
        "white-space: nowrap; }",
        "</style>",
    ]
    estilo = "".join(estilo_linhas)
    grid = '<div class="donut-grid">' + "".join(cards_html) + "</div>"
    st.markdown(estilo + grid, unsafe_allow_html=True)


def render_color_section(color, df, total):
    """Renderiza a seção de estatísticas (tabela + gráficos) para uma cor."""
    emoji = "⚪" if color == "Brancas" else "⚫"
    st.subheader(f"{emoji} Jogando de {color}")

    if total == 0:
        st.info(f"Nenhuma partida encontrada com o jogador de {color.lower()}.")
        return

    st.caption(f"Total de {total} partida(s) analisada(s).")
    st.caption(
        "As colunas \"(ajustado)\" ponderam a taxa pelo número de partidas "
        "(quanto menos partidas, mais o percentual é puxado para perto de 33,3%) "
        "— evita que uma abertura com poucas partidas pareça melhor/pior do que "
        "realmente é. Clique no cabeçalho de qualquer coluna para ordenar por ela."
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Gráfico de barras: partidas por abertura (top 10)
    top_freq = df.head(10).set_index("Abertura")["Nº de partidas"]
    st.markdown("**Aberturas mais usadas**")
    st.bar_chart(top_freq)

    # Gráficos de rosca (donut) neon: proporção de V/E/D nas aberturas mais usadas
    st.markdown("**Desempenho por abertura (Vitória / Empate / Derrota)**")
    if len(df) > 6:
        st.caption("Mostrando as 6 aberturas mais jogadas.")
    render_donut_grid(df)


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

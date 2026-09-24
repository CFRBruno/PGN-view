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
# Paleta de cores da interface — tons modernos e suaves (não neon puro),
# confortáveis para leitura prolongada sobre fundo escuro.
# ---------------------------------------------------------------------------
COR_VITORIA = "#34D399"    # verde esmeralda suave
COR_EMPATE = "#94A3B8"     # cinza-azulado suave
COR_DERROTA = "#FB7185"    # rosa/coral suave
COR_FUNDO = "#161B22"      # fundo escuro (não preto puro)
COR_CARD_BORDA = "#2A3140"
COR_TEXTO_SECUNDARIO = "#9CA3AF"
COR_TRILHA_BARRA = "#232935"   # fundo (trilha) das barras horizontais
COR_ACCENT = "#60A5FA"     # azul suave, usado nas barras de frequência de aberturas

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


def categorizar_metodo_por_texto(termination_text):
    """
    Categoriza o método de finalização a partir do texto da tag PGN
    'Termination' (usada quando o tabuleiro final não é suficiente para
    identificar o método sozinho — ex: desistência, tempo, acordo mútuo).
    Cobre tanto termos em inglês quanto em português, já que o formato varia
    entre plataformas e idiomas (ex: Chess.com em pt-BR usa "venceu por
    desistência", "venceu por tempo", "Empate por repetição" etc.).
    """
    if not termination_text:
        return "Não informado"
    t = termination_text.lower()
    if "tempo" in t or "time" in t:
        return "Tempo (flag caiu)"
    if "desist" in t or "resign" in t:
        return "Desistência"
    if "abandon" in t:
        return "Abandono"
    if "acordo" in t or "agree" in t:
        return "Acordo mútuo"
    if "mate" in t:
        return "Xeque-mate"
    if "repet" in t:
        return "Repetição de posição"
    if "afoga" in t or "stalemate" in t:
        return "Afogamento"
    if "insuficiente" in t or "insufficient" in t:
        return "Material insuficiente"
    if "50" in t or "fifty" in t:
        return "Regra dos 50 lances"
    if "infra" in t or "disqualif" in t or "desqualific" in t or "trapa" in t or "cheat" in t:
        return "Infração/Desqualificação"
    if t.strip() == "normal":
        return "Não especificado"
    return "Outro"


def determinar_metodo(board, termination_text):
    """
    Determina o método de finalização da partida. Prioriza o estado final
    real do tabuleiro (xeque-mate, afogamento, material insuficiente,
    repetição/75 lances automáticos) — que é 100% confiável, pois vem do
    replay real dos lances — e só recorre ao texto da tag Termination do
    PGN para os casos que o tabuleiro sozinho não revela (desistência,
    tempo esgotado, acordo mútuo, abandono).
    """
    if board.is_checkmate():
        return "Xeque-mate"
    if board.is_stalemate():
        return "Afogamento"
    if board.is_insufficient_material():
        return "Material insuficiente"
    if board.is_seventyfive_moves():
        return "Regra dos 75 lances"
    if board.is_fivefold_repetition():
        return "Repetição de posição"
    return categorizar_metodo_por_texto(termination_text)


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

        # Identifica como a partida terminou (xeque-mate, tempo, abandono etc.)
        metodo = determinar_metodo(board, headers.get("Termination", "").strip())

        # Guarda adversário e data — usados no filtro "ver partidas desta abertura"
        adversario = black if color == "Brancas" else white
        data_partida = headers.get("Date", "").strip().replace("?", "").strip(".") or "—"

        registros.append(
            {
                "cor": color,
                "abertura": opening_name,
                "resultado": outcome,
                "lances": format_move_sequence(moves_san),
                "metodo": metodo,
                "adversario": adversario or "—",
                "data": data_partida,
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


def compute_stats_all(registros):
    """
    Mesma lógica de compute_stats, mas somando Brancas + Pretas juntas
    (usado na aba de comparação entre jogadores, onde queremos um resumo
    geral em vez de separado por cor).
    """
    registros_unificados = [dict(r, cor="Todas") for r in registros]
    return compute_stats(registros_unificados, "Todas")


def compute_method_stats(registros, color):
    """
    Agrega, para uma cor específica, como as vitórias, empates e derrotas
    terminaram (xeque-mate, tempo, abandono, afogamento etc.), somando todas
    as aberturas juntas.

    Retorna um dict {"Vitória": {...}, "Empate": {...}, "Derrota": {...}},
    onde cada valor é {"total": N, "itens": [{"metodo", "n", "pct"}, ...]}
    ordenado do método mais comum para o menos comum.
    """
    filtrados = [r for r in registros if r["cor"] == color]
    por_resultado = defaultdict(Counter)
    for r in filtrados:
        por_resultado[r["resultado"]][r.get("metodo", "Não informado")] += 1

    saida = {}
    for resultado in ("Vitória", "Empate", "Derrota"):
        contagem = por_resultado.get(resultado, Counter())
        total_resultado = sum(contagem.values())
        itens = [
            {
                "metodo": metodo,
                "n": n,
                "pct": round(n / total_resultado * 100, 1) if total_resultado else 0.0,
            }
            for metodo, n in contagem.most_common()
        ]
        saida[resultado] = {"total": total_resultado, "itens": itens}
    return saida


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
        f'.donut-card {{ background: {COR_FUNDO}; border: 1px solid {COR_CARD_BORDA}; border-radius: 16px; '
        "padding: 16px; width: 190px; display: flex; flex-direction: column; "
        "align-items: center; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.30); }",
        ".donut-circle { width: 128px; height: 128px; border-radius: 50%; "
        "display: flex; align-items: center; justify-content: center; }",
        f'.donut-hole {{ width: 76px; height: 76px; background: {COR_FUNDO}; border-radius: 50%; '
        "display: flex; flex-direction: column; align-items: center; justify-content: center; }",
        ".donut-total { color: white; font-weight: 700; font-size: 18px; line-height: 1.1; }",
        f'.donut-total-label {{ color: {COR_TEXTO_SECUNDARIO}; font-size: 10px; }}',
        ".donut-title { color: white; font-size: 13px; font-weight: 600; text-align: center; "
        "margin-top: 10px; min-height: 34px; }",
        f'.donut-moves {{ color: {COR_TEXTO_SECUNDARIO}; font-size: 11px; font-family: "Courier New", monospace; '
        "text-align: center; margin-top: 4px; min-height: 30px; }",
        ".donut-legend { display: flex; gap: 8px; margin-top: 8px; font-size: 11px; color: #ddd; "
        "white-space: nowrap; }",
        "</style>",
    ]
    estilo = "".join(estilo_linhas)
    grid = '<div class="donut-grid">' + "".join(cards_html) + "</div>"
    st.markdown(estilo + grid, unsafe_allow_html=True)


def render_barras_horizontais(itens, cor_barra, unidade="", bicolor=False):
    """
    Desenha uma lista de barras horizontais em CSS puro — usada tanto para
    "aberturas mais usadas" quanto para "métodos de finalização". Muito mais
    legível que um gráfico de colunas quando os rótulos (nomes de aberturas,
    métodos) são longos, pois o texto fica na horizontal, sem cortar/rotacionar.

    `itens`: lista de dicts {"label": str, "valor": num, "pct": num}. No modo
    `bicolor=True`, cada item também precisa de "pct_vitoria" e "pct_derrota"
    (0-100): o comprimento total da barra continua representando a frequência,
    mas o preenchimento é dividido em verde (% vitória) e vermelho (% derrota)
    daquela abertura, nas mesmas cores dos donuts.
    `cor_barra`: cor de preenchimento das barras (ignorado se bicolor=True).
    `unidade`: texto mostrado ao lado do valor (ex: "partidas")
    """
    if not itens:
        return
    maior_valor = max(item["valor"] for item in itens) or 1

    linhas = ["<style>",
        ".barra-lista { display: flex; flex-direction: column; gap: 10px; margin: 10px 0 22px 0; }",
        ".barra-item { display: flex; align-items: center; gap: 12px; }",
        ".barra-label { width: 220px; flex-shrink: 0; color: #E5E7EB; font-size: 13px; "
        "white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }",
        f'.barra-track {{ flex: 1; background: {COR_TRILHA_BARRA}; border-radius: 8px; height: 16px; overflow: hidden; display: flex; }}',
        ".barra-fill { height: 100%; }",
        f'.barra-valor {{ min-width: 92px; text-align: right; color: {COR_TEXTO_SECUNDARIO}; font-size: 12px; white-space: nowrap; }}',
        "</style>"]

    linhas.append('<div class="barra-lista">')
    for item in itens:
        largura_pct = max(item["valor"] / maior_valor * 100, 3)  # mínimo visível
        label = html.escape(str(item["label"]))
        texto_valor = f'{item["valor"]}{unidade}'
        if "pct" in item:
            texto_valor += f' &middot; {item["pct"]:.1f}%'
        linhas.append('<div class="barra-item">')
        linhas.append(f'<div class="barra-label" title="{label}">{label}</div>')
        linhas.append('<div class="barra-track">')
        if bicolor:
            # Divide o comprimento total da barra (largura_pct) proporcionalmente
            # entre vitória (verde) e derrota (vermelho) daquela abertura
            pv = item.get("pct_vitoria", 0)
            pd_ = item.get("pct_derrota", 0)
            largura_v = largura_pct * (pv / 100)
            largura_d = largura_pct * (pd_ / 100)
            linhas.append(f'<div class="barra-fill" style="width:{largura_v:.1f}%; background:{COR_VITORIA}; border-radius:8px 0 0 8px;"></div>')
            linhas.append(f'<div class="barra-fill" style="width:{largura_d:.1f}%; background:{COR_DERROTA}; border-radius:0 8px 8px 0;"></div>')
        else:
            linhas.append(f'<div class="barra-fill" style="width:{largura_pct:.1f}%; background:{cor_barra}; border-radius:8px;"></div>')
        linhas.append("</div>")
        linhas.append(f'<div class="barra-valor">{texto_valor}</div>')
        linhas.append("</div>")
    linhas.append("</div>")

    st.markdown("".join(linhas), unsafe_allow_html=True)


def render_metodos_section(method_stats):
    """
    Renderiza, para uma cor já filtrada, como vitórias/empates/derrotas
    terminaram (xeque-mate, tempo, abandono etc.), uma seção por resultado.
    """
    rotulos = {
        "Vitória": ("🏆 Como as vitórias aconteceram", COR_VITORIA),
        "Empate": ("⚖️ Como os empates aconteceram", COR_EMPATE),
        "Derrota": ("💀 Como as derrotas aconteceram", COR_DERROTA),
    }
    colunas = st.columns(3)
    for coluna, resultado in zip(colunas, ("Vitória", "Empate", "Derrota")):
        titulo, cor = rotulos[resultado]
        dados = method_stats.get(resultado, {"total": 0, "itens": []})
        with coluna:
            st.markdown(f"**{titulo}**")
            if dados["total"] == 0:
                st.caption("Sem dados.")
                continue
            itens = [
                {"label": item["metodo"], "valor": item["n"], "pct": item["pct"]}
                for item in dados["itens"]
            ]
            render_barras_horizontais(itens, cor_barra=cor)


def render_color_section(color, df, total, method_stats, registros_cor):
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

    # Frequência de uso das aberturas: barras horizontais em CSS (mais legível
    # que um gráfico de colunas quando os nomes das aberturas são longos)
    col_titulo, col_toggle = st.columns([3, 2])
    with col_titulo:
        st.markdown("**Aberturas mais usadas**")
    with col_toggle:
        bicolor = st.toggle(
            "Colorir por desempenho (Vitória/Derrota)",
            key=f"toggle_bicolor_{color}",
        )

    top_freq = df.head(10)
    if bicolor:
        itens_freq = [
            {
                "label": row["Abertura"],
                "valor": int(row["Nº de partidas"]),
                "pct_vitoria": row["% Vitórias"],
                "pct_derrota": row["% Derrotas"],
            }
            for _, row in top_freq.iterrows()
        ]
        render_barras_horizontais(itens_freq, cor_barra=COR_ACCENT, unidade=" partidas", bicolor=True)
        st.caption(
            "Cada barra mantém o comprimento total (frequência), mas o preenchimento "
            "é dividido em verde (% vitória) e vermelho (% derrota) daquela abertura."
        )
    else:
        itens_freq = [
            {"label": row["Abertura"], "valor": int(row["Nº de partidas"]), "pct": row["% do total"]}
            for _, row in top_freq.iterrows()
        ]
        render_barras_horizontais(itens_freq, cor_barra=COR_ACCENT, unidade=" partidas")

    # Gráficos de rosca (donut): proporção de V/E/D nas aberturas mais usadas
    st.markdown("**Desempenho por abertura (Vitória / Empate / Derrota)**")
    if len(df) > 6:
        st.caption("Mostrando as 6 aberturas mais jogadas.")
    render_donut_grid(df)

    # Como as partidas terminaram (xeque-mate, tempo, abandono etc.)
    st.markdown("**Método de finalização**")
    render_metodos_section(method_stats)

    # Filtro: ver as partidas de uma abertura específica
    st.markdown("**🔍 Ver partidas de uma abertura específica**")
    opcoes = ["Selecione uma abertura..."] + list(df["Abertura"])
    escolha = st.selectbox(
        "Abertura", opcoes, key=f"sel_abertura_{color}", label_visibility="collapsed"
    )
    if escolha != opcoes[0]:
        partidas_da_abertura = [r for r in registros_cor if r["abertura"] == escolha]
        tabela_partidas = pd.DataFrame(
            [
                {
                    "Data": r.get("data", "—"),
                    "Adversário": r.get("adversario", "—"),
                    "Resultado": r["resultado"],
                    "Método": r.get("metodo", "—"),
                    "Lances": r.get("lances", ""),
                }
                for r in partidas_da_abertura
            ]
        )
        st.caption(f"{len(partidas_da_abertura)} partida(s) com esta abertura.")
        st.dataframe(tabela_partidas, use_container_width=True, hide_index=True)


def render_compare_tab():
    """
    Aba de comparação: recebe até 2 arquivos PGN (pode ser o mesmo arquivo
    duas vezes) e o nome de 2 jogadores, e mostra um resumo geral
    (Brancas+Pretas juntas) lado a lado.
    """
    st.write(
        "Envie um PGN e o nome de cada jogador para comparar o repertório "
        "geral dos dois lado a lado (pode ser o mesmo arquivo, com dois "
        "nomes diferentes)."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Jogador A**")
        arquivo_a = st.file_uploader("PGN — Jogador A", type=["pgn"], key="cmp_file_a")
        nome_a = st.text_input("Nome exato — Jogador A", key="cmp_name_a")
    with col_b:
        st.markdown("**Jogador B**")
        arquivo_b = st.file_uploader("PGN — Jogador B", type=["pgn"], key="cmp_file_b")
        nome_b = st.text_input("Nome exato — Jogador B", key="cmp_name_b")

    if not (arquivo_a and nome_a and arquivo_b and nome_b):
        st.info("Preencha o arquivo e o nome dos dois jogadores para comparar.")
        return

    with st.spinner("Comparando..."):
        regs_a, _, total_lidas_a = parse_pgn(arquivo_a.read(), nome_a)
        regs_b, _, total_lidas_b = parse_pgn(arquivo_b.read(), nome_b)

    if not regs_a or not regs_b:
        faltando = nome_a if not regs_a else nome_b
        st.warning(f"Não encontrei partidas de '{faltando}' no PGN correspondente. Confira o nome exato.")
        return

    df_a, total_a = compute_stats_all(regs_a)
    df_b, total_b = compute_stats_all(regs_b)

    def resumo_geral(registros):
        c = Counter(r["resultado"] for r in registros)
        total = len(registros)
        return {
            "total": total,
            "v": c.get("Vitória", 0),
            "e": c.get("Empate", 0),
            "d": c.get("Derrota", 0),
            "pct_v": round(c.get("Vitória", 0) / total * 100, 1) if total else 0,
        }

    resumo_a, resumo_b = resumo_geral(regs_a), resumo_geral(regs_b)

    st.divider()
    col_a, col_b = st.columns(2)
    for coluna, nome, resumo, df_j in ((col_a, nome_a, resumo_a, df_a), (col_b, nome_b, resumo_b, df_b)):
        with coluna:
            st.subheader(nome)
            st.metric("Partidas analisadas", resumo["total"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Vitórias", resumo["v"], f'{resumo["pct_v"]}%')
            m2.metric("Empates", resumo["e"])
            m3.metric("Derrotas", resumo["d"])
            st.markdown("**Aberturas mais usadas**")
            top5 = df_j.head(5)
            if top5.empty:
                st.caption("Sem dados suficientes.")
            else:
                itens = [
                    {"label": row["Abertura"], "valor": int(row["Nº de partidas"]), "pct": row["% do total"]}
                    for _, row in top5.iterrows()
                ]
                render_barras_horizontais(itens, cor_barra=COR_ACCENT, unidade=" partidas")

    # Comparação direta das aberturas em comum
    aberturas_comuns = set(df_a["Abertura"]) & set(df_b["Abertura"])
    if aberturas_comuns:
        st.markdown("**Aberturas que os dois jogam**")
        linhas = []
        for abertura in aberturas_comuns:
            la = df_a[df_a["Abertura"] == abertura].iloc[0]
            lb = df_b[df_b["Abertura"] == abertura].iloc[0]
            linhas.append(
                {
                    "Abertura": abertura,
                    f"{nome_a} — partidas": int(la["Nº de partidas"]),
                    f"{nome_a} — % vitórias": la["% Vitórias"],
                    f"{nome_b} — partidas": int(lb["Nº de partidas"]),
                    f"{nome_b} — % vitórias": lb["% Vitórias"],
                }
            )
        st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)


def inject_theme():
    """
    Injeta CSS para dar uma identidade visual mais forte ao app: fundo
    escuro em gradiente, "manchas" de cor com leve desfoque (aurora) que se
    movem sozinhas, uma textura de tabuleiro bem sutil e peças de xadrez
    decorativas ao fundo.

    Observação técnica: um parallax de verdade (ligado à posição de scroll,
    via JavaScript) não é confiável dentro do Streamlit — o `st.markdown`
    insere HTML no DOM, mas o navegador não executa tags <script> inseridas
    dessa forma por segurança. Por isso o efeito de profundidade aqui é
    puramente em CSS (animações @keyframes independentes de scroll), o que
    é mais simples e roda de forma confiável.
    Como isso depende de seletores internos do Streamlit (".stApp"), pode
    variar ou quebrar em versões futuras do Streamlit.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {COR_FUNDO}; }}

        .bg-checker {{
            position: fixed; inset: 0; z-index: -4; pointer-events: none;
            background-image:
                linear-gradient(45deg, rgba(255,255,255,0.02) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.02) 75%),
                linear-gradient(45deg, rgba(255,255,255,0.02) 25%, transparent 25%, transparent 75%, rgba(255,255,255,0.02) 75%);
            background-size: 64px 64px;
            background-position: 0 0, 32px 32px;
        }}

        .aurora-blob {{
            position: fixed; border-radius: 50%; filter: blur(70px);
            pointer-events: none; z-index: -3;
            animation: drift 24s ease-in-out infinite alternate;
        }}
        .blob1 {{ width: 480px; height: 480px; top: -120px; left: -100px; background: radial-gradient(circle, rgba(167,139,250,0.28), transparent 70%); }}
        .blob2 {{ width: 560px; height: 560px; top: 40vh; right: -160px; background: radial-gradient(circle, rgba(96,165,250,0.22), transparent 70%); animation-delay: -8s; }}
        .blob3 {{ width: 460px; height: 460px; bottom: -140px; left: 10%; background: radial-gradient(circle, rgba(52,211,153,0.18), transparent 70%); animation-delay: -14s; }}
        @keyframes drift {{
            0%   {{ transform: translate(0, 0) scale(1); }}
            50%  {{ transform: translate(30px, -20px) scale(1.06); }}
            100% {{ transform: translate(-25px, 18px) scale(0.96); }}
        }}

        .chess-deco {{
            position: fixed; z-index: -2; pointer-events: none; user-select: none;
            line-height: 1; color: rgba(167, 139, 250, 0.06);
        }}
        .chess-deco.n2 {{ color: rgba(96, 165, 250, 0.06); }}
        .chess-deco.n3 {{ color: rgba(52, 211, 153, 0.055); }}

        h1, .hero-title {{
            background: linear-gradient(90deg, #FFFFFF, #A78BFA);
            -webkit-background-clip: text; background-clip: text; color: transparent !important;
        }}
        </style>

        <div class="bg-checker"></div>
        <div class="aurora-blob blob1"></div>
        <div class="aurora-blob blob2"></div>
        <div class="aurora-blob blob3"></div>

        <span class="chess-deco" style="top:2%; left:4%; font-size:130px; transform:rotate(-8deg);">♞</span>
        <span class="chess-deco n2" style="top:60%; left:85%; font-size:150px; transform:rotate(10deg);">♜</span>
        <span class="chess-deco n3" style="top:80%; left:8%; font-size:110px; transform:rotate(6deg);">♟</span>
        <span class="chess-deco n2" style="top:25%; left:92%; font-size:90px; transform:rotate(-6deg);">♗</span>
        <span class="chess-deco" style="top:45%; left:2%; font-size:100px; transform:rotate(5deg);">♛</span>
        """,
        unsafe_allow_html=True,
    )


def render_single_player_tab():
    """Aba original: analisar um único jogador, separado por cor."""
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
    metodos_brancas = compute_method_stats(registros, "Brancas")
    metodos_pretas = compute_method_stats(registros, "Pretas")
    registros_brancas = [r for r in registros if r["cor"] == "Brancas"]
    registros_pretas = [r for r in registros if r["cor"] == "Pretas"]

    render_color_section("Brancas", df_brancas, total_brancas, metodos_brancas, registros_brancas)
    st.divider()
    render_color_section("Pretas", df_pretas, total_pretas, metodos_pretas, registros_pretas)


def main():
    st.set_page_config(page_title="Analisador de Aberturas de Xadrez", page_icon="♟️", layout="wide")
    inject_theme()
    st.title("♟️ Analisador de Repertório de Aberturas")

    aba_individual, aba_comparar = st.tabs(["Analisar jogador", "⚔️ Comparar jogadores"])
    with aba_individual:
        render_single_player_tab()
    with aba_comparar:
        render_compare_tab()


if __name__ == "__main__":
    main()

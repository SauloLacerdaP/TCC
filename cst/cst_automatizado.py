from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import comb
import re
import time


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ORDEM_CST = 6

# Apenas para reconstrução/validação visual
PONTOS_RECONSTRUCAO = 120

# Limites de aviso
RMSE_WARNING = 0.0020
MAX_ERROR_WARNING = 0.0100

# Coeficientes CST absurdamente elevados podem indicar
# problema na geometria ou no parsing
COEFF_WARNING = 5.0

# Cordas extremamente diferentes de 1 podem indicar
# formato interpretado incorretamente
CHORD_MIN_WARNING = 0.8
CHORD_MAX_WARNING = 1.2


# ============================================================
# 1. CST
# ============================================================

def class_function(x, n1=0.5, n2=1.0):
    """
    Função classe do CST.

    C(x) = x^N1 (1-x)^N2
    """

    x = np.asarray(x, dtype=float)

    return (
        (x ** n1)
        *
        ((1.0 - x) ** n2)
    )


def bernstein_poly(i, n, x):
    """
    Polinômio de Bernstein.
    """

    x = np.asarray(x, dtype=float)

    return (
        comb(n, i)
        *
        (x ** i)
        *
        ((1.0 - x) ** (n - i))
    )


def matriz_cst(x, ordem):
    """
    Matriz do sistema CST.
    """

    x = np.asarray(x, dtype=float)

    C = class_function(x)

    return np.column_stack([
        C * bernstein_poly(i, ordem, x)
        for i in range(ordem + 1)
    ])


def avaliar_cst(x, coeffs, delta_te):
    """
    Reconstrói a superfície.

    y = C(x) S(x) + x Delta_TE
    """

    x = np.asarray(x, dtype=float)

    ordem = len(coeffs) - 1

    B = matriz_cst(
        x,
        ordem
    )

    return (
        B @ np.asarray(coeffs)
        +
        x * delta_te
    )


# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================

def limpar_linha(linha):
    """
    Substitui separadores comuns.
    """

    return (
        linha
        .replace(",", " ")
        .replace("\t", " ")
        .strip()
    )


def tentar_float(valor):
    """
    Tenta converter string para float.
    """

    try:
        return float(valor)

    except Exception:
        return None


def parece_contagem_pontos(x, y):
    """
    Detecta linhas do tipo:

        61 61
        33 35

    comuns no formato Lednicer.

    Só considera contagem se ambos forem praticamente inteiros
    e claramente maiores que coordenadas típicas de aerofólios.
    """

    if not np.isfinite(x) or not np.isfinite(y):
        return False

    inteiro_x = abs(x - round(x)) < 1e-10
    inteiro_y = abs(y - round(y)) < 1e-10

    if not (inteiro_x and inteiro_y):
        return False

    if x >= 5 and y >= 5:
        return True

    return False


# ============================================================
# 3. LEITURA BRUTA DO DAT
# ============================================================

def ler_dat_bruto(arquivo_dat):
    """
    Lê todas as linhas do .dat.

    Mantém apenas linhas contendo dois números.
    Também registra linhas vazias para ajudar
    na detecção de Lednicer.
    """

    arquivo_dat = Path(
        arquivo_dat
    )

    linhas_processadas = []

    with open(
        arquivo_dat,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as arquivo:

        for numero_linha, linha in enumerate(
            arquivo,
            start=1
        ):

            limpa = limpar_linha(
                linha
            )

            if limpa == "":

                linhas_processadas.append({
                    "linha": numero_linha,
                    "tipo": "vazia"
                })

                continue

            partes = limpa.split()

            if len(partes) < 2:
                continue

            x = tentar_float(
                partes[0]
            )

            y = tentar_float(
                partes[1]
            )

            if x is None or y is None:
                continue

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            if parece_contagem_pontos(x, y):

                linhas_processadas.append({
                    "linha": numero_linha,
                    "tipo": "contagem",
                    "x": x,
                    "y": y
                })

            else:

                linhas_processadas.append({
                    "linha": numero_linha,
                    "tipo": "ponto",
                    "x": x,
                    "y": y
                })

    return linhas_processadas


# ============================================================
# 4. DETECÇÃO DE FORMATO
# ============================================================

def detectar_formato(linhas):
    """
    Tenta distinguir Selig de Lednicer.

    Selig:
        TE upper -> LE -> TE lower

    Lednicer:
        duas superfícies separadas,
        frequentemente com linha de contagem
        ou linha vazia.
    """

    contagens = [
        item
        for item in linhas
        if item["tipo"] == "contagem"
    ]

    if contagens:
        return "lednicer"

    pontos = [
        item
        for item in linhas
        if item["tipo"] == "ponto"
    ]

    if len(pontos) < 10:
        raise ValueError(
            "Poucos pontos numéricos válidos."
        )

    coords = np.array(
        [
            [p["x"], p["y"]]
            for p in pontos
        ],
        dtype=float
    )

    x = coords[:, 0]

    idx_min = np.argmin(
        x
    )

    # Caso clássico Selig:
    # mínimo x ocorre no interior do arquivo
    if 2 <= idx_min <= len(x) - 3:

        return "selig"

    # Caso contrário, provavelmente duas superfícies
    return "lednicer"


# ============================================================
# 5. PARSER SELIG
# ============================================================

def parse_selig(linhas):
    """
    Converte um arquivo Selig em:

        upper
        lower
    """

    coords = np.array(
        [
            [item["x"], item["y"]]
            for item in linhas
            if item["tipo"] == "ponto"
        ],
        dtype=float
    )

    if len(coords) < 10:

        raise ValueError(
            "Arquivo Selig com poucos pontos."
        )

    idx_le = np.argmin(
        coords[:, 0]
    )

    upper = coords[
        :idx_le + 1
    ]

    lower = coords[
        idx_le:
    ]

    if (
        len(upper) < 4
        or
        len(lower) < 4
    ):

        raise ValueError(
            "Separação Selig inválida."
        )

    return upper, lower


# ============================================================
# 6. PARSER LEDNICER
# ============================================================

def parse_lednicer(linhas):
    """
    Parser robusto para Lednicer.

    Estratégias:
    1. usa linha de contagem de pontos, se existir;
    2. caso contrário, procura separação por linha vazia;
    3. caso ainda não funcione, identifica reinício de x.
    """

    # --------------------------------------------------------
    # CASO 1: linha de contagem
    # --------------------------------------------------------

    indices_contagem = [
        i
        for i, item in enumerate(linhas)
        if item["tipo"] == "contagem"
    ]

    if indices_contagem:

        idx = indices_contagem[0]

        item = linhas[idx]

        n1 = int(
            round(item["x"])
        )

        n2 = int(
            round(item["y"])
        )

        pontos_depois = [
            p
            for p in linhas[idx + 1:]
            if p["tipo"] == "ponto"
        ]

        if len(pontos_depois) >= n1 + n2:

            superficie_1 = np.array(
                [
                    [p["x"], p["y"]]
                    for p in pontos_depois[:n1]
                ],
                dtype=float
            )

            superficie_2 = np.array(
                [
                    [p["x"], p["y"]]
                    for p in pontos_depois[n1:n1 + n2]
                ],
                dtype=float
            )

            return classificar_upper_lower(
                superficie_1,
                superficie_2
            )

    # --------------------------------------------------------
    # CASO 2: blocos separados por linha vazia
    # --------------------------------------------------------

    blocos = []
    bloco_atual = []

    for item in linhas:

        if item["tipo"] == "ponto":

            bloco_atual.append(
                [item["x"], item["y"]]
            )

        elif item["tipo"] == "vazia":

            if len(bloco_atual) >= 4:

                blocos.append(
                    np.array(
                        bloco_atual,
                        dtype=float
                    )
                )

                bloco_atual = []

    if len(bloco_atual) >= 4:

        blocos.append(
            np.array(
                bloco_atual,
                dtype=float
            )
        )

    if len(blocos) >= 2:

        # Pega os dois maiores blocos
        blocos = sorted(
            blocos,
            key=len,
            reverse=True
        )

        return classificar_upper_lower(
            blocos[0],
            blocos[1]
        )

    # --------------------------------------------------------
    # CASO 3: tentativa automática pelo padrão x
    # --------------------------------------------------------

    coords = np.array(
        [
            [item["x"], item["y"]]
            for item in linhas
            if item["tipo"] == "ponto"
        ],
        dtype=float
    )

    if len(coords) < 10:

        raise ValueError(
            "Lednicer com poucos pontos."
        )

    x = coords[:, 0]

    # Procura maior salto/reinício na sequência de x
    dx = np.abs(
        np.diff(x)
    )

    # Evita extremos
    candidatos = range(
        3,
        len(coords) - 3
    )

    melhor_idx = None
    melhor_score = -np.inf

    for i in candidatos:

        parte1 = coords[:i]
        parte2 = coords[i:]

        if len(parte1) < 4 or len(parte2) < 4:
            continue

        faixa1 = np.ptp(
            parte1[:, 0]
        )

        faixa2 = np.ptp(
            parte2[:, 0]
        )

        score = (
            faixa1
            +
            faixa2
        )

        # prefere divisões próximas ao centro
        equilibrio = min(
            len(parte1),
            len(parte2)
        ) / max(
            len(parte1),
            len(parte2)
        )

        score *= equilibrio

        if score > melhor_score:

            melhor_score = score
            melhor_idx = i

    if melhor_idx is None:

        raise ValueError(
            "Não foi possível separar superfícies Lednicer."
        )

    superficie_1 = coords[
        :melhor_idx
    ]

    superficie_2 = coords[
        melhor_idx:
    ]

    return classificar_upper_lower(
        superficie_1,
        superficie_2
    )


# ============================================================
# 7. CLASSIFICAÇÃO UPPER / LOWER
# ============================================================

def classificar_upper_lower(
    superficie_1,
    superficie_2
):
    """
    Define automaticamente qual superfície é upper/lower.
    """

    y1 = np.mean(
        superficie_1[:, 1]
    )

    y2 = np.mean(
        superficie_2[:, 1]
    )

    if y1 >= y2:

        upper = superficie_1
        lower = superficie_2

    else:

        upper = superficie_2
        lower = superficie_1

    return upper, lower


# ============================================================
# 8. ORIENTAÇÃO DAS SUPERFÍCIES
# ============================================================

def ordenar_superficie_por_x(superficie):
    """
    Ordena uma superfície por x crescente.
    """

    superficie = np.asarray(
        superficie,
        dtype=float
    )

    ordem = np.argsort(
        superficie[:, 0]
    )

    return superficie[
        ordem
    ]


# ============================================================
# 9. NORMALIZAÇÃO GEOMÉTRICA
# ============================================================

def normalizar_superficies(
    upper,
    lower
):
    """
    Normaliza conjuntamente extradorso e intradorso.

    Define:
        LE = ponto médio dos pontos de menor x
        TE = ponto médio dos pontos de maior x

    Depois:
        translada
        rotaciona
        normaliza pela corda
    """

    upper = np.asarray(
        upper,
        dtype=float
    )

    lower = np.asarray(
        lower,
        dtype=float
    )

    todos = np.vstack([
        upper,
        lower
    ])

    # --------------------------------------------------------
    # Leading edge
    # --------------------------------------------------------

    min_x = np.min(
        todos[:, 0]
    )

    tolerancia_x = max(
        1e-8,
        np.ptp(todos[:, 0]) * 0.002
    )

    candidatos_le = todos[
        np.abs(
            todos[:, 0] - min_x
        ) <= tolerancia_x
    ]

    le = np.mean(
        candidatos_le,
        axis=0
    )

    # --------------------------------------------------------
    # Trailing edge
    # --------------------------------------------------------

    max_x = np.max(
        todos[:, 0]
    )

    candidatos_te = todos[
        np.abs(
            todos[:, 0] - max_x
        ) <= tolerancia_x
    ]

    te = np.mean(
        candidatos_te,
        axis=0
    )

    vetor_corda = (
        te - le
    )

    chord = np.linalg.norm(
        vetor_corda
    )

    if chord <= 1e-12:

        raise ValueError(
            "Corda inválida."
        )

    ex = (
        vetor_corda /
        chord
    )

    ey = np.array([
        -ex[1],
        ex[0]
    ])

    def transformar(
        superficie
    ):

        relativo = (
            superficie - le
        )

        x_norm = (
            relativo @ ex
        ) / chord

        y_norm = (
            relativo @ ey
        ) / chord

        return np.column_stack([
            x_norm,
            y_norm
        ])

    upper_norm = transformar(
        upper
    )

    lower_norm = transformar(
        lower
    )

    return (
        upper_norm,
        lower_norm,
        chord
    )


# ============================================================
# 10. PREPARAÇÃO NUMÉRICA
# ============================================================

def preparar_superficie(
    superficie
):
    """
    Remove duplicatas e ordena por x.
    """

    superficie = ordenar_superficie_por_x(
        superficie
    )

    x = np.clip(
        superficie[:, 0],
        0.0,
        1.0
    )

    y = superficie[:, 1]

    x_round = np.round(
        x,
        10
    )

    valores = np.unique(
        x_round
    )

    novo_x = []
    novo_y = []

    for valor in valores:

        mascara = (
            x_round == valor
        )

        novo_x.append(
            np.mean(
                x[mascara]
            )
        )

        novo_y.append(
            np.mean(
                y[mascara]
            )
        )

    return (
        np.asarray(novo_x),
        np.asarray(novo_y)
    )


# ============================================================
# 11. AJUSTE CST
# ============================================================

def ajustar_cst(
    x,
    y,
    ordem
):
    """
    Ajuste CST por mínimos quadrados.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    if len(x) < ordem + 2:

        raise ValueError(
            "Quantidade insuficiente de pontos "
            "para a ordem CST."
        )

    # --------------------------------------------------------
    # Trailing edge
    # --------------------------------------------------------

    idx_te = np.argmax(
        x
    )

    delta_te = float(
        y[idx_te]
    )

    y_shape = (
        y -
        x * delta_te
    )

    B = matriz_cst(
        x,
        ordem
    )

    coeffs, residuos, rank, singular = (
        np.linalg.lstsq(
            B,
            y_shape,
            rcond=None
        )
    )

    y_fit = avaliar_cst(
        x,
        coeffs,
        delta_te
    )

    erro = (
        y -
        y_fit
    )

    rmse = float(
        np.sqrt(
            np.mean(
                erro ** 2
            )
        )
    )

    max_error = float(
        np.max(
            np.abs(
                erro
            )
        )
    )

    return {
        "coeffs": coeffs,
        "delta_te": delta_te,
        "rmse": rmse,
        "max_error": max_error,
        "rank": rank
    }


# ============================================================
# 12. PROCESSAMENTO DE UM PERFIL
# ============================================================

def processar_aerofolio(
    arquivo_dat,
    ordem=6
):
    """
    Pipeline completo.
    """

    arquivo_dat = Path(
        arquivo_dat
    )

    try:

        linhas = ler_dat_bruto(
            arquivo_dat
        )

        formato = detectar_formato(
            linhas
        )

        if formato == "selig":

            upper, lower = parse_selig(
                linhas
            )

        elif formato == "lednicer":

            upper, lower = parse_lednicer(
                linhas
            )

        else:

            raise ValueError(
                f"Formato não reconhecido: {formato}"
            )

        # ----------------------------------------------------
        # Normalização
        # ----------------------------------------------------

        (
            upper_norm,
            lower_norm,
            chord_original
        ) = normalizar_superficies(
            upper,
            lower
        )

        # Garante upper realmente acima do lower
        if (
            np.mean(upper_norm[:, 1])
            <
            np.mean(lower_norm[:, 1])
        ):

            upper_norm, lower_norm = (
                lower_norm,
                upper_norm
            )

        # ----------------------------------------------------
        # Preparação
        # ----------------------------------------------------

        xu, yu = preparar_superficie(
            upper_norm
        )

        xl, yl = preparar_superficie(
            lower_norm
        )

        # ----------------------------------------------------
        # CST
        # ----------------------------------------------------

        resultado_u = ajustar_cst(
            xu,
            yu,
            ordem
        )

        resultado_l = ajustar_cst(
            xl,
            yl,
            ordem
        )

        return {

            "formato": formato,

            "upper": resultado_u,

            "lower": resultado_l,

            "chord_original":
                chord_original,

            "x_upper": xu,

            "y_upper": yu,

            "x_lower": xl,

            "y_lower": yl

        }

    except Exception as e:

        print(
            f"❌ Falha em {arquivo_dat.name}: {e}"
        )

        return None


# ============================================================
# 13. COSINE SPACING
# ============================================================

def gerar_cosine_spacing(
    pontos
):

    beta = np.linspace(
        0,
        np.pi,
        pontos
    )

    return (
        1 -
        np.cos(beta)
    ) / 2


# ============================================================
# 14. DAT CST RECONSTRUÍDO
# ============================================================

def salvar_reconstrucao(
    airfoil_id,
    resultado,
    pontos,
    out_dir
):
    """
    Arquivo usado apenas para inspeção/validação.

    O XFOIL oficial do seu banco continuará
    rodando nos .dat originais.
    """

    out_dir = Path(
        out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    x = gerar_cosine_spacing(
        pontos
    )

    yu = avaliar_cst(
        x,
        resultado["upper"]["coeffs"],
        resultado["upper"]["delta_te"]
    )

    yl = avaliar_cst(
        x,
        resultado["lower"]["coeffs"],
        resultado["lower"]["delta_te"]
    )

    # TE upper -> LE -> TE lower
    x_final = np.concatenate([
        x[::-1],
        x[1:]
    ])

    y_final = np.concatenate([
        yu[::-1],
        yl[1:]
    ])

    caminho = (
        out_dir /
        f"{airfoil_id}_cst.dat"
    )

    with open(
        caminho,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            f"{airfoil_id}_CST\n"
        )

        for xi, yi in zip(
            x_final,
            y_final
        ):

            arquivo.write(
                f"{xi: .8f} {yi: .8f}\n"
            )

    return caminho


# ============================================================
# 15. AIRFOIL ID
# ============================================================

def criar_airfoil_id(
    nome
):

    return re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        nome.strip()
    )


# ============================================================
# 16. ANÁLISE DE QUALIDADE
# ============================================================

def analisar_qualidade(
    airfoil_id,
    resultado
):

    avisos = []

    upper = resultado["upper"]
    lower = resultado["lower"]

    if upper["rmse"] > RMSE_WARNING:

        avisos.append(
            f"RMSE upper alto: "
            f"{upper['rmse']:.6f}"
        )

    if lower["rmse"] > RMSE_WARNING:

        avisos.append(
            f"RMSE lower alto: "
            f"{lower['rmse']:.6f}"
        )

    if (
        upper["max_error"]
        >
        MAX_ERROR_WARNING
    ):

        avisos.append(
            f"MaxError upper alto: "
            f"{upper['max_error']:.6f}"
        )

    if (
        lower["max_error"]
        >
        MAX_ERROR_WARNING
    ):

        avisos.append(
            f"MaxError lower alto: "
            f"{lower['max_error']:.6f}"
        )

    max_coeff = max(
        np.max(
            np.abs(
                upper["coeffs"]
            )
        ),
        np.max(
            np.abs(
                lower["coeffs"]
            )
        )
    )

    if max_coeff > COEFF_WARNING:

        avisos.append(
            f"Coeficiente CST elevado: "
            f"{max_coeff:.3f}"
        )

    chord = resultado[
        "chord_original"
    ]

    if (
        chord < CHORD_MIN_WARNING
        or
        chord > CHORD_MAX_WARNING
    ):

        avisos.append(
            f"Corda original incomum: "
            f"{chord:.6f}"
        )

    return avisos


# ============================================================
# 17. CRIAR LINHA DO CSV
# ============================================================

def criar_linha_csv(
    airfoil_id,
    arquivo,
    resultado,
    arquivo_cst
):

    row = {

        "airfoil_id":
            airfoil_id,

        "arquivo_original":
            arquivo.name,

        "formato_dat":
            resultado["formato"],

        "arquivo_cst":
            arquivo_cst.name,

        "ordem_cst":
            len(
                resultado["upper"]["coeffs"]
            ) - 1,

        "chord_original":
            resultado["chord_original"],

        "DeltaTE_upper":
            resultado["upper"]["delta_te"],

        "DeltaTE_lower":
            resultado["lower"]["delta_te"],

        "RMSE_upper":
            resultado["upper"]["rmse"],

        "RMSE_lower":
            resultado["lower"]["rmse"],

        "MaxError_upper":
            resultado["upper"]["max_error"],

        "MaxError_lower":
            resultado["lower"]["max_error"]

    }

    for i, val in enumerate(
        resultado["upper"]["coeffs"]
    ):

        row[f"Au{i}"] = float(
            val
        )

    for i, val in enumerate(
        resultado["lower"]["coeffs"]
    ):

        row[f"Al{i}"] = float(
            val
        )

    return row


# ============================================================
# 18. PROCESSAMENTO DO DIRETÓRIO
# ============================================================

def processar_diretorio_airfoils(
    diretorio_airfoils,
    ordem=6,
    pontos_reconstrucao=120,
    csv_nome="database_cst.csv"
):

    inicio = time.time()

    diretorio_airfoils = Path(
        diretorio_airfoils
    )

    if not diretorio_airfoils.exists():

        raise FileNotFoundError(
            f"Diretório inexistente: "
            f"{diretorio_airfoils}"
        )

    dat_files = sorted(
        diretorio_airfoils.glob(
            "*.dat"
        )
    )

    print("\n" + "=" * 70)

    print(
        "PARAMETRIZAÇÃO CST - SELIG + LEDNICER"
    )

    print("=" * 70)

    print(
        f"\nPerfis encontrados: "
        f"{len(dat_files)}"
    )

    # --------------------------------------------------------
    # Saídas
    # --------------------------------------------------------

    output_dir = Path(
        r"C:\Repositorios\TCC\cst\cst_output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cst_out = (
        output_dir /
        "cst_reconstruidos"
    )

    cst_out.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        output_dir /
        csv_nome
    )

    linhas_csv = []

    falhas = []

    alertas = []

    formatos = {
        "selig": 0,
        "lednicer": 0
    }

    # ========================================================
    # LOOP
    # ========================================================

    for indice, arquivo in enumerate(
        dat_files,
        start=1
    ):

        airfoil_id = criar_airfoil_id(
            arquivo.stem
        )

        print(
            "\n" + "-" * 70
        )

        print(
            f"[{indice}/{len(dat_files)}] "
            f"{arquivo.name}"
        )

        resultado = processar_aerofolio(
            arquivo,
            ordem
        )

        if resultado is None:

            falhas.append(
                airfoil_id
            )

            continue

        formato = resultado[
            "formato"
        ]

        formatos[formato] = (
            formatos.get(formato, 0)
            + 1
        )

        print(
            f"   Formato     : "
            f"{formato.upper()}"
        )

        print(
            f"   RMSE upper  : "
            f"{resultado['upper']['rmse']:.8f}"
        )

        print(
            f"   RMSE lower  : "
            f"{resultado['lower']['rmse']:.8f}"
        )

        print(
            f"   MAX upper   : "
            f"{resultado['upper']['max_error']:.8f}"
        )

        print(
            f"   MAX lower   : "
            f"{resultado['lower']['max_error']:.8f}"
        )

        avisos = analisar_qualidade(
            airfoil_id,
            resultado
        )

        if avisos:

            alertas.append(
                airfoil_id
            )

            print(
                "   ⚠️ Revisão recomendada:"
            )

            for aviso in avisos:

                print(
                    f"      - {aviso}"
                )

        else:

            print(
                "   ✅ CST adequado"
            )

        arquivo_cst = salvar_reconstrucao(

            airfoil_id,

            resultado,

            pontos_reconstrucao,

            cst_out

        )

        row = criar_linha_csv(

            airfoil_id,

            arquivo,

            resultado,

            arquivo_cst

        )

        linhas_csv.append(
            row
        )

    # ========================================================
    # CSV
    # ========================================================

    if linhas_csv:

        df = pd.DataFrame(
            linhas_csv
        )

        au_cols = sorted(
            [
                c
                for c in df.columns
                if re.fullmatch(
                    r"Au\d+",
                    c
                )
            ],
            key=lambda c:
                int(c[2:])
        )

        al_cols = sorted(
            [
                c
                for c in df.columns
                if re.fullmatch(
                    r"Al\d+",
                    c
                )
            ],
            key=lambda c:
                int(c[2:])
        )

        colunas = [

            "airfoil_id",

            "arquivo_original",

            "formato_dat",

            "arquivo_cst",

            "ordem_cst",

            "chord_original",

        ] + au_cols + al_cols + [

            "DeltaTE_upper",

            "DeltaTE_lower",

            "RMSE_upper",

            "RMSE_lower",

            "MaxError_upper",

            "MaxError_lower"

        ]

        df = df[
            colunas
        ]

        df.to_csv(
            csv_path,
            index=False,
            float_format="%.10f"
        )

    # ========================================================
    # RESUMO
    # ========================================================

    duracao = (
        time.time()
        -
        inicio
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RESUMO FINAL"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal de arquivos: "
        f"{len(dat_files)}"
    )

    print(
        f"Processados com sucesso: "
        f"{len(linhas_csv)}"
    )

    print(
        f"Falhas: "
        f"{len(falhas)}"
    )

    print(
        f"Avisos de qualidade: "
        f"{len(alertas)}"
    )

    print(
        f"\nSELIG detectados: "
        f"{formatos.get('selig', 0)}"
    )

    print(
        f"LEDNICER detectados: "
        f"{formatos.get('lednicer', 0)}"
    )

    if falhas:

        print(
            "\n❌ Perfis que ainda falharam:"
        )

        for nome in falhas:

            print(
                f"   - {nome}"
            )

    if alertas:

        print(
            "\n⚠️ Perfis que merecem inspeção:"
        )

        for nome in alertas:

            print(
                f"   - {nome}"
            )

    print(
        f"\nCSV:\n{csv_path}"
    )

    print(
        f"\nReconstruções CST:\n"
        f"{cst_out}"
    )

    print(
        f"\nTempo total: "
        f"{duracao:.2f}s"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    processar_diretorio_airfoils(

        r"C:\Repositorios\TCC\Airfoils",

        ordem=ORDEM_CST,

        pontos_reconstrucao=
            PONTOS_RECONSTRUCAO,

        csv_nome=
            "database_cst.csv"

    )
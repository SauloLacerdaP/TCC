from pathlib import Path
import numpy as np
import shutil
import csv
import re


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_ORIGINAL = Path(
    r"C:\Repositorios\TCC\Airfoils"
)

PASTA_SAIDA = Path(
    r"C:\Repositorios\TCC\lednicer_to_selig\Airfoils_Selig"
)

RELATORIO_CSV = (
    PASTA_SAIDA /
    "relatorio_conversao.csv"
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def limpar_nome(nome):

    return re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        nome
    )


def tentar_float(valor):

    try:
        return float(valor)

    except (ValueError, TypeError):
        return None


def parece_contagem_pontos(x, y):
    """
    Identifica linhas típicas Lednicer:

        61 61
        35 35
        33 34
    """

    if not np.isfinite(x) or not np.isfinite(y):
        return False

    x_int = abs(x - round(x)) < 1e-8
    y_int = abs(y - round(y)) < 1e-8

    return (
        x_int
        and y_int
        and x >= 5
        and y >= 5
    )


# ============================================================
# LEITURA DO DAT
# ============================================================

def ler_dat(arquivo_dat):

    arquivo_dat = Path(
        arquivo_dat
    )

    itens = []

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

            limpa = (
                linha
                .replace(",", " ")
                .replace("\t", " ")
                .strip()
            )

            # Linha vazia
            if limpa == "":

                itens.append({
                    "tipo": "vazia",
                    "linha": numero_linha
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

                itens.append({
                    "tipo": "contagem",
                    "linha": numero_linha,
                    "n_upper": int(round(x)),
                    "n_lower": int(round(y))
                })

            else:

                itens.append({
                    "tipo": "ponto",
                    "linha": numero_linha,
                    "x": float(x),
                    "y": float(y)
                })

    return itens


# ============================================================
# DETECÇÃO DO FORMATO
# ============================================================

def detectar_formato(itens):

    # --------------------------------------------------------
    # Se houver linha explícita de contagem:
    # praticamente certamente Lednicer
    # --------------------------------------------------------

    if any(
        item["tipo"] == "contagem"
        for item in itens
    ):

        return "lednicer"

    pontos = [
        item
        for item in itens
        if item["tipo"] == "ponto"
    ]

    if len(pontos) < 10:

        raise ValueError(
            "Poucos pontos válidos."
        )

    coords = np.asarray([
        [p["x"], p["y"]]
        for p in pontos
    ])

    x = coords[:, 0]

    idx_le = int(
        np.argmin(x)
    )

    # --------------------------------------------------------
    # Selig clássico:
    #
    # TE -> LE -> TE
    #
    # portanto o menor x deve aparecer
    # aproximadamente no meio da sequência.
    # --------------------------------------------------------

    if (
        idx_le >= 2
        and
        idx_le <= len(coords) - 3
    ):

        return "selig"

    # --------------------------------------------------------
    # Caso contrário:
    # provavelmente duas superfícies separadas
    # --------------------------------------------------------

    return "lednicer"


# ============================================================
# REMOÇÃO DE PONTOS DUPLICADOS
# ============================================================

def remover_duplicatas_sequenciais(pontos):

    pontos = np.asarray(
        pontos,
        dtype=float
    )

    if len(pontos) <= 1:
        return pontos

    resultado = [
        pontos[0]
    ]

    for ponto in pontos[1:]:

        if not np.allclose(
            ponto,
            resultado[-1],
            atol=1e-12
        ):

            resultado.append(
                ponto
            )

    return np.asarray(
        resultado,
        dtype=float
    )


# ============================================================
# CLASSIFICAR UPPER / LOWER
# ============================================================

def classificar_superficies(
    superficie_1,
    superficie_2
):

    superficie_1 = np.asarray(
        superficie_1,
        dtype=float
    )

    superficie_2 = np.asarray(
        superficie_2,
        dtype=float
    )

    # --------------------------------------------------------
    # Compara y médio.
    #
    # Para aerofólios convencionais:
    # upper possui y médio maior.
    # --------------------------------------------------------

    media_1 = np.mean(
        superficie_1[:, 1]
    )

    media_2 = np.mean(
        superficie_2[:, 1]
    )

    if media_1 >= media_2:

        upper = superficie_1
        lower = superficie_2

    else:

        upper = superficie_2
        lower = superficie_1

    return upper, lower


# ============================================================
# ORDENAR SUPERFÍCIES PARA SELIG
# ============================================================

def ordenar_para_selig(
    upper,
    lower
):

    upper = np.asarray(
        upper,
        dtype=float
    )

    lower = np.asarray(
        lower,
        dtype=float
    )

    # --------------------------------------------------------
    # Upper precisa:
    #
    # TE -> LE
    # x decrescente
    # --------------------------------------------------------

    upper = upper[
        np.argsort(
            upper[:, 0]
        )[::-1]
    ]

    # --------------------------------------------------------
    # Lower precisa:
    #
    # LE -> TE
    # x crescente
    # --------------------------------------------------------

    lower = lower[
        np.argsort(
            lower[:, 0]
        )
    ]

    upper = remover_duplicatas_sequenciais(
        upper
    )

    lower = remover_duplicatas_sequenciais(
        lower
    )

    return upper, lower


# ============================================================
# PARSER LEDNICER
# ============================================================

def extrair_lednicer(itens):

    # ========================================================
    # MÉTODO 1:
    # Linha explícita com quantidade de pontos
    # ========================================================

    for i, item in enumerate(itens):

        if item["tipo"] == "contagem":

            n1 = item["n_upper"]
            n2 = item["n_lower"]

            pontos_depois = [
                p
                for p in itens[i + 1:]
                if p["tipo"] == "ponto"
            ]

            if len(pontos_depois) >= n1 + n2:

                superficie_1 = np.asarray([
                    [p["x"], p["y"]]
                    for p
                    in pontos_depois[:n1]
                ])

                superficie_2 = np.asarray([
                    [p["x"], p["y"]]
                    for p
                    in pontos_depois[n1:n1 + n2]
                ])

                return classificar_superficies(
                    superficie_1,
                    superficie_2
                )

    # ========================================================
    # MÉTODO 2:
    # Blocos separados por linha vazia
    # ========================================================

    blocos = []

    bloco_atual = []

    for item in itens:

        if item["tipo"] == "ponto":

            bloco_atual.append(
                [
                    item["x"],
                    item["y"]
                ]
            )

        elif item["tipo"] == "vazia":

            if len(bloco_atual) >= 4:

                blocos.append(
                    np.asarray(
                        bloco_atual,
                        dtype=float
                    )
                )

            bloco_atual = []

    if len(bloco_atual) >= 4:

        blocos.append(
            np.asarray(
                bloco_atual,
                dtype=float
            )
        )

    # Remove blocos muito pequenos
    blocos = [
        bloco
        for bloco in blocos
        if len(bloco) >= 4
    ]

    if len(blocos) >= 2:

        # Usa os dois maiores
        blocos = sorted(
            blocos,
            key=len,
            reverse=True
        )

        return classificar_superficies(
            blocos[0],
            blocos[1]
        )

    # ========================================================
    # MÉTODO 3:
    # Detecta reinício de x
    # ========================================================

    coords = np.asarray([
        [item["x"], item["y"]]
        for item in itens
        if item["tipo"] == "ponto"
    ])

    if len(coords) < 10:

        raise ValueError(
            "Poucos pontos para separar Lednicer."
        )

    # --------------------------------------------------------
    # Muitos Lednicer têm:
    #
    # LE -> TE
    #
    # seguido novamente de:
    #
    # LE -> TE
    #
    # Então procuramos queda brusca de x.
    # --------------------------------------------------------

    x = coords[:, 0]

    dx = np.diff(
        x
    )

    candidatos = np.where(
        dx < -0.3 * np.ptp(x)
    )[0]

    if len(candidatos) > 0:

        idx = candidatos[0] + 1

        superficie_1 = coords[:idx]
        superficie_2 = coords[idx:]

        if (
            len(superficie_1) >= 4
            and
            len(superficie_2) >= 4
        ):

            return classificar_superficies(
                superficie_1,
                superficie_2
            )

    raise ValueError(
        "Não foi possível separar as superfícies Lednicer."
    )


# ============================================================
# VALIDAR GEOMETRIA
# ============================================================

def validar_superficies(
    upper,
    lower
):

    erros = []

    if len(upper) < 5:

        erros.append(
            "Poucos pontos no upper."
        )

    if len(lower) < 5:

        erros.append(
            "Poucos pontos no lower."
        )

    todos = np.vstack([
        upper,
        lower
    ])

    xmin = np.min(
        todos[:, 0]
    )

    xmax = np.max(
        todos[:, 0]
    )

    chord = (
        xmax - xmin
    )

    if chord <= 0:

        erros.append(
            "Corda inválida."
        )

    if not np.all(
        np.isfinite(todos)
    ):

        erros.append(
            "Coordenadas não finitas."
        )

    return erros


# ============================================================
# SALVAR SELIG
# ============================================================

def salvar_selig(
    nome,
    upper,
    lower,
    destino
):

    destino = Path(
        destino
    )

    upper, lower = ordenar_para_selig(
        upper,
        lower
    )

    # --------------------------------------------------------
    # Evita LE duplicado.
    #
    # Se último upper ≈ primeiro lower,
    # remove o primeiro lower.
    # --------------------------------------------------------

    if (
        len(upper) > 0
        and
        len(lower) > 0
        and
        np.allclose(
            upper[-1],
            lower[0],
            atol=1e-10
        )
    ):

        lower = lower[1:]

    with open(
        destino,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            f"{nome}\n"
        )

        # Upper: TE -> LE
        for x, y in upper:

            arquivo.write(
                f"{x: .8f} {y: .8f}\n"
            )

        # Lower: LE -> TE
        for x, y in lower:

            arquivo.write(
                f"{x: .8f} {y: .8f}\n"
            )


# ============================================================
# PROCESSAR UM ARQUIVO
# ============================================================

def processar_arquivo(
    arquivo_original,
    pasta_saida
):

    arquivo_original = Path(
        arquivo_original
    )

    airfoil_id = limpar_nome(
        arquivo_original.stem
    )

    arquivo_saida = (
        pasta_saida /
        f"{airfoil_id}.dat"
    )

    try:

        itens = ler_dat(
            arquivo_original
        )

        formato = detectar_formato(
            itens
        )

        # ====================================================
        # JÁ É SELIG
        # ====================================================

        if formato == "selig":

            shutil.copy2(
                arquivo_original,
                arquivo_saida
            )

            return {
                "perfil":
                    airfoil_id,

                "arquivo":
                    arquivo_original.name,

                "formato_original":
                    "selig",

                "acao":
                    "copiado",

                "status":
                    "OK",

                "erro":
                    ""
            }

        # ====================================================
        # LEDNICER
        # ====================================================

        upper, lower = extrair_lednicer(
            itens
        )

        erros = validar_superficies(
            upper,
            lower
        )

        if erros:

            raise ValueError(
                " | ".join(erros)
            )

        salvar_selig(
            airfoil_id,
            upper,
            lower,
            arquivo_saida
        )

        return {
            "perfil":
                airfoil_id,

            "arquivo":
                arquivo_original.name,

            "formato_original":
                "lednicer",

            "acao":
                "convertido_para_selig",

            "status":
                "OK",

            "erro":
                ""
        }

    except Exception as e:

        return {
            "perfil":
                airfoil_id,

            "arquivo":
                arquivo_original.name,

            "formato_original":
                "desconhecido",

            "acao":
                "nenhuma",

            "status":
                "FALHA",

            "erro":
                str(e)
        }


# ============================================================
# PROCESSAR TODA A PASTA
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CONVERSÃO LEDNICER → SELIG")
    print("=" * 70)

    if not PASTA_ORIGINAL.exists():

        print(
            "\nERRO: pasta original não encontrada:"
        )

        print(
            PASTA_ORIGINAL
        )

        return

    # --------------------------------------------------------
    # Recria pasta de saída
    # --------------------------------------------------------

    if PASTA_SAIDA.exists():

        shutil.rmtree(
            PASTA_SAIDA
        )

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    arquivos = sorted(
        PASTA_ORIGINAL.glob(
            "*.dat"
        )
    )

    print(
        f"\nArquivos encontrados: "
        f"{len(arquivos)}"
    )

    relatorio = []

    total_selig = 0
    total_lednicer = 0
    total_falhas = 0

    # ========================================================
    # LOOP
    # ========================================================

    for indice, arquivo in enumerate(
        arquivos,
        start=1
    ):

        resultado = processar_arquivo(
            arquivo,
            PASTA_SAIDA
        )

        relatorio.append(
            resultado
        )

        if resultado["status"] == "FALHA":

            total_falhas += 1

            print(
                f"[{indice}/{len(arquivos)}] "
                f"❌ {arquivo.name}"
            )

            print(
                f"    {resultado['erro']}"
            )

        elif (
            resultado["formato_original"]
            == "lednicer"
        ):

            total_lednicer += 1

            print(
                f"[{indice}/{len(arquivos)}] "
                f"🔄 {arquivo.name} "
                f"LEDNICER → SELIG"
            )

        else:

            total_selig += 1

            print(
                f"[{indice}/{len(arquivos)}] "
                f"✅ {arquivo.name} "
                f"já era SELIG"
            )

    # ========================================================
    # RELATÓRIO CSV
    # ========================================================

    campos = [
        "perfil",
        "arquivo",
        "formato_original",
        "acao",
        "status",
        "erro"
    ]

    with open(
        RELATORIO_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        writer = csv.DictWriter(
            arquivo,
            fieldnames=campos
        )

        writer.writeheader()

        writer.writerows(
            relatorio
        )

    # ========================================================
    # RESUMO
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CONVERSÃO FINALIZADA")
    print("=" * 70)

    print(
        f"\nTotal: "
        f"{len(arquivos)}"
    )

    print(
        f"Selig já existentes: "
        f"{total_selig}"
    )

    print(
        f"Lednicer convertidos: "
        f"{total_lednicer}"
    )

    print(
        f"Falhas: "
        f"{total_falhas}"
    )

    print(
        "\nPasta pronta para XFOIL:"
    )

    print(
        PASTA_SAIDA
    )

    print(
        "\nRelatório:"
    )

    print(
        RELATORIO_CSV
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()
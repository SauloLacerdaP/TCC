from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_XFOIL = Path(
    r"C:\Ciencia de Dados\TCC\Output_dados\banco_dados_xfoil.csv"
)

# Pasta contendo todos os .dat
# Usada somente para encontrar perfis com ZERO resultados
PASTA_PERFIS = Path(
    r"C:\Ciencia de Dados\TCC\Airfoils_Selig"
)

PASTA_SAIDA = Path(
    r"C:\Ciencia de Dados\TCC\Output_dados"
)


# ============================================================
# DOMÍNIO QUE SERÁ CONSIDERADO
# ============================================================

REYNOLDS_ESPERADOS = [
    200000,
    250000,
    300000
]

ALPHA_INICIAL = 0
ALPHA_FINAL = 12
PASSO_ALPHA = 1


# ============================================================
# CRITÉRIOS DE CLASSIFICAÇÃO
# ============================================================

LIMITE_OK = 0.80
LIMITE_CRITICO = 0.50


# ============================================================
# FUNÇÕES
# ============================================================

def gerar_alphas_esperados():

    return list(
        range(
            ALPHA_INICIAL,
            ALPHA_FINAL + 1,
            PASSO_ALPHA
        )
    )


def classificar(percentual):

    if percentual >= LIMITE_OK:
        return "OK"

    elif percentual >= LIMITE_CRITICO:
        return "REVISAR"

    else:
        return "CRITICO"


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("RELATÓRIO DE CONVERGÊNCIA XFOIL")
    print("DOMÍNIO: ALPHA 0° A 12°")
    print("=" * 70)

    if not ARQUIVO_XFOIL.exists():

        raise FileNotFoundError(
            f"CSV não encontrado:\n{ARQUIVO_XFOIL}"
        )

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CARREGAR CSV
    # ========================================================

    df = pd.read_csv(
        ARQUIVO_XFOIL
    )

    df["perfil"] = (
        df["perfil"]
        .astype(str)
        .str.strip()
    )

    print(
        f"\nLinhas originais no CSV: "
        f"{len(df)}"
    )

    # ========================================================
    # FILTRAR ALPHA 0–12
    # ========================================================

    df = df[
        (df["alpha"] >= ALPHA_INICIAL)
        &
        (df["alpha"] <= ALPHA_FINAL)
    ].copy()

    print(
        f"Linhas consideradas entre "
        f"{ALPHA_INICIAL}° e {ALPHA_FINAL}°: "
        f"{len(df)}"
    )

    # ========================================================
    # CONFIGURAÇÃO ESPERADA
    # ========================================================

    alphas_esperados = gerar_alphas_esperados()

    pontos_por_reynolds = len(
        alphas_esperados
    )

    pontos_por_perfil = (
        pontos_por_reynolds
        *
        len(REYNOLDS_ESPERADOS)
    )

    print(
        f"\nPontos esperados por Reynolds: "
        f"{pontos_por_reynolds}"
    )

    print(
        f"Pontos esperados por perfil: "
        f"{pontos_por_perfil}"
    )

    # ========================================================
    # PERFIS PRESENTES
    # ========================================================

    perfis_csv = sorted(
        df["perfil"].unique()
    )

    # ========================================================
    # RELATÓRIO DETALHADO
    # PERFIL × REYNOLDS
    # ========================================================

    linhas_detalhadas = []

    for perfil in perfis_csv:

        df_perfil = df[
            df["perfil"] == perfil
        ]

        for reynolds in REYNOLDS_ESPERADOS:

            subset = df_perfil[
                df_perfil["Re"] == reynolds
            ]

            alphas_encontrados = sorted(
                set(
                    int(round(alpha))
                    for alpha in subset["alpha"]
                )
            )

            alphas_faltantes = [

                alpha

                for alpha in alphas_esperados

                if alpha
                not in alphas_encontrados

            ]

            convergidos = len(
                alphas_encontrados
            )

            percentual = (
                convergidos
                /
                pontos_por_reynolds
            )

            linhas_detalhadas.append({

                "perfil":
                    perfil,

                "Re":
                    reynolds,

                "pontos_esperados":
                    pontos_por_reynolds,

                "pontos_convergidos":
                    convergidos,

                "pontos_faltantes":
                    pontos_por_reynolds
                    -
                    convergidos,

                "percentual_convergencia":
                    percentual * 100,

                "classificacao":
                    classificar(
                        percentual
                    ),

                "alphas_convergidos":
                    ", ".join(
                        str(a)
                        for a
                        in alphas_encontrados
                    ),

                "alphas_faltantes":
                    ", ".join(
                        str(a)
                        for a
                        in alphas_faltantes
                    )

            })

    df_detalhado = pd.DataFrame(
        linhas_detalhadas
    )

    # ========================================================
    # RELATÓRIO POR PERFIL
    # ========================================================

    linhas_resumo = []

    for perfil in perfis_csv:

        df_perfil = df[
            df["perfil"] == perfil
        ]

        pontos_convergidos = len(

            df_perfil.drop_duplicates(
                subset=[
                    "Re",
                    "alpha"
                ]
            )

        )

        percentual = (
            pontos_convergidos
            /
            pontos_por_perfil
        )

        linha = {

            "perfil":
                perfil,

            "pontos_esperados":
                pontos_por_perfil,

            "pontos_convergidos":
                pontos_convergidos,

            "pontos_faltantes":
                pontos_por_perfil
                -
                pontos_convergidos,

            "percentual_convergencia":
                percentual * 100,

            "classificacao":
                classificar(
                    percentual
                )

        }

        # ----------------------------------------------------
        # INFORMAÇÕES POR REYNOLDS
        # ----------------------------------------------------

        for reynolds in REYNOLDS_ESPERADOS:

            subset = df_perfil[
                df_perfil["Re"]
                ==
                reynolds
            ]

            qtd = len(

                subset.drop_duplicates(
                    subset=["alpha"]
                )

            )

            linha[
                f"pontos_Re_{reynolds}"
            ] = qtd

            linha[
                f"percentual_Re_{reynolds}"
            ] = (
                qtd
                /
                pontos_por_reynolds
                *
                100
            )

        # ----------------------------------------------------
        # CONDIÇÕES FALTANTES
        # ----------------------------------------------------

        faltantes_total = []

        for reynolds in REYNOLDS_ESPERADOS:

            subset = df_perfil[
                df_perfil["Re"]
                ==
                reynolds
            ]

            encontrados = set(

                int(round(a))

                for a in subset["alpha"]

            )

            faltantes = [

                a

                for a
                in alphas_esperados

                if a not in encontrados

            ]

            if faltantes:

                faltantes_total.append(

                    f"Re={reynolds}: "
                    +
                    ",".join(
                        str(a)
                        for a in faltantes
                    )

                )

        linha[
            "condicoes_faltantes"
        ] = " | ".join(
            faltantes_total
        )

        linhas_resumo.append(
            linha
        )

    df_resumo = pd.DataFrame(
        linhas_resumo
    )

    # ========================================================
    # PERFIS COM ZERO RESULTADOS
    # ========================================================

    perfis_zero = []

    if PASTA_PERFIS.exists():

        perfis_esperados = sorted(

            arquivo.stem

            for arquivo
            in PASTA_PERFIS.glob(
                "*.dat"
            )

        )

        conjunto_csv = set(
            perfis_csv
        )

        perfis_zero = [

            perfil

            for perfil
            in perfis_esperados

            if perfil
            not in conjunto_csv

        ]

    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    arquivo_resumo = (
        PASTA_SAIDA
        /
        "relatorio_convergencia_0a12_perfis.csv"
    )

    arquivo_detalhado = (
        PASTA_SAIDA
        /
        "relatorio_convergencia_0a12_detalhado.csv"
    )

    df_resumo.to_csv(
        arquivo_resumo,
        index=False,
        float_format="%.2f"
    )

    df_detalhado.to_csv(
        arquivo_detalhado,
        index=False,
        float_format="%.2f"
    )

    # ========================================================
    # ESTATÍSTICAS
    # ========================================================

    qtd_ok = int(
        (
            df_resumo["classificacao"]
            ==
            "OK"
        ).sum()
    )

    qtd_revisar = int(
        (
            df_resumo["classificacao"]
            ==
            "REVISAR"
        ).sum()
    )

    qtd_critico = int(
        (
            df_resumo["classificacao"]
            ==
            "CRITICO"
        ).sum()
    )

    total_perfis_esperados = (
        len(perfis_csv)
        +
        len(perfis_zero)
    )

    pontos_teoricos = (
        total_perfis_esperados
        *
        pontos_por_perfil
    )

    pontos_reais = len(

        df.drop_duplicates(
            subset=[
                "perfil",
                "Re",
                "alpha"
            ]
        )

    )

    taxa_global = (
        pontos_reais
        /
        pontos_teoricos
        *
        100
    )

    # ========================================================
    # RESUMO TERMINAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RESULTADO — ALPHA 0° A 12°")
    print("=" * 70)

    print(
        f"\nPerfis com resultados: "
        f"{len(perfis_csv)}"
    )

    print(
        f"Perfis com zero resultados: "
        f"{len(perfis_zero)}"
    )

    if perfis_zero:

        print(
            "\nPerfis com 0 pontos:"
        )

        for perfil in perfis_zero:

            print(
                f"   - {perfil}"
            )

    print(
        "\nClassificação:"
    )

    print(
        f"   OK      : {qtd_ok}"
    )

    print(
        f"   REVISAR : {qtd_revisar}"
    )

    print(
        f"   CRITICO : {qtd_critico}"
    )

    print(
        f"\nPontos teóricos: "
        f"{pontos_teoricos}"
    )

    print(
        f"Pontos convergidos: "
        f"{pontos_reais}"
    )

    print(
        f"\nTaxa global de convergência: "
        f"{taxa_global:.2f}%"
    )

    print(
        f"\nRelatório por perfil:\n"
        f"{arquivo_resumo}"
    )

    print(
        f"\nRelatório detalhado:\n"
        f"{arquivo_detalhado}"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()
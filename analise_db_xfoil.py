import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO = Path(
    r"C:\Repositorios\TCC\Output_dados\database_ml.csv"
)

PASTA_SAIDA = Path(
    r"C:\Repositorios\TCC\Output_dados\diagnostico_convergencia"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

# Intervalo esperado após o ETL
ALPHAS_ESPERADOS = list(range(0, 13))


# ============================================================
# 1. CARREGAMENTO
# ============================================================

print("=" * 70)
print("ANÁLISE DE QUALIDADE E CONVERGÊNCIA DO BANCO")
print("=" * 70)

df = pd.read_csv(ARQUIVO)

print(f"\nArquivo: {ARQUIVO}")
print(f"Linhas: {len(df)}")
print(f"Colunas: {len(df.columns)}")


# ============================================================
# 2. INFORMAÇÕES BÁSICAS
# ============================================================

perfis = sorted(df["perfil"].unique())
reynolds = sorted(df["Re"].unique())
machs = sorted(df["Mach"].unique())
alphas_encontrados = sorted(df["alpha"].unique())

n_perfis = len(perfis)
n_re = len(reynolds)
n_mach = len(machs)
n_alpha = len(ALPHAS_ESPERADOS)

print("\n" + "=" * 70)
print("CONFIGURAÇÃO DO BANCO")
print("=" * 70)

print(f"Perfis encontrados: {n_perfis}")
print(f"Reynolds encontrados: {reynolds}")
print(f"Mach encontrados: {machs}")
print(f"Alphas encontrados: {alphas_encontrados}")


# ============================================================
# 3. VERIFICAR NaN
# ============================================================

print("\n" + "=" * 70)
print("1. VALORES AUSENTES (NaN)")
print("=" * 70)

nan_por_coluna = df.isna().sum()

nan_por_coluna = nan_por_coluna[
    nan_por_coluna > 0
].sort_values(ascending=False)

if nan_por_coluna.empty:
    print("OK - Nenhum valor NaN encontrado.")
else:
    print("ATENÇÃO - Foram encontrados valores NaN:")
    print(nan_por_coluna)

nan_por_coluna.to_csv(
    PASTA_SAIDA / "valores_nan.csv",
    header=["quantidade_nan"]
)


# ============================================================
# 4. VERIFICAR DUPLICATAS
# ============================================================

print("\n" + "=" * 70)
print("2. DUPLICATAS")
print("=" * 70)

chave = [
    "perfil",
    "Re",
    "Mach",
    "alpha"
]

duplicadas = df[
    df.duplicated(
        subset=chave,
        keep=False
    )
].copy()

if duplicadas.empty:
    print("OK - Nenhuma combinação duplicada.")
else:
    print(
        f"ATENÇÃO - {len(duplicadas)} linhas "
        f"participam de duplicatas."
    )

    duplicadas.to_csv(
        PASTA_SAIDA / "linhas_duplicadas.csv",
        index=False
    )


# ============================================================
# 5. CRIAR GRADE TEÓRICA
# ============================================================

print("\n" + "=" * 70)
print("3. COBERTURA / PROVÁVEL CONVERGÊNCIA")
print("=" * 70)

grade_teorica = pd.MultiIndex.from_product(
    [
        perfis,
        reynolds,
        machs,
        ALPHAS_ESPERADOS
    ],
    names=[
        "perfil",
        "Re",
        "Mach",
        "alpha"
    ]
).to_frame(index=False)


# Marca quais condições realmente existem
existentes = (
    df[chave]
    .drop_duplicates()
    .copy()
)

existentes["resultado_xfoil"] = True


grade = grade_teorica.merge(
    existentes,
    on=chave,
    how="left"
)

grade["resultado_xfoil"] = (
    grade["resultado_xfoil"]
    .fillna(False)
    .astype(bool)
)


# ============================================================
# 6. ESTATÍSTICAS GERAIS DE COBERTURA
# ============================================================

total_teorico = len(grade)
total_existente = grade["resultado_xfoil"].sum()
total_ausente = total_teorico - total_existente

taxa_cobertura = (
    total_existente
    / total_teorico
    * 100
)

taxa_ausencia = (
    total_ausente
    / total_teorico
    * 100
)

print(f"\nCombinações teóricas: {total_teorico}")
print(f"Resultados existentes: {total_existente}")
print(f"Resultados ausentes: {total_ausente}")

print(
    f"\nTaxa de cobertura: "
    f"{taxa_cobertura:.2f}%"
)

print(
    f"Taxa de ausência/provável não convergência: "
    f"{taxa_ausencia:.2f}%"
)


# ============================================================
# 7. LISTA COMPLETA DE CASOS AUSENTES
# ============================================================

ausentes = grade[
    ~grade["resultado_xfoil"]
].copy()

ausentes.to_csv(
    PASTA_SAIDA / "casos_ausentes.csv",
    index=False
)


# ============================================================
# 8. CONVERGÊNCIA POR ALPHA
# ============================================================

por_alpha = (
    grade
    .groupby("alpha")
    .agg(
        esperado=("resultado_xfoil", "size"),
        existente=("resultado_xfoil", "sum")
    )
    .reset_index()
)

por_alpha["ausente"] = (
    por_alpha["esperado"]
    - por_alpha["existente"]
)

por_alpha["cobertura_pct"] = (
    por_alpha["existente"]
    / por_alpha["esperado"]
    * 100
)

por_alpha["ausencia_pct"] = (
    por_alpha["ausente"]
    / por_alpha["esperado"]
    * 100
)

print("\n" + "=" * 70)
print("COBERTURA POR ALPHA")
print("=" * 70)

print(
    por_alpha.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

por_alpha.to_csv(
    PASTA_SAIDA / "convergencia_por_alpha.csv",
    index=False
)


# ============================================================
# 9. CONVERGÊNCIA POR REYNOLDS
# ============================================================

por_re = (
    grade
    .groupby("Re")
    .agg(
        esperado=("resultado_xfoil", "size"),
        existente=("resultado_xfoil", "sum")
    )
    .reset_index()
)

por_re["ausente"] = (
    por_re["esperado"]
    - por_re["existente"]
)

por_re["cobertura_pct"] = (
    por_re["existente"]
    / por_re["esperado"]
    * 100
)

por_re["ausencia_pct"] = (
    por_re["ausente"]
    / por_re["esperado"]
    * 100
)

print("\n" + "=" * 70)
print("COBERTURA POR REYNOLDS")
print("=" * 70)

print(
    por_re.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

por_re.to_csv(
    PASTA_SAIDA / "convergencia_por_reynolds.csv",
    index=False
)


# ============================================================
# 10. CONVERGÊNCIA POR PERFIL
# ============================================================

por_perfil = (
    grade
    .groupby("perfil")
    .agg(
        esperado=("resultado_xfoil", "size"),
        existente=("resultado_xfoil", "sum")
    )
    .reset_index()
)

por_perfil["ausente"] = (
    por_perfil["esperado"]
    - por_perfil["existente"]
)

por_perfil["cobertura_pct"] = (
    por_perfil["existente"]
    / por_perfil["esperado"]
    * 100
)

por_perfil["ausencia_pct"] = (
    por_perfil["ausente"]
    / por_perfil["esperado"]
    * 100
)

por_perfil = por_perfil.sort_values(
    by=[
        "cobertura_pct",
        "perfil"
    ]
)

print("\n" + "=" * 70)
print("10 PERFIS COM MENOR COBERTURA")
print("=" * 70)

print(
    por_perfil.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

por_perfil.to_csv(
    PASTA_SAIDA / "convergencia_por_perfil.csv",
    index=False
)


# ============================================================
# 11. PERFIL x REYNOLDS
# ============================================================

perfil_re = (
    grade
    .groupby(
        [
            "perfil",
            "Re"
        ]
    )
    .agg(
        esperado=("resultado_xfoil", "size"),
        existente=("resultado_xfoil", "sum")
    )
    .reset_index()
)

perfil_re["ausente"] = (
    perfil_re["esperado"]
    - perfil_re["existente"]
)

perfil_re["cobertura_pct"] = (
    perfil_re["existente"]
    / perfil_re["esperado"]
    * 100
)

perfil_re.to_csv(
    PASTA_SAIDA / "convergencia_perfil_reynolds.csv",
    index=False
)


# ============================================================
# 12. COMBINAÇÕES PERFIL x RE COMPLETAS
# ============================================================

completas = perfil_re[
    perfil_re["ausente"] == 0
]

incompletas = perfil_re[
    perfil_re["ausente"] > 0
]

print("\n" + "=" * 70)
print("PERFIL x REYNOLDS")
print("=" * 70)

print(
    f"Combinações completas: "
    f"{len(completas)}"
)

print(
    f"Combinações incompletas: "
    f"{len(incompletas)}"
)

print(
    f"Total: "
    f"{len(perfil_re)}"
)


# ============================================================
# 13. QUANTIDADE DE ALPHAS POR PERFIL x RE
# ============================================================

distribuicao_n_alpha = (
    perfil_re["existente"]
    .value_counts()
    .sort_index()
    .rename_axis("numero_de_alphas")
    .reset_index(name="quantidade_perfil_re")
)

print("\n" + "=" * 70)
print("DISTRIBUIÇÃO DO NÚMERO DE ALPHAS POR PERFIL x RE")
print("=" * 70)

print(
    distribuicao_n_alpha.to_string(
        index=False
    )
)

distribuicao_n_alpha.to_csv(
    PASTA_SAIDA / "distribuicao_numero_alphas.csv",
    index=False
)


# ============================================================
# 14. QUALIDADE DOS VALORES AERODINÂMICOS
# ============================================================

print("\n" + "=" * 70)
print("4. QUALIDADE DAS VARIÁVEIS AERODINÂMICAS")
print("=" * 70)


# ------------------------------------------------------------
# CD <= 0
# ------------------------------------------------------------

cd_invalido = df[
    df["CD"] <= 0
].copy()

print(
    f"\nCD <= 0: "
    f"{len(cd_invalido)} linhas"
)

if not cd_invalido.empty:
    cd_invalido.to_csv(
        PASTA_SAIDA / "cd_invalido.csv",
        index=False
    )


# ------------------------------------------------------------
# Infinitos
# ------------------------------------------------------------

colunas_numericas = df.select_dtypes(
    include=np.number
).columns

inf_mask = np.isinf(
    df[colunas_numericas]
)

quantidade_inf = int(
    inf_mask.sum().sum()
)

print(
    f"Valores infinitos: "
    f"{quantidade_inf}"
)


# ============================================================
# 15. VERIFICAR Top_Xtr e Bot_Xtr
# ============================================================

for coluna in [
    "Top_Xtr",
    "Bot_Xtr"
]:

    if coluna in df.columns:

        fora_faixa = df[
            (df[coluna] < 0)
            | (df[coluna] > 1)
        ]

        print(
            f"{coluna} fora de [0, 1]: "
            f"{len(fora_faixa)}"
        )

        if not fora_faixa.empty:
            fora_faixa.to_csv(
                PASTA_SAIDA
                / f"{coluna}_fora_faixa.csv",
                index=False
            )


# ============================================================
# 16. ESTATÍSTICAS AERODINÂMICAS
# ============================================================

variaveis_aero = [
    "CL",
    "CD",
    "CDp",
    "CM"
]

variaveis_aero = [
    c
    for c in variaveis_aero
    if c in df.columns
]

estatisticas_aero = (
    df[variaveis_aero]
    .describe()
    .T
)

print("\n" + "=" * 70)
print("ESTATÍSTICAS DAS VARIÁVEIS AERODINÂMICAS")
print("=" * 70)

print(estatisticas_aero)

estatisticas_aero.to_csv(
    PASTA_SAIDA
    / "estatisticas_aerodinamicas.csv"
)


# ============================================================
# 17. CL/CD
# ============================================================

df_analise = df.copy()

df_analise["CL_CD"] = (
    df_analise["CL"]
    / df_analise["CD"]
)

cl_cd_inf = np.isinf(
    df_analise["CL_CD"]
).sum()

print(
    f"\nCL/CD infinitos: "
    f"{cl_cd_inf}"
)

estatisticas_clcd = (
    df_analise["CL_CD"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .describe()
)

print("\nEstatísticas CL/CD:")
print(estatisticas_clcd)

estatisticas_clcd.to_csv(
    PASTA_SAIDA
    / "estatisticas_cl_cd.csv"
)


# ============================================================
# 18. QUALIDADE DO AJUSTE CST
# ============================================================

colunas_erro_cst = [
    "RMSE_upper",
    "RMSE_lower",
    "MaxError_upper",
    "MaxError_lower"
]

colunas_erro_cst = [
    c
    for c in colunas_erro_cst
    if c in df.columns
]

if colunas_erro_cst:

    qualidade_cst = (
        df[
            ["perfil"]
            + colunas_erro_cst
        ]
        .drop_duplicates(
            subset=["perfil"]
        )
    )

    estatisticas_cst = (
        qualidade_cst[
            colunas_erro_cst
        ]
        .describe()
        .T
    )

    print("\n" + "=" * 70)
    print("QUALIDADE DA PARAMETRIZAÇÃO CST")
    print("=" * 70)

    print(estatisticas_cst)

    estatisticas_cst.to_csv(
        PASTA_SAIDA
        / "estatisticas_erros_cst.csv"
    )

    qualidade_cst.to_csv(
        PASTA_SAIDA
        / "erros_cst_por_perfil.csv",
        index=False
    )


# ============================================================
# 19. GRÁFICO - COBERTURA POR ALPHA
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    por_alpha["alpha"],
    por_alpha["cobertura_pct"]
)

plt.axhline(
    100,
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "Ângulo de ataque α (graus)"
)

plt.ylabel(
    "Cobertura (%)"
)

plt.title(
    "Cobertura dos resultados do XFOIL por ângulo de ataque"
)

plt.xticks(
    ALPHAS_ESPERADOS
)

plt.ylim(
    0,
    105
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PASTA_SAIDA
    / "cobertura_por_alpha.png",
    dpi=300
)

plt.show()


# ============================================================
# 20. GRÁFICO - CASOS AUSENTES POR ALPHA
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    por_alpha["alpha"],
    por_alpha["ausente"]
)

plt.xlabel(
    "Ângulo de ataque α (graus)"
)

plt.ylabel(
    "Número de casos ausentes"
)

plt.title(
    "Resultados ausentes do XFOIL por ângulo de ataque"
)

plt.xticks(
    ALPHAS_ESPERADOS
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PASTA_SAIDA
    / "casos_ausentes_por_alpha.png",
    dpi=300
)

plt.show()


# ============================================================
# 21. GRÁFICO - COBERTURA POR REYNOLDS
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    por_re["Re"].astype(str),
    por_re["cobertura_pct"]
)

plt.xlabel(
    "Número de Reynolds"
)

plt.ylabel(
    "Cobertura (%)"
)

plt.title(
    "Cobertura dos resultados do XFOIL por Reynolds"
)

plt.ylim(
    0,
    105
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PASTA_SAIDA
    / "cobertura_por_reynolds.png",
    dpi=300
)

plt.show()


# ============================================================
# 22. HEATMAP PERFIL x REYNOLDS
# ============================================================

matriz = perfil_re.pivot(
    index="perfil",
    columns="Re",
    values="cobertura_pct"
)

plt.figure(
    figsize=(10, 30)
)

plt.imshow(
    matriz,
    aspect="auto"
)

plt.colorbar(
    label="Cobertura (%)"
)

plt.xlabel(
    "Reynolds"
)

plt.ylabel(
    "Perfil"
)

plt.title(
    "Cobertura de resultados por perfil e Reynolds"
)

plt.xticks(
    range(len(matriz.columns)),
    matriz.columns
)

plt.yticks(
    range(len(matriz.index)),
    matriz.index,
    fontsize=5
)

plt.tight_layout()

plt.savefig(
    PASTA_SAIDA
    / "heatmap_convergencia_perfil_re.png",
    dpi=300
)

plt.show()


# ============================================================
# 23. RESUMO FINAL
# ============================================================

resumo = pd.DataFrame(
    {
        "metrica": [
            "numero_perfis",
            "numero_reynolds",
            "numero_mach",
            "numero_alphas_esperados",
            "casos_teoricos",
            "casos_existentes",
            "casos_ausentes",
            "taxa_cobertura_pct",
            "taxa_ausencia_pct",
            "combinacoes_perfil_re_completas",
            "combinacoes_perfil_re_incompletas",
            "duplicatas",
            "cd_menor_igual_zero",
            "valores_infinitos"
        ],
        "valor": [
            n_perfis,
            n_re,
            n_mach,
            n_alpha,
            total_teorico,
            total_existente,
            total_ausente,
            taxa_cobertura,
            taxa_ausencia,
            len(completas),
            len(incompletas),
            len(duplicadas),
            len(cd_invalido),
            quantidade_inf
        ]
    }
)

resumo.to_csv(
    PASTA_SAIDA
    / "resumo_qualidade_convergencia.csv",
    index=False
)


print("\n" + "=" * 70)
print("RESUMO FINAL")
print("=" * 70)

print(resumo.to_string(index=False))

print(
    f"\nArquivos de diagnóstico salvos em:\n"
    f"{PASTA_SAIDA}"
)

print("\nAnálise concluída.")
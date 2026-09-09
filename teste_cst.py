import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ============================================================
# 1. CONFIGURAÇÃO DE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Output_dados"
ML_DIR = DATA_DIR / "ml_preparado"
OUTPUT_DIR = DATA_DIR / "diagnostico_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARREGAR DADOS
# ============================================================

df = pd.read_csv(DATA_DIR / "database_ml.csv")


# ============================================================
# 3. COLUNAS GEOMÉTRICAS CST
# ============================================================

features_cst = [
    "Au0", "Au1", "Au2", "Au3", "Au4", "Au5", "Au6",
    "Al0", "Al1", "Al2", "Al3", "Al4", "Al5", "Al6",
    "DeltaTE_upper",
    "DeltaTE_lower"
]


# ============================================================
# 4. UM VETOR CST POR PERFIL
# ============================================================

geometrias = (
    df[["perfil"] + features_cst]
    .drop_duplicates(subset="perfil")
    .reset_index(drop=True)
)

print(f"Total de perfis: {len(geometrias)}")


# ============================================================
# 5. PERFIS DE TREINO E TESTE
# ============================================================
# Usa a mesma divisão salva pelo script de preparação do ML.

perfis_treino = pd.read_csv(ML_DIR / "perfis_train.csv")["perfil"].astype(str).tolist()
perfis_teste = pd.read_csv(ML_DIR / "perfis_test.csv")["perfil"].astype(str).tolist()

geo_train = geometrias[
    geometrias["perfil"].isin(perfis_treino)
].copy()

geo_test = geometrias[
    geometrias["perfil"].isin(perfis_teste)
].copy()

print(f"Perfis de treino: {len(geo_train)}")
print(f"Perfis de teste: {len(geo_test)}")


# ============================================================
# 6. NORMALIZAÇÃO
# ============================================================
# O scaler é ajustado somente no treino.

scaler = StandardScaler()

X_geo_train = scaler.fit_transform(geo_train[features_cst])
X_geo_test = scaler.transform(geo_test[features_cst])


# ============================================================
# 7. DISTÂNCIA ATÉ O PERFIL DE TREINO MAIS PRÓXIMO
# ============================================================

nn = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

nn.fit(X_geo_train)

distancias, indices = nn.kneighbors(X_geo_test)

geo_test["distancia_treino"] = distancias[:, 0]
geo_test["vizinho_treino"] = geo_train.iloc[indices[:, 0]]["perfil"].values


# ============================================================
# 8. RANKING
# ============================================================

ranking = (
    geo_test[
        ["perfil", "distancia_treino", "vizinho_treino"]
    ]
    .sort_values("distancia_treino", ascending=False)
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("PERFIS DE TESTE MAIS DISTANTES DO ESPAÇO DE TREINAMENTO")
print("=" * 70)
print(ranking.to_string(index=False))

ranking.to_csv(
    OUTPUT_DIR / "distancia_cst_perfis_teste.csv",
    index=False
)


# ============================================================
# 9. ESTATÍSTICAS
# ============================================================

print("\nEstatísticas das distâncias:")
print(ranking["distancia_treino"].describe())


# ============================================================
# 10. PERFIS PROBLEMÁTICOS
# ============================================================

problematicos = [
    "npl9510",
    "s4096",
    "hs1430",
    "fx68h120"
]

resultado_problematicos = ranking[
    ranking["perfil"].isin(problematicos)
]

print("\n" + "=" * 70)
print("PERFIS PROBLEMÁTICOS")
print("=" * 70)
print(resultado_problematicos.to_string(index=False))


# ============================================================
# 11. PERCENTIL DA DISTÂNCIA
# ============================================================

ranking["percentil_distancia"] = (
    ranking["distancia_treino"].rank(pct=True) * 100
)

print("\nPercentis dos perfis problemáticos:")
print(
    ranking[
        ranking["perfil"].isin(problematicos)
    ][
        ["perfil", "distancia_treino", "percentil_distancia", "vizinho_treino"]
    ].to_string(index=False)
)


# ============================================================
# 12. PCA PARA VISUALIZAÇÃO
# ============================================================

pca = PCA(n_components=2)
X_todos = np.vstack([X_geo_train, X_geo_test])
X_pca = pca.fit_transform(X_todos)

n_train = len(X_geo_train)
pca_train = X_pca[:n_train]
pca_test = X_pca[n_train:]

plt.figure(figsize=(10, 7))
plt.scatter(pca_train[:, 0], pca_train[:, 1], label="Treino", alpha=0.6)
plt.scatter(pca_test[:, 0], pca_test[:, 1], label="Teste", marker="x")

for perfil in problematicos:
    if perfil in geo_test["perfil"].values:
        idx = geo_test.reset_index(drop=True).index[
            geo_test.reset_index(drop=True)["perfil"] == perfil
        ][0]

        plt.scatter(pca_test[idx, 0], pca_test[idx, 1], s=120)
        plt.annotate(
            perfil,
            (pca_test[idx, 0], pca_test[idx, 1]),
            xytext=(5, 5),
            textcoords="offset points"
        )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Distribuição dos perfis no espaço CST")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nVariância explicada pelo PCA:")
print(pca.explained_variance_ratio_)
print("Total:", pca.explained_variance_ratio_.sum())

# ============================================================
# 13. DISTÂNCIA CST x ERRO DO MODELO
# ============================================================
#
# Objetivo:
# verificar se perfis geometricamente mais distantes do conjunto
# de treinamento apresentam maiores erros de previsão.
#
# A análise é realizada na condição de projeto:
# Re = 250000
# alpha = 5°
#
# O arquivo de previsões deve conter:
#
# perfil
# Re
# alpha
# CL_real
# CL_pred
# CD_real
# CD_pred
# CM_real
# CM_pred
#
# Ajuste o caminho/nome do arquivo se necessário.
# ============================================================

PREDICOES_DIR = DATA_DIR / "resultados_xgboost"

candidatos_predicoes = sorted(PREDICOES_DIR.glob("predicoes_*.csv"))

arquivo_predicoes = None
for item in candidatos_predicoes:
    if "Re250000_alpha5" in item.name:
        arquivo_predicoes = item
        break

if arquivo_predicoes is None:
    if candidatos_predicoes:
        arquivo_predicoes = candidatos_predicoes[0]
    else:
        raise FileNotFoundError(
            "Nenhum arquivo de predição foi encontrado em "
            f"{PREDICOES_DIR}."
        )

pred = pd.read_csv(arquivo_predicoes)

# ============================================================
# 14. FILTRAR CONDIÇÃO DE PROJETO
# ============================================================

if {"Re", "alpha"}.issubset(pred.columns):
    pred = pred[
        (pred["Re"] == 250000) &
        (pred["alpha"] == 5.0)
    ].copy()

print("\n" + "=" * 70)
print("ANÁLISE NA CONDIÇÃO DE PROJETO")
print("Re = 250000 | alpha = 5°")
print("=" * 70)

print(f"Perfis encontrados: {len(pred)}")


# ============================================================
# 15. CALCULAR ERROS ABSOLUTOS
# ============================================================

# Compatível com os arquivos gerados no repositório.
if {"CL_real", "CL_pred"}.issubset(pred.columns):
    pred["erro_CL"] = np.abs(pred["CL_real"] - pred["CL_pred"])
else:
    pred["erro_CL"] = np.abs(pred["real"] - pred["previsto"])

if {"CD_real", "CD_pred"}.issubset(pred.columns):
    pred["erro_CD"] = np.abs(pred["CD_real"] - pred["CD_pred"])
else:
    pred["erro_CD"] = np.abs(pred["real"] - pred["previsto"])

if {"CM_real", "CM_pred"}.issubset(pred.columns):
    pred["erro_CM"] = np.abs(pred["CM_real"] - pred["CM_pred"])
else:
    pred["erro_CM"] = np.abs(pred["real"] - pred["previsto"])


# Também calculamos CL/CD real e previsto
if {"CL_real", "CL_pred", "CD_real", "CD_pred"}.issubset(pred.columns):
    pred["CL_CD_real"] = pred["CL_real"] / pred["CD_real"]
    pred["CL_CD_pred"] = pred["CL_pred"] / pred["CD_pred"]
else:
    pred["CL_CD_real"] = np.nan
    pred["CL_CD_pred"] = np.nan

pred["erro_CL_CD"] = np.abs(pred["CL_CD_real"] - pred["CL_CD_pred"])


# ============================================================
# 16. JUNTAR COM DISTÂNCIA CST
# ============================================================

analise = pred.merge(
    ranking[
        [
            "perfil",
            "distancia_treino",
            "percentil_distancia",
            "vizinho_treino"
        ]
    ],
    on="perfil",
    how="inner"
)

print(f"Perfis utilizados na análise: {len(analise)}")


# ============================================================
# 17. CORRELAÇÕES
# ============================================================

variaveis_erro = {
    "CL": "erro_CL",
    "CD": "erro_CD",
    "CM": "erro_CM",
    "CL/CD": "erro_CL_CD"
}

resultados_correlacao = []


print("\n" + "=" * 70)
print("CORRELAÇÃO: DISTÂNCIA CST x ERRO")
print("=" * 70)


for nome, coluna_erro in variaveis_erro.items():

    pearson_r, pearson_p = pearsonr(
        analise["distancia_treino"],
        analise[coluna_erro]
    )

    spearman_rho, spearman_p = spearmanr(
        analise["distancia_treino"],
        analise[coluna_erro]
    )

    resultados_correlacao.append({
        "coeficiente": nome,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p
    })

    print(f"\n{nome}")
    print("-" * 40)

    print(
        f"Pearson:  r = {pearson_r:.4f} "
        f"| p = {pearson_p:.4g}"
    )

    print(
        f"Spearman: ρ = {spearman_rho:.4f} "
        f"| p = {spearman_p:.4g}"
    )


df_correlacoes = pd.DataFrame(
    resultados_correlacao
)

df_correlacoes.to_csv(
    OUTPUT_DIR / "correlacao_distancia_cst_erros.csv",
    index=False
)


# ============================================================
# 18. TABELA COMPLETA
# ============================================================

colunas_saida = [
    "perfil",
    "distancia_treino",
    "percentil_distancia",
    "vizinho_treino",

    "CL_real",
    "CL_pred",
    "erro_CL",

    "CD_real",
    "CD_pred",
    "erro_CD",

    "CM_real",
    "CM_pred",
    "erro_CM",

    "CL_CD_real",
    "CL_CD_pred",
    "erro_CL_CD"
]

analise_saida = analise[
    colunas_saida
].sort_values(
    "distancia_treino",
    ascending=False
)


analise_saida.to_csv(
    OUTPUT_DIR / "distancia_cst_vs_erros.csv",
    index=False
)


print("\n" + "=" * 70)
print("DISTÂNCIA CST E ERROS POR PERFIL")
print("=" * 70)

print(
    analise_saida.to_string(index=False)
)


# ============================================================
# 19. GRÁFICOS DISTÂNCIA CST x ERRO
# ============================================================

for nome, coluna_erro in variaveis_erro.items():

    plt.figure(figsize=(9, 6))

    plt.scatter(
        analise["distancia_treino"],
        analise[coluna_erro],
        s=60,
        alpha=0.7
    )

    # --------------------------------------------------------
    # Identificar os quatro perfis problemáticos
    # --------------------------------------------------------

    for perfil in problematicos:

        linha = analise[
            analise["perfil"] == perfil
        ]

        if not linha.empty:

            x = linha["distancia_treino"].iloc[0]
            y = linha[coluna_erro].iloc[0]

            plt.scatter(
                x,
                y,
                s=130
            )

            plt.annotate(
                perfil,
                (x, y),
                xytext=(6, 6),
                textcoords="offset points"
            )


    # --------------------------------------------------------
    # Linha de tendência linear
    # Apenas para visualização
    # --------------------------------------------------------

    x = analise["distancia_treino"].values
    y = analise[coluna_erro].values

    coef = np.polyfit(x, y, 1)

    x_linha = np.linspace(
        x.min(),
        x.max(),
        100
    )

    y_linha = (
        coef[0] * x_linha +
        coef[1]
    )

    plt.plot(
        x_linha,
        y_linha,
        linestyle="--"
    )


    # --------------------------------------------------------
    # Spearman no próprio gráfico
    # --------------------------------------------------------

    rho, p = spearmanr(
        analise["distancia_treino"],
        analise[coluna_erro]
    )

    plt.text(
        0.05,
        0.95,
        f"Spearman ρ = {rho:.3f}\np = {p:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            alpha=0.2
        )
    )


    plt.xlabel(
        "Distância ao perfil de treino mais próximo"
    )

    plt.ylabel(
        f"Erro absoluto de {nome}"
    )

    plt.title(
        f"Distância no espaço CST × erro de {nome}\n"
        "Re = 250000 | α = 5°"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()


    # --------------------------------------------------------
    # Salvar figura
    # --------------------------------------------------------

    nome_arquivo = (
        nome
        .replace("/", "_")
        .replace(" ", "_")
    )

    plt.savefig(
        OUTPUT_DIR /
        f"distancia_cst_vs_erro_{nome_arquivo}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 20. ANÁLISE SEM OS DOIS OUTLIERS GEOMÉTRICOS EXTREMOS
# ============================================================
#
# s4096 e hs1430 possuem distâncias muito superiores aos demais.
# Portanto, calculamos também as correlações sem esses dois
# perfis para verificar se a relação permanece no restante
# da distribuição.
#
# Isso NÃO significa removê-los da avaliação do modelo.
# É apenas uma análise estatística complementar.
# ============================================================

outliers_extremos = [
    "s4096",
    "hs1430"
]

analise_sem_extremos = analise[
    ~analise["perfil"].isin(outliers_extremos)
].copy()


print("\n" + "=" * 70)
print("CORRELAÇÕES SEM s4096 E hs1430")
print("=" * 70)


resultados_sem_extremos = []


for nome, coluna_erro in variaveis_erro.items():

    pearson_r, pearson_p = pearsonr(
        analise_sem_extremos["distancia_treino"],
        analise_sem_extremos[coluna_erro]
    )

    spearman_rho, spearman_p = spearmanr(
        analise_sem_extremos["distancia_treino"],
        analise_sem_extremos[coluna_erro]
    )

    resultados_sem_extremos.append({
        "coeficiente": nome,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p
    })

    print(f"\n{nome}")
    print("-" * 40)

    print(
        f"Pearson:  r = {pearson_r:.4f} "
        f"| p = {pearson_p:.4g}"
    )

    print(
        f"Spearman: ρ = {spearman_rho:.4f} "
        f"| p = {spearman_p:.4g}"
    )


pd.DataFrame(
    resultados_sem_extremos
).to_csv(
    OUTPUT_DIR /
    "correlacao_distancia_cst_erros_sem_extremos.csv",
    index=False
)


# ============================================================
# 21. RESUMO DOS ARQUIVOS GERADOS
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISE CONCLUÍDA")
print("=" * 70)

print("\nArquivos gerados:")

print(
    OUTPUT_DIR /
    "distancia_cst_perfis_teste.csv"
)

print(
    OUTPUT_DIR /
    "distancia_cst_vs_erros.csv"
)

print(
    OUTPUT_DIR /
    "correlacao_distancia_cst_erros.csv"
)

print(
    OUTPUT_DIR /
    "correlacao_distancia_cst_erros_sem_extremos.csv"
)

print("\nGráficos:")
print("distancia_cst_vs_erro_CL.png")
print("distancia_cst_vs_erro_CD.png")
print("distancia_cst_vs_erro_CM.png")
print("distancia_cst_vs_erro_CL_CD.png")

# A análise revelou associação positiva e estatisticamente significativa entre a distância
# geométrica no espaço CST e os erros de previsão dos modelos substitutos. Na
# condição \(Re=250.000\) e \(\alpha=5^\circ\), foram obtidos coeficientes de Spearman de 0,790
# para \(C_L\), 0,732 para \(C_D\), 0,539 para \(C_M\) e 0,588 para \(C_L/C_D\). A tendência
# permaneceu significativa mesmo após a exclusão dos dois perfis geometricamente
# mais extremos da análise de correlação, indicando que o efeito não decorre
# exclusivamente desses casos. Os resultados sugerem redução da capacidade de
# generalização à medida que as geometrias se afastam das regiões representadas no
# conjunto de treinamento.


# qual distância CST será considerada aceitável durante a otimização.
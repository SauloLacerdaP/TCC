import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DADOS = Path(
    r"C:\Repositorios\TCC\Output_dados\ml_preparado"
)

PASTA_SAIDA = Path(
    r"C:\Repositorios\TCC\Output_dados\diagnostico_split"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)


FEATURES = [
    "Au0",
    "Au1",
    "Au2",
    "Au3",
    "Au4",
    "Au5",
    "Au6",
    "Al0",
    "Al1",
    "Al2",
    "Al3",
    "Al4",
    "Al5",
    "Al6",
    "DeltaTE_upper",
    "DeltaTE_lower",
    "Re",
    "alpha"
]

TARGETS = [
    "CL",
    "CD",
    "CM"
]


# ============================================================
# 1. CARREGAR SPLITS
# ============================================================

train = pd.read_csv(
    PASTA_DADOS / "train.csv"
)

valid = pd.read_csv(
    PASTA_DADOS / "validation.csv"
)

test = pd.read_csv(
    PASTA_DADOS / "test.csv"
)


print("=" * 70)
print("DIAGNÓSTICO DO SPLIT")
print("=" * 70)

print(
    f"\nTreino: {len(train)} linhas / "
    f"{train['perfil'].nunique()} perfis"
)

print(
    f"Validação: {len(valid)} linhas / "
    f"{valid['perfil'].nunique()} perfis"
)

print(
    f"Teste: {len(test)} linhas / "
    f"{test['perfil'].nunique()} perfis"
)


# ============================================================
# 2. DISTRIBUIÇÃO DOS TARGETS
# ============================================================

estatisticas = []

for nome, df in [
    ("treino", train),
    ("validacao", valid),
    ("teste", test)
]:

    for target in TARGETS:

        estatisticas.append(
            {
                "conjunto": nome,
                "target": target,
                "media": df[target].mean(),
                "std": df[target].std(),
                "min": df[target].min(),
                "q25": df[target].quantile(0.25),
                "mediana": df[target].median(),
                "q75": df[target].quantile(0.75),
                "max": df[target].max()
            }
        )


estatisticas = pd.DataFrame(
    estatisticas
)

print("\n" + "=" * 70)
print("ESTATÍSTICAS DOS TARGETS")
print("=" * 70)

print(
    estatisticas.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

estatisticas.to_csv(
    PASTA_SAIDA / "estatisticas_targets_split.csv",
    index=False
)


# ============================================================
# 3. FAIXA DAS FEATURES NO TREINO
# ============================================================

min_train = train[FEATURES].min()
max_train = train[FEATURES].max()


def analisar_extrapolacao(
    df,
    nome
):

    resultados = []

    for feature in FEATURES:

        abaixo = (
            df[feature]
            < min_train[feature]
        )

        acima = (
            df[feature]
            > max_train[feature]
        )

        fora = abaixo | acima

        resultados.append(
            {
                "conjunto": nome,
                "feature": feature,
                "min_treino": min_train[feature],
                "max_treino": max_train[feature],
                "min_conjunto": df[feature].min(),
                "max_conjunto": df[feature].max(),
                "linhas_fora_faixa": int(
                    fora.sum()
                ),
                "percentual_fora_faixa": (
                    fora.mean()
                    * 100
                )
            }
        )

    return pd.DataFrame(
        resultados
    )


extrap_valid = analisar_extrapolacao(
    valid,
    "validacao"
)

extrap_test = analisar_extrapolacao(
    test,
    "teste"
)


extrapolacao = pd.concat(
    [
        extrap_valid,
        extrap_test
    ],
    ignore_index=True
)


print("\n" + "=" * 70)
print("EXTRAPOLAÇÃO DAS FEATURES")
print("=" * 70)

print(
    extrapolacao[
        extrapolacao[
            "linhas_fora_faixa"
        ] > 0
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


extrapolacao.to_csv(
    PASTA_SAIDA / "extrapolacao_features.csv",
    index=False
)


# ============================================================
# 4. NÚMERO DE FEATURES FORA DA FAIXA POR LINHA
# ============================================================

def contar_features_fora(
    df
):

    contagem = np.zeros(
        len(df),
        dtype=int
    )

    for feature in FEATURES:

        fora = (
            (df[feature] < min_train[feature])
            |
            (df[feature] > max_train[feature])
        )

        contagem += fora.astype(int)

    return contagem


valid = valid.copy()
test = test.copy()


valid[
    "n_features_fora_treino"
] = contar_features_fora(
    valid
)

test[
    "n_features_fora_treino"
] = contar_features_fora(
    test
)


print("\n" + "=" * 70)
print("LINHAS COM EXTRAPOLAÇÃO")
print("=" * 70)

for nome, df in [
    ("Validação", valid),
    ("Teste", test)
]:

    linhas_fora = (
        df[
            "n_features_fora_treino"
        ] > 0
    ).sum()

    print(
        f"{nome}: "
        f"{linhas_fora}/{len(df)} "
        f"({linhas_fora / len(df) * 100:.2f}%)"
    )


# ============================================================
# 5. PERFIS COM MAIOR EXTRAPOLAÇÃO
# ============================================================

def resumo_por_perfil(
    df,
    nome
):

    resumo = (
        df
        .groupby("perfil")
        .agg(
            linhas=("perfil", "size"),
            media_features_fora=(
                "n_features_fora_treino",
                "mean"
            ),
            max_features_fora=(
                "n_features_fora_treino",
                "max"
            )
        )
        .reset_index()
    )

    resumo["conjunto"] = nome

    return resumo


perfil_valid = resumo_por_perfil(
    valid,
    "validacao"
)

perfil_test = resumo_por_perfil(
    test,
    "teste"
)


resumo_perfis = pd.concat(
    [
        perfil_valid,
        perfil_test
    ],
    ignore_index=True
)


resumo_perfis = resumo_perfis.sort_values(
    by=[
        "media_features_fora",
        "max_features_fora"
    ],
    ascending=False
)


print("\n" + "=" * 70)
print("PERFIS COM MAIOR EXTRAPOLAÇÃO")
print("=" * 70)

print(
    resumo_perfis.head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


resumo_perfis.to_csv(
    PASTA_SAIDA / "extrapolacao_por_perfil.csv",
    index=False
)


# ============================================================
# 6. DISTÂNCIA GEOMÉTRICA DOS PERFIS
# ============================================================

# Para avaliar geometria, retiramos Re e alpha,
# porque queremos medir distância entre os perfis CST.

FEATURES_GEOMETRIA = [
    "Au0",
    "Au1",
    "Au2",
    "Au3",
    "Au4",
    "Au5",
    "Au6",
    "Al0",
    "Al1",
    "Al2",
    "Al3",
    "Al4",
    "Al5",
    "Al6",
    "DeltaTE_upper",
    "DeltaTE_lower"
]


# Um vetor geométrico por perfil
geom_train = (
    train[
        ["perfil"]
        + FEATURES_GEOMETRIA
    ]
    .drop_duplicates(
        subset=["perfil"]
    )
    .reset_index(drop=True)
)

geom_valid = (
    valid[
        ["perfil"]
        + FEATURES_GEOMETRIA
    ]
    .drop_duplicates(
        subset=["perfil"]
    )
    .reset_index(drop=True)
)

geom_test = (
    test[
        ["perfil"]
        + FEATURES_GEOMETRIA
    ]
    .drop_duplicates(
        subset=["perfil"]
    )
    .reset_index(drop=True)
)


# Padronização ajustada apenas no treino
scaler_geom = StandardScaler()

X_geom_train = scaler_geom.fit_transform(
    geom_train[
        FEATURES_GEOMETRIA
    ]
)

X_geom_valid = scaler_geom.transform(
    geom_valid[
        FEATURES_GEOMETRIA
    ]
)

X_geom_test = scaler_geom.transform(
    geom_test[
        FEATURES_GEOMETRIA
    ]
)


# Vizinho mais próximo no treino
nn = NearestNeighbors(
    n_neighbors=1
)

nn.fit(
    X_geom_train
)


dist_valid, idx_valid = nn.kneighbors(
    X_geom_valid
)

dist_test, idx_test = nn.kneighbors(
    X_geom_test
)


geom_valid[
    "distancia_treino_mais_proximo"
] = dist_valid[:, 0]

geom_valid[
    "perfil_treino_mais_proximo"
] = geom_train.iloc[
    idx_valid[:, 0]
]["perfil"].values


geom_test[
    "distancia_treino_mais_proximo"
] = dist_test[:, 0]

geom_test[
    "perfil_treino_mais_proximo"
] = geom_train.iloc[
    idx_test[:, 0]
]["perfil"].values


print("\n" + "=" * 70)
print("DISTÂNCIA GEOMÉTRICA AO TREINO")
print("=" * 70)


print(
    "\nValidação:"
)

print(
    geom_valid[
        "distancia_treino_mais_proximo"
    ].describe()
)


print(
    "\nTeste:"
)

print(
    geom_test[
        "distancia_treino_mais_proximo"
    ].describe()
)


geom_valid.sort_values(
    by="distancia_treino_mais_proximo",
    ascending=False
).to_csv(
    PASTA_SAIDA
    / "distancia_geometrica_validacao.csv",
    index=False
)


geom_test.sort_values(
    by="distancia_treino_mais_proximo",
    ascending=False
).to_csv(
    PASTA_SAIDA
    / "distancia_geometrica_teste.csv",
    index=False
)


# ============================================================
# 7. GRÁFICO DAS DISTÂNCIAS
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.hist(
    geom_valid[
        "distancia_treino_mais_proximo"
    ],
    bins=15,
    alpha=0.6,
    label="Validação"
)

plt.hist(
    geom_test[
        "distancia_treino_mais_proximo"
    ],
    bins=15,
    alpha=0.6,
    label="Teste"
)

plt.xlabel(
    "Distância padronizada ao perfil de treino mais próximo"
)

plt.ylabel(
    "Número de perfis"
)

plt.title(
    "Distância geométrica ao conjunto de treino"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PASTA_SAIDA
    / "distancia_geometrica_validacao_teste.png",
    dpi=300
)

plt.close()


# ============================================================
# 8. DISTRIBUIÇÃO DOS TARGETS
# ============================================================

for target in TARGETS:

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        train[target],
        bins=30,
        alpha=0.5,
        label="Treino"
    )

    plt.hist(
        valid[target],
        bins=30,
        alpha=0.5,
        label="Validação"
    )

    plt.hist(
        test[target],
        bins=30,
        alpha=0.5,
        label="Teste"
    )

    plt.xlabel(
        target
    )

    plt.ylabel(
        "Frequência"
    )

    plt.title(
        f"Distribuição de {target}"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / f"distribuicao_{target}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 9. RESUMO FINAL
# ============================================================

print("\n" + "=" * 70)
print("DIAGNÓSTICO DO SPLIT FINALIZADO")
print("=" * 70)

print(
    f"\nArquivos salvos em:"
)

print(
    PASTA_SAIDA
)

print(
    "\nArquivos principais:"
)

print(
    "  - estatisticas_targets_split.csv"
)

print(
    "  - extrapolacao_features.csv"
)

print(
    "  - extrapolacao_por_perfil.csv"
)

print(
    "  - distancia_geometrica_validacao.csv"
)

print(
    "  - distancia_geometrica_teste.csv"
)

print(
    "\nDiagnóstico concluído."
)
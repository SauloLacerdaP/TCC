import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import joblib


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_ENTRADA = Path(
    r"C:\Repositorios\TCC\Output_dados\database_ml.csv"
)

PASTA_SAIDA = Path(
    r"C:\Repositorios\TCC\Output_dados\ml_preparado"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)


RANDOM_STATE = 42

# Proporções aproximadas por PERFIL
TRAIN_SIZE = 0.70
VALID_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# 1. CARREGAMENTO
# ============================================================

print("=" * 70)
print("PREPARAÇÃO DO DATASET PARA MACHINE LEARNING")
print("=" * 70)

df = pd.read_csv(
    ARQUIVO_ENTRADA
)


print(f"\nArquivo carregado:")
print(ARQUIVO_ENTRADA)

print(
    f"\nNúmero de linhas: "
    f"{len(df)}"
)

print(
    f"Número de perfis: "
    f"{df['perfil'].nunique()}"
)

print(
    f"Número de colunas: "
    f"{len(df.columns)}"
)


# ============================================================
# 2. VERIFICAÇÕES INICIAIS
# ============================================================

print("\n" + "=" * 70)
print("VERIFICAÇÕES INICIAIS")
print("=" * 70)


# ------------------------------------------------------------
# NaN
# ------------------------------------------------------------

total_nan = int(
    df.isna().sum().sum()
)

print(
    f"Valores NaN: "
    f"{total_nan}"
)


# ------------------------------------------------------------
# Infinitos
# ------------------------------------------------------------

colunas_numericas = df.select_dtypes(
    include=np.number
).columns

total_inf = int(
    np.isinf(
        df[colunas_numericas]
    ).sum().sum()
)

print(
    f"Valores infinitos: "
    f"{total_inf}"
)


# ------------------------------------------------------------
# Duplicatas
# ------------------------------------------------------------

chave = [
    "perfil",
    "Re",
    "Mach",
    "alpha"
]

duplicadas = df.duplicated(
    subset=chave
).sum()

print(
    f"Duplicatas: "
    f"{duplicadas}"
)


# ------------------------------------------------------------
# Impedir continuação em caso de problema
# ------------------------------------------------------------

if total_nan > 0:
    raise ValueError(
        "Existem valores NaN no dataset."
    )

if total_inf > 0:
    raise ValueError(
        "Existem valores infinitos no dataset."
    )

if duplicadas > 0:
    raise ValueError(
        "Existem combinações duplicadas no dataset."
    )


# ============================================================
# 3. DEFINIÇÃO DAS FEATURES
# ============================================================

# ------------------------------------------------------------
# Variáveis geométricas CST
# ------------------------------------------------------------

FEATURES_CST = [
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


# ------------------------------------------------------------
# Condições de operação
# ------------------------------------------------------------

FEATURES_OPERACAO = [
    "Re",
    "alpha"
]


# ------------------------------------------------------------
# Features finais
# ------------------------------------------------------------

FEATURES = (
    FEATURES_CST
    + FEATURES_OPERACAO
)


# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

TARGETS = [
    "CL",
    "CD",
    "CM"
]


print("\n" + "=" * 70)
print("FEATURES UTILIZADAS")
print("=" * 70)

for feature in FEATURES:
    print(
        f"  - {feature}"
    )


print("\nTargets:")

for target in TARGETS:
    print(
        f"  - {target}"
    )


print(
    f"\nTotal de features: "
    f"{len(FEATURES)}"
)


# ============================================================
# 4. VERIFICAR COLUNAS
# ============================================================

colunas_necessarias = (
    ["perfil"]
    + FEATURES
    + TARGETS
)

faltantes = [
    coluna
    for coluna in colunas_necessarias
    if coluna not in df.columns
]


if faltantes:

    raise ValueError(
        f"Colunas ausentes no dataset: "
        f"{faltantes}"
    )


# ============================================================
# 5. EXPLICAÇÃO DE VARIÁVEIS NÃO UTILIZADAS
# ============================================================

print("\n" + "=" * 70)
print("VARIÁVEIS NÃO UTILIZADAS COMO FEATURES")
print("=" * 70)

print(
    "perfil -> usado exclusivamente para separação dos grupos"
)

print(
    "Mach -> removido porque é constante"
)

print(
    "CDp -> saída do XFOIL, portanto poderia causar vazamento"
)

print(
    "Top_Xtr/Bot_Xtr -> saídas do XFOIL"
)

print(
    "RMSE/MaxError CST -> métricas de qualidade da parametrização"
)


# ============================================================
# 6. VERIFICAR MACH
# ============================================================

if "Mach" in df.columns:

    machs = sorted(
        df["Mach"].unique()
    )

    print(
        f"\nValores de Mach encontrados: "
        f"{machs}"
    )

    if len(machs) == 1:

        print(
            "Mach é constante e não será utilizado "
            "como feature."
        )

    else:

        print(
            "ATENÇÃO: existem múltiplos valores de Mach."
        )


# ============================================================
# 7. SEPARAÇÃO POR PERFIL
# ============================================================

print("\n" + "=" * 70)
print("DIVISÃO TREINO / VALIDAÇÃO / TESTE")
print("=" * 70)


# ------------------------------------------------------------
# Primeiro split:
# 70% treino
# 30% temporário
# ------------------------------------------------------------

split_train = GroupShuffleSplit(
    n_splits=1,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_STATE
)


indices_train, indices_temp = next(
    split_train.split(
        df,
        groups=df["perfil"]
    )
)


df_train = df.iloc[
    indices_train
].copy()

df_temp = df.iloc[
    indices_temp
].copy()


# ------------------------------------------------------------
# Segundo split:
# divide os 30% restantes igualmente
# entre validação e teste.
# ------------------------------------------------------------

proporcao_valid_temp = (
    VALID_SIZE
    / (
        VALID_SIZE
        + TEST_SIZE
    )
)


split_valid_test = GroupShuffleSplit(
    n_splits=1,
    train_size=proporcao_valid_temp,
    random_state=RANDOM_STATE
)


indices_valid, indices_test = next(
    split_valid_test.split(
        df_temp,
        groups=df_temp["perfil"]
    )
)


df_valid = df_temp.iloc[
    indices_valid
].copy()

df_test = df_temp.iloc[
    indices_test
].copy()


# ============================================================
# 8. VERIFICAR SOBREPOSIÇÃO DE PERFIS
# ============================================================

perfis_train = set(
    df_train["perfil"].unique()
)

perfis_valid = set(
    df_valid["perfil"].unique()
)

perfis_test = set(
    df_test["perfil"].unique()
)


overlap_train_valid = (
    perfis_train
    & perfis_valid
)

overlap_train_test = (
    perfis_train
    & perfis_test
)

overlap_valid_test = (
    perfis_valid
    & perfis_test
)


if (
    overlap_train_valid
    or overlap_train_test
    or overlap_valid_test
):

    raise RuntimeError(
        "ERRO: existem perfis presentes "
        "em mais de um conjunto."
    )


print(
    "\nOK - Nenhum perfil aparece "
    "em mais de um conjunto."
)


# ============================================================
# 9. RESUMO DA DIVISÃO
# ============================================================

def resumo_split(
    nome,
    dados
):

    print(
        f"\n{nome}"
    )

    print(
        f"  Perfis: "
        f"{dados['perfil'].nunique()}"
    )

    print(
        f"  Linhas: "
        f"{len(dados)}"
    )

    print(
        f"  Percentual das linhas: "
        f"{len(dados) / len(df) * 100:.2f}%"
    )


resumo_split(
    "TREINO",
    df_train
)

resumo_split(
    "VALIDAÇÃO",
    df_valid
)

resumo_split(
    "TESTE",
    df_test
)


# ============================================================
# 10. CRIAR X E y
# ============================================================

X_train = df_train[
    FEATURES
].copy()

X_valid = df_valid[
    FEATURES
].copy()

X_test = df_test[
    FEATURES
].copy()


y_train = df_train[
    TARGETS
].copy()

y_valid = df_valid[
    TARGETS
].copy()

y_test = df_test[
    TARGETS
].copy()


# ============================================================
# 11. NORMALIZAÇÃO
# ============================================================

# IMPORTANTE:
# O scaler é ajustado SOMENTE usando o conjunto de treino.
#
# Isso evita vazamento de informação dos conjuntos
# de validação e teste.


scaler = StandardScaler()


X_train_scaled_array = (
    scaler.fit_transform(
        X_train
    )
)

X_valid_scaled_array = (
    scaler.transform(
        X_valid
    )
)

X_test_scaled_array = (
    scaler.transform(
        X_test
    )
)


# Converter novamente para DataFrame
X_train_scaled = pd.DataFrame(
    X_train_scaled_array,
    columns=FEATURES,
    index=X_train.index
)

X_valid_scaled = pd.DataFrame(
    X_valid_scaled_array,
    columns=FEATURES,
    index=X_valid.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled_array,
    columns=FEATURES,
    index=X_test.index
)


# ============================================================
# 12. SALVAR SCALER
# ============================================================

ARQUIVO_SCALER = (
    PASTA_SAIDA
    / "standard_scaler.pkl"
)


joblib.dump(
    scaler,
    ARQUIVO_SCALER
)


# ============================================================
# 13. SALVAR DATASETS COMPLETOS
# ============================================================

df_train.to_csv(
    PASTA_SAIDA
    / "train.csv",
    index=False
)

df_valid.to_csv(
    PASTA_SAIDA
    / "validation.csv",
    index=False
)

df_test.to_csv(
    PASTA_SAIDA
    / "test.csv",
    index=False
)


# ============================================================
# 14. SALVAR X NÃO NORMALIZADO
# ============================================================

X_train.to_csv(
    PASTA_SAIDA
    / "X_train.csv",
    index=False
)

X_valid.to_csv(
    PASTA_SAIDA
    / "X_validation.csv",
    index=False
)

X_test.to_csv(
    PASTA_SAIDA
    / "X_test.csv",
    index=False
)


# ============================================================
# 15. SALVAR X NORMALIZADO
# ============================================================

X_train_scaled.to_csv(
    PASTA_SAIDA
    / "X_train_scaled.csv",
    index=False
)

X_valid_scaled.to_csv(
    PASTA_SAIDA
    / "X_validation_scaled.csv",
    index=False
)

X_test_scaled.to_csv(
    PASTA_SAIDA
    / "X_test_scaled.csv",
    index=False
)


# ============================================================
# 16. SALVAR TARGETS
# ============================================================

y_train.to_csv(
    PASTA_SAIDA
    / "y_train.csv",
    index=False
)

y_valid.to_csv(
    PASTA_SAIDA
    / "y_validation.csv",
    index=False
)

y_test.to_csv(
    PASTA_SAIDA
    / "y_test.csv",
    index=False
)


# ============================================================
# 17. SALVAR LISTA DE PERFIS
# ============================================================

pd.DataFrame(
    sorted(perfis_train),
    columns=["perfil"]
).to_csv(
    PASTA_SAIDA
    / "perfis_train.csv",
    index=False
)


pd.DataFrame(
    sorted(perfis_valid),
    columns=["perfil"]
).to_csv(
    PASTA_SAIDA
    / "perfis_validation.csv",
    index=False
)


pd.DataFrame(
    sorted(perfis_test),
    columns=["perfil"]
).to_csv(
    PASTA_SAIDA
    / "perfis_test.csv",
    index=False
)


# ============================================================
# 18. ESTATÍSTICAS DOS TARGETS
# ============================================================

estatisticas_targets = []

for nome, dados in [
    ("train", df_train),
    ("validation", df_valid),
    ("test", df_test)
]:

    for target in TARGETS:

        estatisticas_targets.append(
            {
                "conjunto": nome,
                "target": target,
                "media": dados[target].mean(),
                "std": dados[target].std(),
                "min": dados[target].min(),
                "max": dados[target].max()
            }
        )


estatisticas_targets = pd.DataFrame(
    estatisticas_targets
)


estatisticas_targets.to_csv(
    PASTA_SAIDA
    / "estatisticas_targets.csv",
    index=False
)


# ============================================================
# 19. RELATÓRIO DO SPLIT
# ============================================================

relatorio_split = pd.DataFrame(
    {
        "conjunto": [
            "treino",
            "validacao",
            "teste"
        ],

        "numero_perfis": [
            len(perfis_train),
            len(perfis_valid),
            len(perfis_test)
        ],

        "numero_linhas": [
            len(df_train),
            len(df_valid),
            len(df_test)
        ],

        "percentual_linhas": [
            len(df_train)
            / len(df)
            * 100,

            len(df_valid)
            / len(df)
            * 100,

            len(df_test)
            / len(df)
            * 100
        ]
    }
)


relatorio_split.to_csv(
    PASTA_SAIDA
    / "relatorio_split.csv",
    index=False
)


# ============================================================
# 20. SALVAR FEATURES E TARGETS
# ============================================================

pd.DataFrame(
    {
        "feature": FEATURES
    }
).to_csv(
    PASTA_SAIDA
    / "features_utilizadas.csv",
    index=False
)


pd.DataFrame(
    {
        "target": TARGETS
    }
).to_csv(
    PASTA_SAIDA
    / "targets_utilizados.csv",
    index=False
)


# ============================================================
# 21. RESUMO FINAL
# ============================================================

print("\n" + "=" * 70)
print("PREPARAÇÃO FINALIZADA")
print("=" * 70)


print(
    f"\nDataset original:"
)

print(
    f"  Perfis: "
    f"{df['perfil'].nunique()}"
)

print(
    f"  Observações: "
    f"{len(df)}"
)


print(
    f"\nTreino:"
)

print(
    f"  Perfis: "
    f"{len(perfis_train)}"
)

print(
    f"  Observações: "
    f"{len(df_train)}"
)


print(
    f"\nValidação:"
)

print(
    f"  Perfis: "
    f"{len(perfis_valid)}"
)

print(
    f"  Observações: "
    f"{len(df_valid)}"
)


print(
    f"\nTeste:"
)

print(
    f"  Perfis: "
    f"{len(perfis_test)}"
)

print(
    f"  Observações: "
    f"{len(df_test)}"
)


print(
    f"\nNúmero de features: "
    f"{len(FEATURES)}"
)

print(
    f"Targets: "
    f"{TARGETS}"
)


print(
    "\nSeparação realizada por PERFIL."
)

print(
    "Nenhum perfil é compartilhado "
    "entre treino, validação e teste."
)


print(
    f"\nArquivos salvos em:"
)

print(
    PASTA_SAIDA
)


print(
    "\nPreparação para Machine Learning concluída."
)
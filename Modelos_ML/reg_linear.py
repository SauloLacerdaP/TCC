import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

import joblib


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DADOS = Path(
    r"C:\Repositorios\TCC\Output_dados\ml_preparado"
)

PASTA_SAIDA = Path(
    r"C:\Repositorios\TCC\Output_dados\resultados_linear_regression"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)


TARGETS = [
    "CL",
    "CD",
    "CM"
]


# ============================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================

print("=" * 70)
print("REGRESSÃO LINEAR - BASELINE")
print("=" * 70)


X_train = pd.read_csv(
    PASTA_DADOS / "X_train_scaled.csv"
)

X_valid = pd.read_csv(
    PASTA_DADOS / "X_validation_scaled.csv"
)

X_test = pd.read_csv(
    PASTA_DADOS / "X_test_scaled.csv"
)


y_train = pd.read_csv(
    PASTA_DADOS / "y_train.csv"
)

y_valid = pd.read_csv(
    PASTA_DADOS / "y_validation.csv"
)

y_test = pd.read_csv(
    PASTA_DADOS / "y_test.csv"
)


print("\nDados carregados com sucesso.")

print(
    f"Treino: "
    f"{X_train.shape}"
)

print(
    f"Validação: "
    f"{X_valid.shape}"
)

print(
    f"Teste: "
    f"{X_test.shape}"
)


# ============================================================
# 2. VERIFICAÇÕES
# ============================================================

if len(X_train) != len(y_train):
    raise ValueError(
        "X_train e y_train possuem tamanhos diferentes."
    )

if len(X_valid) != len(y_valid):
    raise ValueError(
        "X_valid e y_valid possuem tamanhos diferentes."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test e y_test possuem tamanhos diferentes."
    )


# ============================================================
# 3. FUNÇÃO DE MÉTRICAS
# ============================================================

def calcular_metricas(
    y_real,
    y_pred
):

    mse = mean_squared_error(
        y_real,
        y_pred
    )

    rmse = np.sqrt(
        mse
    )

    r2 = r2_score(
        y_real,
        y_pred
    )

    return {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 4. FUNÇÃO PARA GRÁFICO REAL x PREVISTO
# ============================================================

def plot_real_vs_pred(
    y_real,
    y_pred,
    target,
    conjunto,
    arquivo_saida
):

    plt.figure(
        figsize=(7, 7)
    )

    plt.scatter(
        y_real,
        y_pred,
        alpha=0.6
    )

    minimo = min(
        y_real.min(),
        y_pred.min()
    )

    maximo = max(
        y_real.max(),
        y_pred.max()
    )

    plt.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--"
    )

    plt.xlabel(
        f"{target} real"
    )

    plt.ylabel(
        f"{target} previsto"
    )

    plt.title(
        f"Regressão Linear - {target}\n"
        f"{conjunto}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        arquivo_saida,
        dpi=300
    )

    plt.close()


# ============================================================
# 5. TREINAMENTO
# ============================================================

resultados_metricas = []

predicoes_salvas = {}


for target in TARGETS:

    print("\n" + "=" * 70)

    print(
        f"TREINANDO MODELO PARA {target}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Selecionar target
    # --------------------------------------------------------

    y_train_target = y_train[
        target
    ]

    y_valid_target = y_valid[
        target
    ]

    y_test_target = y_test[
        target
    ]


    # --------------------------------------------------------
    # Criar modelo
    # --------------------------------------------------------

    modelo = LinearRegression()


    # --------------------------------------------------------
    # Treinar
    # --------------------------------------------------------

    modelo.fit(
        X_train,
        y_train_target
    )


    # --------------------------------------------------------
    # Predições
    # --------------------------------------------------------

    pred_train = modelo.predict(
        X_train
    )

    pred_valid = modelo.predict(
        X_valid
    )

    pred_test = modelo.predict(
        X_test
    )


    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    metricas_train = calcular_metricas(
        y_train_target,
        pred_train
    )

    metricas_valid = calcular_metricas(
        y_valid_target,
        pred_valid
    )

    metricas_test = calcular_metricas(
        y_test_target,
        pred_test
    )


    # --------------------------------------------------------
    # Mostrar métricas
    # --------------------------------------------------------

    print("\nTREINO")

    print(
        f"R²:   "
        f"{metricas_train['R2']:.6f}"
    )

    print(
        f"RMSE: "
        f"{metricas_train['RMSE']:.6f}"
    )

    print(
        f"MSE:  "
        f"{metricas_train['MSE']:.8f}"
    )


    print("\nVALIDAÇÃO")

    print(
        f"R²:   "
        f"{metricas_valid['R2']:.6f}"
    )

    print(
        f"RMSE: "
        f"{metricas_valid['RMSE']:.6f}"
    )

    print(
        f"MSE:  "
        f"{metricas_valid['MSE']:.8f}"
    )


    print("\nTESTE")

    print(
        f"R²:   "
        f"{metricas_test['R2']:.6f}"
    )

    print(
        f"RMSE: "
        f"{metricas_test['RMSE']:.6f}"
    )

    print(
        f"MSE:  "
        f"{metricas_test['MSE']:.8f}"
    )


    # --------------------------------------------------------
    # Salvar métricas
    # --------------------------------------------------------

    for conjunto, metricas in [
        (
            "treino",
            metricas_train
        ),
        (
            "validacao",
            metricas_valid
        ),
        (
            "teste",
            metricas_test
        )
    ]:

        resultados_metricas.append(
            {
                "modelo": "LinearRegression",
                "target": target,
                "conjunto": conjunto,
                "R2": metricas["R2"],
                "RMSE": metricas["RMSE"],
                "MSE": metricas["MSE"]
            }
        )


    # --------------------------------------------------------
    # Salvar modelo
    # --------------------------------------------------------

    caminho_modelo = (
        PASTA_SAIDA
        / f"linear_regression_{target}.pkl"
    )

    joblib.dump(
        modelo,
        caminho_modelo
    )


    # --------------------------------------------------------
    # Salvar predições
    # --------------------------------------------------------

    predicoes_salvas[
        target
    ] = {
        "train_real": y_train_target,
        "train_pred": pred_train,

        "valid_real": y_valid_target,
        "valid_pred": pred_valid,

        "test_real": y_test_target,
        "test_pred": pred_test
    }


    # --------------------------------------------------------
    # Gráficos
    # --------------------------------------------------------

    plot_real_vs_pred(
        y_train_target,
        pred_train,
        target,
        "Treino",
        PASTA_SAIDA
        / f"{target}_real_vs_pred_treino.png"
    )

    plot_real_vs_pred(
        y_valid_target,
        pred_valid,
        target,
        "Validação",
        PASTA_SAIDA
        / f"{target}_real_vs_pred_validacao.png"
    )

    plot_real_vs_pred(
        y_test_target,
        pred_test,
        target,
        "Teste",
        PASTA_SAIDA
        / f"{target}_real_vs_pred_teste.png"
    )


# ============================================================
# 6. SALVAR TABELA DE MÉTRICAS
# ============================================================

df_metricas = pd.DataFrame(
    resultados_metricas
)

df_metricas = df_metricas[
    [
        "modelo",
        "target",
        "conjunto",
        "R2",
        "RMSE",
        "MSE"
    ]
]


df_metricas.to_csv(
    PASTA_SAIDA
    / "metricas_linear_regression.csv",
    index=False
)


# ============================================================
# 7. SALVAR PREDIÇÕES
# ============================================================

for target in TARGETS:

    dados = predicoes_salvas[
        target
    ]


    df_pred_train = pd.DataFrame(
        {
            "real": dados[
                "train_real"
            ].values,

            "previsto": dados[
                "train_pred"
            ]
        }
    )

    df_pred_valid = pd.DataFrame(
        {
            "real": dados[
                "valid_real"
            ].values,

            "previsto": dados[
                "valid_pred"
            ]
        }
    )

    df_pred_test = pd.DataFrame(
        {
            "real": dados[
                "test_real"
            ].values,

            "previsto": dados[
                "test_pred"
            ]
        }
    )


    df_pred_train.to_csv(
        PASTA_SAIDA
        / f"predicoes_{target}_treino.csv",
        index=False
    )

    df_pred_valid.to_csv(
        PASTA_SAIDA
        / f"predicoes_{target}_validacao.csv",
        index=False
    )

    df_pred_test.to_csv(
        PASTA_SAIDA
        / f"predicoes_{target}_teste.csv",
        index=False
    )


# ============================================================
# 8. COEFICIENTES DOS MODELOS
# ============================================================

nomes_features = X_train.columns


for target in TARGETS:

    caminho_modelo = (
        PASTA_SAIDA
        / f"linear_regression_{target}.pkl"
    )

    modelo = joblib.load(
        caminho_modelo
    )


    df_coef = pd.DataFrame(
        {
            "feature": nomes_features,
            "coeficiente": modelo.coef_
        }
    )


    df_coef["abs_coeficiente"] = (
        df_coef[
            "coeficiente"
        ].abs()
    )


    df_coef = df_coef.sort_values(
        by="abs_coeficiente",
        ascending=False
    )


    df_coef.to_csv(
        PASTA_SAIDA
        / f"coeficientes_{target}.csv",
        index=False
    )


# ============================================================
# 9. RESUMO FINAL
# ============================================================

print("\n" + "=" * 70)
print("REGRESSÃO LINEAR FINALIZADA")
print("=" * 70)


print(
    "\nMétricas:"
)

print(
    df_metricas.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print(
    f"\nArquivos salvos em:"
)

print(
    PASTA_SAIDA
)


print(
    "\nModelos gerados:"
)

for target in TARGETS:

    print(
        f"  - linear_regression_{target}.pkl"
    )


print(
    "\nTreinamento da baseline concluído."
)
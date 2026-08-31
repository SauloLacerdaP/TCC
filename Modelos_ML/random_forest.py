import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import RandomizedSearchCV

import joblib


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DADOS = Path(
    r"C:\Repositorios\TCC\Output_dados\ml_preparado"
)

PASTA_SAIDA = Path(
    r"C:\Repositorios\TCC\Output_dados\resultados_random_forest"
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


RANDOM_STATE = 42


# ============================================================
# HIPERPARÂMETROS INICIAIS
# ============================================================

PARAMETROS_RF = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": RANDOM_STATE
}

PARAMETROS_RF_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 1.0],
    "bootstrap": [True, False]
}


# ============================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================

print("=" * 70)
print("RANDOM FOREST REGRESSION")
print("=" * 70)


X_train = pd.read_csv(
    PASTA_DADOS / "X_train.csv"
)

X_valid = pd.read_csv(
    PASTA_DADOS / "X_validation.csv"
)

X_test = pd.read_csv(
    PASTA_DADOS / "X_test.csv"
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
    f"Treino: {X_train.shape}"
)

print(
    f"Validação: {X_valid.shape}"
)

print(
    f"Teste: {X_test.shape}"
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
# 4. FUNÇÃO GRÁFICO REAL x PREVISTO
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
        f"Random Forest - {target}\n"
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
# 5. OTIMIZAÇÃO DOS HIPERPARÂMETROS
# ============================================================


def otimizar_hyperparametros(
    X_train,
    y_train_target,
    X_valid,
    y_valid_target
):
    """Busca automatizada de hiperparâmetros com validação cruzada."""

    modelo_base = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    busca = RandomizedSearchCV(
        estimator=modelo_base,
        param_distributions=PARAMETROS_RF_GRID,
        n_iter=8,
        scoring="neg_root_mean_squared_error",
        cv=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )

    busca.fit(
        X_train,
        y_train_target
    )

    return busca.best_params_


# ============================================================
# 6. TREINAMENTO
# ============================================================

resultados_metricas = []
resultados_parametros = []


for target in TARGETS:

    print("\n" + "=" * 70)

    print(
        f"TREINANDO RANDOM FOREST PARA {target}"
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
    # Otimização dos parâmetros
    # --------------------------------------------------------

    parametros_otimizados = otimizar_hyperparametros(
        X_train,
        y_train_target,
        X_valid,
        y_valid_target
    )

    parametros_modelo = {
        **PARAMETROS_RF,
        **parametros_otimizados
    }

    print(
        f"Melhores parâmetros para {target}:"
    )
    print(parametros_modelo)

    for nome_parametro, valor in parametros_modelo.items():
        resultados_parametros.append(
            {
                "target": target,
                "parametro": nome_parametro,
                "valor": valor
            }
        )


    # --------------------------------------------------------
    # Criar modelo
    # --------------------------------------------------------

    modelo = RandomForestRegressor(
        **parametros_modelo
    )


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
    # Registrar métricas
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
                "modelo": "RandomForest",
                "target": target,
                "conjunto": conjunto,
                "R2": metricas["R2"],
                "RMSE": metricas["RMSE"],
                "MSE": metricas["MSE"]
            }
        )


    # ========================================================
    # 6. SALVAR MODELO
    # ========================================================

    caminho_modelo = (
        PASTA_SAIDA
        / f"random_forest_{target}.pkl"
    )

    joblib.dump(
        modelo,
        caminho_modelo
    )


    # ========================================================
    # 7. SALVAR PREDIÇÕES
    # ========================================================

    for conjunto, y_real, y_pred in [
        (
            "treino",
            y_train_target,
            pred_train
        ),
        (
            "validacao",
            y_valid_target,
            pred_valid
        ),
        (
            "teste",
            y_test_target,
            pred_test
        )
    ]:

        df_pred = pd.DataFrame(
            {
                "real": y_real.values,
                "previsto": y_pred,
                "erro": (
                    y_pred
                    - y_real.values
                ),
                "erro_absoluto": np.abs(
                    y_pred
                    - y_real.values
                )
            }
        )

        df_pred.to_csv(
            PASTA_SAIDA
            / f"predicoes_{target}_{conjunto}.csv",
            index=False
        )


    # ========================================================
    # 8. GRÁFICOS REAL x PREVISTO
    # ========================================================

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


    # ========================================================
    # 9. FEATURE IMPORTANCE
    # ========================================================

    df_importancia = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": (
                modelo.feature_importances_
            )
        }
    )

    df_importancia = (
        df_importancia
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


    df_importancia.to_csv(
        PASTA_SAIDA
        / f"feature_importance_{target}.csv",
        index=False
    )


    # --------------------------------------------------------
    # Gráfico de importância
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    plt.barh(
        df_importancia[
            "feature"
        ][::-1],
        df_importancia[
            "importance"
        ][::-1]
    )

    plt.xlabel(
        "Importância"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Random Forest - Importância das features - {target}"
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_SAIDA
        / f"feature_importance_{target}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 10. SALVAR MÉTRICAS
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
    / "metricas_random_forest.csv",
    index=False
)


# ============================================================
# 11. SALVAR HIPERPARÂMETROS
# ============================================================

df_parametros = pd.DataFrame(
    resultados_parametros,
    columns=[
        "target",
        "parametro",
        "valor"
    ]
)


df_parametros.to_csv(
    PASTA_SAIDA
    / "hiperparametros_random_forest.csv",
    index=False
)


# ============================================================
# 12. RESUMO FINAL
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FINALIZADO")
print("=" * 70)


print("\nMétricas:")

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
        f"  - random_forest_{target}.pkl"
    )


print(
    "\nRandom Forest concluído."
)
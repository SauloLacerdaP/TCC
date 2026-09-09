"""
analisar_erro_vs_distancia.py

Analisa quantitativamente como o erro dos modelos XGBoost de CL, CD e CM
varia em função da distância geométrica CST ao conjunto de treinamento.

Objetivos:
1) carregar train.csv e test.csv;
2) carregar xgboost_CL.pkl, xgboost_CD.pkl e xgboost_CM.pkl;
3) calcular a distância CST padronizada de cada ponto do teste ao ponto
   de treino mais próximo;
4) prever CL, CD e CM para o teste;
5) calcular erros absolutos, percentuais, RMSE e R²;
6) avaliar os erros por faixa de distância;
7) avaliar os erros por faixa de alpha;
8) avaliar conjuntamente distância x alpha para CD;
9) gerar CSVs e gráficos úteis para definir uma região de confiança
   do surrogate antes de nova otimização.

IMPORTANTE:
- O StandardScaler é ajustado SOMENTE no conjunto de treinamento.
- A distância usa apenas Au0..Au6 e Al0..Al6.
- As métricas usam o conjunto de teste, que não participou do treino.
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ======================================================================
# CONFIGURAÇÕES
# ======================================================================

# A análise está dentro de Otimizador/analise/analise.py.
# A raiz lógica do projeto é o diretório pai de duas camadas,
# não o diretório do pacote Otimizador.
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "Output_dados" / "ml_preparado"
MODEL_DIR = ROOT / "Output_dados" / "resultados_xgboost"
OUT_DIR = ROOT / "Output_dados" / "analise_erro_distancia"

OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_CL = MODEL_DIR / "xgboost_CL.pkl"
MODEL_CD = MODEL_DIR / "xgboost_CD.pkl"
MODEL_CM = MODEL_DIR / "xgboost_CM.pkl"

CST_COLS = (
    [f"Au{i}" for i in range(7)]
    + [f"Al{i}" for i in range(7)]
)

# Faixas principais de distância.
DISTANCE_BINS = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    np.inf,
]

# Faixas de alpha para análise.
ALPHA_BINS = [
    0.0,
    3.0,
    6.0,
    9.0,
    12.000001,
]

# Para evitar MAPE explosivo quando a referência estiver muito próxima de zero.
PCT_EPS = 1e-10


# ======================================================================
# UTILIDADES
# ======================================================================

def verificar_arquivos():
    arquivos = [
        TRAIN_CSV,
        TEST_CSV,
        MODEL_CL,
        MODEL_CD,
        MODEL_CM,
    ]

    faltantes = [p for p in arquivos if not p.exists()]

    if faltantes:
        msg = "\n".join(str(p) for p in faltantes)
        raise FileNotFoundError(
            "Arquivos necessários não encontrados:\n" + msg
        )


def carregar_modelos():
    return (
        joblib.load(MODEL_CL),
        joblib.load(MODEL_CD),
        joblib.load(MODEL_CM),
    )


def resolver_features_modelo(model):
    """
    Retorna exatamente as features que o modelo espera.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "get_booster"):
        try:
            names = model.get_booster().feature_names
            if names:
                return list(names)
        except Exception:
            pass

    raise RuntimeError(
        "Não foi possível recuperar os nomes das features do modelo."
    )


def preparar_X(df: pd.DataFrame, model):
    features = resolver_features_modelo(model)

    faltantes = [c for c in features if c not in df.columns]
    if faltantes:
        raise KeyError(
            "Features exigidas pelo modelo não existem no dataframe:\n"
            + ", ".join(faltantes)
        )

    return df[features].copy()


def erro_percentual(pred, real):
    pred = np.asarray(pred, dtype=float)
    real = np.asarray(real, dtype=float)

    out = np.full(real.shape, np.nan, dtype=float)

    mask = np.isfinite(real) & (np.abs(real) > PCT_EPS)

    out[mask] = (
        100.0
        * np.abs(pred[mask] - real[mask])
        / np.abs(real[mask])
    )

    return out


def metricas_basicas(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_real) & np.isfinite(y_pred)

    y_real = y_real[mask]
    y_pred = y_pred[mask]

    if len(y_real) == 0:
        return {
            "n": 0,
            "R2": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE_pct": np.nan,
        }

    if len(y_real) >= 2:
        try:
            r2 = r2_score(y_real, y_pred)
        except Exception:
            r2 = np.nan
    else:
        r2 = np.nan

    mae = mean_absolute_error(y_real, y_pred)
    rmse = math.sqrt(mean_squared_error(y_real, y_pred))
    mape = np.nanmean(erro_percentual(y_pred, y_real))

    return {
        "n": len(y_real),
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_pct": mape,
    }


# ======================================================================
# DISTÂNCIA CST
# ======================================================================

def calcular_distancias_ao_treino(
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    """
    Calcula métricas de proximidade e densidade local no espaço CST
    padronizado.

    IMPORTANTE:
    O train.csv possui várias linhas da mesma geometria para diferentes
    condições aerodinâmicas (alpha/Re). Para a análise geométrica,
    essas repetições NÃO podem contar como vizinhos distintos.

    Portanto:
    1) cria-se internamente uma base de geometrias únicas usando os 14 CST;
    2) o StandardScaler é ajustado SOMENTE nessas geometrias únicas do treino;
    3) o KDTree é construído SOMENTE com geometrias únicas;
    4) d1, d3, d5, d10 e d20 passam a representar vizinhos geométricos
       distintos, e não linhas repetidas do mesmo aerofólio.

    Os arquivos train.csv e test.csv NÃO são modificados.

    Métricas geradas:
    - d1, d3, d5, d10, d20:
      distância até o k-ésimo vizinho geométrico distinto;
    - dmean3, dmean5, dmean10, dmean20:
      distância média aos k vizinhos geométricos distintos.
    """

    faltantes_train = [c for c in CST_COLS if c not in train.columns]
    faltantes_test = [c for c in CST_COLS if c not in test.columns]

    if faltantes_train or faltantes_test:
        raise KeyError(
            "Colunas CST faltantes.\n"
            f"Treino: {faltantes_train}\n"
            f"Teste: {faltantes_test}"
        )

    # --------------------------------------------------------------
    # GEOMETRIAS ÚNICAS DO TREINO
    # --------------------------------------------------------------
    # Mantemos a primeira linha de cada combinação CST apenas para poder
    # recuperar eventualmente o nome do perfil. A deduplicação é feita
    # exclusivamente pelos 14 coeficientes geométricos.
    colunas_manter = CST_COLS.copy()

    if "perfil" in train.columns:
        colunas_manter = ["perfil"] + colunas_manter

    train_geom = (
        train[colunas_manter]
        .drop_duplicates(subset=CST_COLS)
        .reset_index(drop=True)
    )

    if len(train_geom) < 20:
        raise ValueError(
            "O conjunto de treino possui menos de 20 geometrias CST únicas. "
            "Não é possível calcular d20 de forma adequada."
        )

    print(
        f"Geometrias únicas no treino para KDTree: "
        f"{len(train_geom)} (de {len(train)} linhas aerodinâmicas)"
    )

    # --------------------------------------------------------------
    # PADRONIZAÇÃO CST
    # --------------------------------------------------------------
    scaler = StandardScaler()

    X_train_geom = scaler.fit_transform(
        train_geom[CST_COLS].astype(float)
    )

    X_test = scaler.transform(
        test[CST_COLS].astype(float)
    )

    # --------------------------------------------------------------
    # KDTree COM GEOMETRIAS ÚNICAS
    # --------------------------------------------------------------
    tree = cKDTree(X_train_geom)

    k_max = min(20, len(train_geom))

    distances, indices = tree.query(
        X_test,
        k=k_max,
    )

    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    resultados = {}

    for k in [1, 3, 5, 10, 20]:
        k_eff = min(k, distances.shape[1])

        # Distância até o k-ésimo vizinho geométrico distinto.
        resultados[f"d{k}"] = distances[:, k_eff - 1]

        # Média das distâncias aos k vizinhos geométricos distintos.
        if k > 1:
            resultados[f"dmean{k}"] = np.mean(
                distances[:, :k_eff],
                axis=1,
            )

    # Índice refere-se agora a train_geom, não ao train.csv original.
    resultados["nearest_geom_index"] = indices[:, 0]

    # Nome do perfil geométrico mais próximo, quando disponível.
    if "perfil" in train_geom.columns:
        resultados["nearest_geom_profile"] = (
            train_geom.iloc[indices[:, 0]]["perfil"]
            .astype(str)
            .to_numpy()
        )

    return resultados, scaler, train_geom


# ======================================================================
# PREDIÇÕES E ERROS
# ======================================================================

def adicionar_predicoes(
    test: pd.DataFrame,
    model_cl,
    model_cd,
    model_cm,
):
    df = test.copy()

    Xcl = preparar_X(df, model_cl)
    Xcd = preparar_X(df, model_cd)
    Xcm = preparar_X(df, model_cm)

    df["CL_pred"] = np.asarray(
        model_cl.predict(Xcl),
        dtype=float,
    ).ravel()

    df["CD_pred"] = np.asarray(
        model_cd.predict(Xcd),
        dtype=float,
    ).ravel()

    df["CM_pred"] = np.asarray(
        model_cm.predict(Xcm),
        dtype=float,
    ).ravel()

    for alvo in ["CL", "CD", "CM"]:
        real = df[alvo].astype(float).to_numpy()
        pred = df[f"{alvo}_pred"].astype(float).to_numpy()

        df[f"erro_{alvo}_abs"] = np.abs(pred - real)
        df[f"erro_{alvo}_pct"] = erro_percentual(pred, real)

    df["CL_CD_real"] = np.where(
        df["CD"].astype(float) > 0,
        df["CL"].astype(float) / df["CD"].astype(float),
        np.nan,
    )

    df["CL_CD_pred"] = np.where(
        df["CD_pred"].astype(float) > 0,
        df["CL_pred"].astype(float) / df["CD_pred"].astype(float),
        np.nan,
    )

    df["erro_CL_CD_abs"] = np.abs(
        df["CL_CD_pred"] - df["CL_CD_real"]
    )

    df["erro_CL_CD_pct"] = erro_percentual(
        df["CL_CD_pred"].to_numpy(),
        df["CL_CD_real"].to_numpy(),
    )

    return df


# ======================================================================
# ANÁLISE POR FAIXA
# ======================================================================

def metricas_por_grupo(
    df: pd.DataFrame,
    group_col: str,
):
    rows = []

    for grupo, sub in df.groupby(group_col, observed=True):
        base = {
            group_col: str(grupo),
            "n": len(sub),
            "distancia_media": (
                float(sub["distance_to_train"].mean())
                if "distance_to_train" in sub.columns
                else np.nan
            ),
            "alpha_medio": (
                float(sub["alpha"].mean())
                if "alpha" in sub.columns
                else np.nan
            ),
        }

        for alvo in ["CL", "CD", "CM"]:
            m = metricas_basicas(
                sub[alvo],
                sub[f"{alvo}_pred"],
            )

            base[f"R2_{alvo}"] = m["R2"]
            base[f"MAE_{alvo}"] = m["MAE"]
            base[f"RMSE_{alvo}"] = m["RMSE"]
            base[f"MAPE_{alvo}_pct"] = m["MAPE_pct"]

        # CL/CD
        m_eff = metricas_basicas(
            sub["CL_CD_real"],
            sub["CL_CD_pred"],
        )

        base["R2_CL_CD"] = m_eff["R2"]
        base["MAE_CL_CD"] = m_eff["MAE"]
        base["RMSE_CL_CD"] = m_eff["RMSE"]
        base["MAPE_CL_CD_pct"] = m_eff["MAPE_pct"]

        rows.append(base)

    return pd.DataFrame(rows)


def analise_distancia_alpha_cd(df: pd.DataFrame):
    """
    Tabela cruzada distância x alpha especificamente para CD.
    """
    rows = []

    grouped = df.groupby(
        ["faixa_distancia", "faixa_alpha"],
        observed=True,
    )

    for (fd, fa), sub in grouped:
        m = metricas_basicas(
            sub["CD"],
            sub["CD_pred"],
        )

        rows.append({
            "faixa_distancia": str(fd),
            "faixa_alpha": str(fa),
            "n": len(sub),
            "distancia_media": float(
                sub["distance_to_train"].mean()
            ),
            "alpha_medio": float(sub["alpha"].mean()),
            "R2_CD": m["R2"],
            "MAE_CD": m["MAE"],
            "RMSE_CD": m["RMSE"],
            "MAPE_CD_pct": m["MAPE_pct"],
        })

    return pd.DataFrame(rows)


# ======================================================================
# MÉTRICAS GERAIS
# ======================================================================

def gerar_metricas_gerais(df: pd.DataFrame):
    rows = []

    for alvo in ["CL", "CD", "CM"]:
        m = metricas_basicas(
            df[alvo],
            df[f"{alvo}_pred"],
        )

        rows.append({
            "variavel": alvo,
            **m,
        })

    m_eff = metricas_basicas(
        df["CL_CD_real"],
        df["CL_CD_pred"],
    )

    rows.append({
        "variavel": "CL/CD",
        **m_eff,
    })

    return pd.DataFrame(rows)


# ======================================================================
# CORRELAÇÕES
# ======================================================================

def gerar_correlacoes(df: pd.DataFrame):
    """
    Compara cada métrica de distância/densidade com o erro absoluto
    dos modelos usando Pearson e Spearman.
    """
    rows = []

    metricas_distancia = [
        "d1",
        "d3",
        "d5",
        "d10",
        "d20",
        "dmean3",
        "dmean5",
        "dmean10",
        "dmean20",
    ]

    for metrica in metricas_distancia:
        for alvo in ["CL", "CD", "CM", "CL_CD"]:
            erro_abs = df[f"erro_{alvo}_abs"]

            pearson = df[metrica].corr(
                erro_abs,
                method="pearson",
            )

            spearman = df[metrica].corr(
                erro_abs,
                method="spearman",
            )

            rows.append({
                "metrica_distancia": metrica,
                "variavel": alvo,
                "pearson_distancia_vs_erro_abs": pearson,
                "spearman_distancia_vs_erro_abs": spearman,
            })

    return pd.DataFrame(rows)


def gerar_percentis_metricas_distancia(df: pd.DataFrame):
    metricas_distancia = [
        "d1",
        "d3",
        "d5",
        "d10",
        "d20",
        "dmean3",
        "dmean5",
        "dmean10",
        "dmean20",
    ]

    percentis = [50, 60, 70, 75, 80, 85, 90, 95, 97.5, 99]

    rows = []

    for metrica in metricas_distancia:
        for p in percentis:
            rows.append({
                "metrica_distancia": metrica,
                "percentil": p,
                "valor": float(
                    np.percentile(df[metrica], p)
                ),
            })

    return pd.DataFrame(rows)


# ======================================================================
# GRÁFICOS
# ======================================================================

def plot_scatter_distancia_erro(
    df: pd.DataFrame,
    alvo: str,
):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["distance_to_train"],
        df[f"erro_{alvo}_abs"],
        alpha=0.45,
    )

    plt.xlabel("Distância CST padronizada ao treino")
    plt.ylabel(f"Erro absoluto de {alvo}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"distancia_vs_erro_abs_{alvo}.png",
        dpi=200,
    )

    plt.close()


def plot_mape_por_distancia(
    metricas_dist: pd.DataFrame,
    alvo: str,
):
    plt.figure(figsize=(9, 5))

    plt.plot(
        metricas_dist["faixa_distancia"],
        metricas_dist[f"MAPE_{alvo}_pct"],
        marker="o",
    )

    plt.xlabel("Faixa de distância CST")
    plt.ylabel(f"MAPE {alvo} (%)")
    plt.xticks(rotation=30)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"mape_{alvo}_por_distancia.png",
        dpi=200,
    )

    plt.close()


def plot_r2_por_distancia(
    metricas_dist: pd.DataFrame,
    alvo: str,
):
    plt.figure(figsize=(9, 5))

    plt.plot(
        metricas_dist["faixa_distancia"],
        metricas_dist[f"R2_{alvo}"],
        marker="o",
    )

    plt.xlabel("Faixa de distância CST")
    plt.ylabel(f"R² {alvo}")
    plt.xticks(rotation=30)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"r2_{alvo}_por_distancia.png",
        dpi=200,
    )

    plt.close()


def plot_mape_cd_por_alpha(
    metricas_alpha: pd.DataFrame,
):
    plt.figure(figsize=(8, 5))

    plt.plot(
        metricas_alpha["faixa_alpha"],
        metricas_alpha["MAPE_CD_pct"],
        marker="o",
    )

    plt.xlabel("Faixa de α")
    plt.ylabel("MAPE CD (%)")
    plt.xticks(rotation=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "mape_CD_por_alpha.png",
        dpi=200,
    )

    plt.close()


def plot_cd_erro_distancia_alpha(
    df: pd.DataFrame,
):
    plt.figure(figsize=(8, 6))

    sc = plt.scatter(
        df["distance_to_train"],
        df["alpha"],
        c=df["erro_CD_pct"],
        alpha=0.65,
    )

    plt.xlabel("Distância CST padronizada ao treino")
    plt.ylabel("α (°)")
    plt.colorbar(sc, label="Erro percentual CD (%)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "mapa_distancia_alpha_erro_CD.png",
        dpi=220,
    )

    plt.close()


def plot_densidade_vs_erro_cd(
    df: pd.DataFrame,
    metrica: str,
):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df[metrica],
        df["erro_CD_abs"],
        alpha=0.45,
    )

    plt.xlabel(metrica)
    plt.ylabel("Erro absoluto de CD")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / f"{metrica}_vs_erro_abs_CD.png",
        dpi=200,
    )

    plt.close()


def plot_spearman_cd(
    correlacoes: pd.DataFrame,
):
    sub = (
        correlacoes[
            correlacoes["variavel"] == "CD"
        ]
        .copy()
        .sort_values(
            "spearman_distancia_vs_erro_abs",
            ascending=False,
        )
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        sub["metrica_distancia"],
        sub["spearman_distancia_vs_erro_abs"],
    )

    plt.xlabel("Métrica de distância/densidade")
    plt.ylabel("Spearman com erro absoluto de CD")
    plt.xticks(rotation=30)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "spearman_metricas_distancia_vs_erro_CD.png",
        dpi=220,
    )

    plt.close()


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 78)
    print("ANÁLISE DO ERRO DO SURROGATE EM FUNÇÃO DA DISTÂNCIA AO TREINO")
    print("=" * 78)

    verificar_arquivos()

    print("\nCarregando dados...")

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    print(f"Treino: {train.shape}")
    print(f"Teste:  {test.shape}")

    obrigatorias = [
        "CL", "CD", "CM", "alpha"
    ] + CST_COLS

    faltantes_train = [
        c for c in obrigatorias
        if c not in train.columns
    ]

    faltantes_test = [
        c for c in obrigatorias
        if c not in test.columns
    ]

    if faltantes_train or faltantes_test:
        raise KeyError(
            "Colunas obrigatórias ausentes.\n"
            f"Treino: {faltantes_train}\n"
            f"Teste: {faltantes_test}"
        )

    model_cl, model_cd, model_cm = carregar_modelos()

    print("\nCalculando métricas de distância/densidade CST ao treino...")

    dist_info, scaler, train_geom = calcular_distancias_ao_treino(
        train,
        test,
    )

    test_an = test.copy()

    # Compatibilidade com a análise anterior:
    test_an["distance_to_train"] = dist_info["d1"]

    for col in [
        "d1",
        "d3",
        "d5",
        "d10",
        "d20",
        "dmean3",
        "dmean5",
        "dmean10",
        "dmean20",
    ]:
        test_an[col] = dist_info[col]

    test_an["nearest_train_geom_index"] = (
        dist_info["nearest_geom_index"]
    )

    if "nearest_geom_profile" in dist_info:
        test_an["nearest_train_profile"] = (
            dist_info["nearest_geom_profile"]
        )

    # Salva também a base geométrica única usada na análise,
    # para auditoria e reprodutibilidade.
    train_geom.to_csv(
        OUT_DIR / "train_geometrias_unicas_cst.csv",
        index=False,
    )

    print("\nGerando predições...")

    test_an = adicionar_predicoes(
        test_an,
        model_cl,
        model_cd,
        model_cm,
    )

    # Faixas
    test_an["faixa_distancia"] = pd.cut(
        test_an["distance_to_train"],
        bins=DISTANCE_BINS,
        include_lowest=True,
    )

    test_an["faixa_alpha"] = pd.cut(
        test_an["alpha"],
        bins=ALPHA_BINS,
        include_lowest=True,
        right=False,
    )

    # --------------------------------------------------------------
    # Salva dados ponto a ponto
    # --------------------------------------------------------------

    test_an.to_csv(
        OUT_DIR / "teste_com_distancias_e_erros.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Métricas gerais
    # --------------------------------------------------------------

    metricas_gerais = gerar_metricas_gerais(test_an)

    metricas_gerais.to_csv(
        OUT_DIR / "metricas_gerais_teste.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Por distância
    # --------------------------------------------------------------

    metricas_dist = metricas_por_grupo(
        test_an,
        "faixa_distancia",
    )

    metricas_dist.to_csv(
        OUT_DIR / "metricas_por_faixa_distancia.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Por alpha
    # --------------------------------------------------------------

    metricas_alpha = metricas_por_grupo(
        test_an,
        "faixa_alpha",
    )

    metricas_alpha.to_csv(
        OUT_DIR / "metricas_por_faixa_alpha.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Distância x alpha para CD
    # --------------------------------------------------------------

    dist_alpha_cd = analise_distancia_alpha_cd(test_an)

    dist_alpha_cd.to_csv(
        OUT_DIR / "metricas_CD_distancia_x_alpha.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Correlações
    # --------------------------------------------------------------

    correlacoes = gerar_correlacoes(test_an)

    correlacoes.to_csv(
        OUT_DIR / "correlacoes_distancia_erro.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Percentis das métricas de distância/densidade
    # --------------------------------------------------------------

    percentis_metricas = gerar_percentis_metricas_distancia(
        test_an
    )

    percentis_metricas.to_csv(
        OUT_DIR / "percentis_metricas_distancia_densidade.csv",
        index=False,
    )

    # Mantém o arquivo antigo referente ao d1.
    dist_percentis = (
        percentis_metricas[
            percentis_metricas["metrica_distancia"] == "d1"
        ][["percentil", "valor"]]
        .rename(
            columns={"valor": "distance_value"}
        )
    )

    dist_percentis.to_csv(
        OUT_DIR / "percentis_distancia_teste.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Gráficos
    # --------------------------------------------------------------

    for alvo in ["CL", "CD", "CM", "CL_CD"]:
        plot_scatter_distancia_erro(
            test_an,
            alvo,
        )

        plot_mape_por_distancia(
            metricas_dist,
            alvo,
        )

        plot_r2_por_distancia(
            metricas_dist,
            alvo,
        )

    plot_mape_cd_por_alpha(metricas_alpha)
    plot_cd_erro_distancia_alpha(test_an)

    for metrica in [
        "d1",
        "d5",
        "d10",
        "d20",
        "dmean5",
        "dmean10",
        "dmean20",
    ]:
        plot_densidade_vs_erro_cd(
            test_an,
            metrica,
        )

    plot_spearman_cd(correlacoes)

    # --------------------------------------------------------------
    # Saída no terminal
    # --------------------------------------------------------------

    print("\n" + "=" * 78)
    print("MÉTRICAS GERAIS DO TESTE")
    print("=" * 78)
    print(metricas_gerais.to_string(index=False))

    print("\n" + "=" * 78)
    print("MÉTRICAS POR FAIXA DE DISTÂNCIA")
    print("=" * 78)
    print(metricas_dist.to_string(index=False))

    print("\n" + "=" * 78)
    print("MÉTRICAS POR FAIXA DE ALPHA")
    print("=" * 78)
    print(metricas_alpha.to_string(index=False))

    print("\n" + "=" * 78)
    print("CORRELAÇÕES DISTÂNCIA/DENSIDADE x ERRO ABSOLUTO")
    print("=" * 78)
    print(correlacoes.to_string(index=False))

    print("\n" + "=" * 78)
    print("PERCENTIS DE d1 NO TESTE")
    print("=" * 78)
    print(dist_percentis.to_string(index=False))

    print("\n" + "=" * 78)
    print("PERCENTIS DAS MÉTRICAS DE DISTÂNCIA/DENSIDADE")
    print("=" * 78)
    print(percentis_metricas.to_string(index=False))

    print("\n" + "=" * 78)
    print("ARQUIVOS GERADOS")
    print("=" * 78)

    for p in sorted(OUT_DIR.iterdir()):
        if p.is_file():
            print(f" - {p.name}")

    print(
        "\nUse principalmente:\n"
        "  metricas_por_faixa_distancia.csv\n"
        "  metricas_por_faixa_alpha.csv\n"
        "  metricas_CD_distancia_x_alpha.csv\n"
        "  correlacoes_distancia_erro.csv\n"
        "  percentis_metricas_distancia_densidade.csv\n"
        "  train_geometrias_unicas_cst.csv\n"
        "  mapa_distancia_alpha_erro_CD.png\n"
        "  spearman_metricas_distancia_vs_erro_CD.png\n"
        "\nCompare d1 com d3/d5/d10/d20 e principalmente com "
        "dmean3/dmean5/dmean10/dmean20 para verificar se densidade local "
        "explica o erro melhor do que apenas o vizinho mais próximo."
    )


if __name__ == "__main__":
    main()

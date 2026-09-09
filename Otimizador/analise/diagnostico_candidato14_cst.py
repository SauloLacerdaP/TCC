from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler


# ======================================================================
# CONFIGURAÇÕES
# ======================================================================

# Coloque este script em:
# C:\Repositorios\TCC\Otimizador\analise\diagnostico_candidato14_cst.py
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "Output_dados" / "ml_preparado"
OPT_DIR = ROOT / "Output_dados" / "otimizacao"

TRAIN_CSV = DATA_DIR / "train.csv"
CANDIDATES_CSV = OPT_DIR / "validacao_candidatos_xfoil.csv"

OUT_DIR = OPT_DIR / "diagnostico_candidato_14_cst"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_ID = 14
K_NEIGHBORS = 10

CST_COLS = (
    [f"Au{i}" for i in range(7)]
    + [f"Al{i}" for i in range(7)]
)


# ======================================================================
# UTILIDADES
# ======================================================================

def verificar_arquivos():
    faltantes = [
        p for p in [TRAIN_CSV, CANDIDATES_CSV]
        if not p.exists()
    ]

    if faltantes:
        raise FileNotFoundError(
            "Arquivos necessários não encontrados:\n"
            + "\n".join(str(p) for p in faltantes)
        )


def carregar_candidato(candidates: pd.DataFrame) -> pd.Series:
    possiveis_ids = [
        "candidate_id",
        "candidato_id",
        "id",
        "rank",
    ]

    id_col = None

    for col in possiveis_ids:
        if col in candidates.columns:
            id_col = col
            break

    if id_col is None:
        raise KeyError(
            "Não encontrei coluna de identificação do candidato."
        )

    sel = candidates[
        pd.to_numeric(candidates[id_col], errors="coerce")
        == CANDIDATE_ID
    ]

    if len(sel) != 1:
        raise ValueError(
            f"Esperava 1 candidato {CANDIDATE_ID}, "
            f"mas encontrei {len(sel)}."
        )

    cand = sel.iloc[0].copy()

    faltantes = [c for c in CST_COLS if c not in cand.index]

    if faltantes:
        raise KeyError(
            "O CSV de candidatos não contém todos os CST:\n"
            + ", ".join(faltantes)
        )

    return cand


def preparar_geometrias_unicas(train: pd.DataFrame):
    cols = CST_COLS.copy()

    if "perfil" in train.columns:
        cols = ["perfil"] + cols

    train_geom = (
        train[cols]
        .drop_duplicates(subset=CST_COLS)
        .reset_index(drop=True)
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_geom[CST_COLS].astype(float)
    )

    tree = cKDTree(X_train)

    return train_geom, scaler, tree, X_train


def encontrar_vizinhos(
    cand: pd.Series,
    train_geom: pd.DataFrame,
    scaler: StandardScaler,
    tree: cKDTree,
):
    x_cand_df = pd.DataFrame(
        [[float(cand[c]) for c in CST_COLS]],
        columns=CST_COLS,
    )

    x_cand_std = scaler.transform(x_cand_df)

    k = min(K_NEIGHBORS, len(train_geom))

    dist, idx = tree.query(
        x_cand_std,
        k=k,
    )

    dist = np.atleast_1d(dist).ravel()
    idx = np.atleast_1d(idx).ravel()

    viz = (
        train_geom.iloc[idx]
        .copy()
        .reset_index(drop=True)
    )

    viz.insert(
        0,
        "neighbor_rank",
        np.arange(1, len(viz) + 1),
    )

    viz.insert(
        1,
        "distance_cst_std",
        dist,
    )

    return viz, x_cand_std.ravel()


# ======================================================================
# DIAGNÓSTICO POR PARÂMETRO
# ======================================================================

def construir_tabela_parametros(
    cand: pd.Series,
    vizinhos: pd.DataFrame,
    train_geom: pd.DataFrame,
    scaler: StandardScaler,
):
    means = pd.Series(
        scaler.mean_,
        index=CST_COLS,
    )

    stds = pd.Series(
        scaler.scale_,
        index=CST_COLS,
    )

    rows = []

    for c in CST_COLS:
        cand_val = float(cand[c])

        viz_vals = vizinhos[c].astype(float)

        train_vals = train_geom[c].astype(float)

        viz_mean = float(viz_vals.mean())
        viz_median = float(viz_vals.median())
        viz_min = float(viz_vals.min())
        viz_max = float(viz_vals.max())
        viz_std = float(viz_vals.std(ddof=0))

        train_min = float(train_vals.min())
        train_max = float(train_vals.max())

        z_global = (
            (cand_val - means[c]) / stds[c]
            if stds[c] > 0 else np.nan
        )

        diff_vs_neighbor_mean = cand_val - viz_mean

        diff_norm_global = (
            diff_vs_neighbor_mean / stds[c]
            if stds[c] > 0 else np.nan
        )

        if viz_std > 1e-12:
            z_local = (cand_val - viz_mean) / viz_std
        else:
            z_local = np.nan

        range_train = train_max - train_min

        posicao_range = (
            (cand_val - train_min) / range_train
            if range_train > 1e-12
            else np.nan
        )

        fora_intervalo_vizinhos = (
            cand_val < viz_min
            or cand_val > viz_max
        )

        dist_intervalo_vizinhos = 0.0

        if cand_val < viz_min:
            dist_intervalo_vizinhos = viz_min - cand_val
        elif cand_val > viz_max:
            dist_intervalo_vizinhos = cand_val - viz_max

        dist_intervalo_norm = (
            dist_intervalo_vizinhos / stds[c]
            if stds[c] > 0 else np.nan
        )

        rows.append({
            "parametro": c,
            "candidato": cand_val,

            "viz_mean": viz_mean,
            "viz_median": viz_median,
            "viz_min": viz_min,
            "viz_max": viz_max,
            "viz_std": viz_std,

            "train_mean": means[c],
            "train_std": stds[c],
            "train_min": train_min,
            "train_max": train_max,

            "z_global_candidato": z_global,
            "z_local_vs_10_vizinhos": z_local,

            "diff_vs_media_vizinhos": diff_vs_neighbor_mean,
            "diff_norm_global_std": diff_norm_global,

            "fora_intervalo_10_vizinhos": fora_intervalo_vizinhos,
            "dist_fora_intervalo_vizinhos": dist_intervalo_vizinhos,
            "dist_fora_intervalo_norm": dist_intervalo_norm,

            "posicao_relativa_range_treino": posicao_range,
        })

    df = pd.DataFrame(rows)

    # Ranking de "extremidade" combinando:
    # - distância em desvios-padrão da média dos vizinhos;
    # - estar fora da faixa dos vizinhos;
    # - magnitude global do z-score.
    df["score_extremidade"] = (
        df["diff_norm_global_std"].abs()
        + 0.75 * df["dist_fora_intervalo_norm"].abs()
        + 0.25 * df["z_global_candidato"].abs()
    )

    df = df.sort_values(
        "score_extremidade",
        ascending=False,
    ).reset_index(drop=True)

    return df


# ======================================================================
# COMBINAÇÃO MULTIVARIADA
# ======================================================================

def diagnostico_multivariado(
    cand: pd.Series,
    vizinhos: pd.DataFrame,
    train_geom: pd.DataFrame,
    scaler: StandardScaler,
):
    X_train_std = scaler.transform(
        train_geom[CST_COLS].astype(float)
    )

    X_viz_std = scaler.transform(
        vizinhos[CST_COLS].astype(float)
    )

    x_cand_df = pd.DataFrame(
        [[float(cand[c]) for c in CST_COLS]],
        columns=CST_COLS,
    )

    x_cand_std = scaler.transform(
        x_cand_df
    ).ravel()

    centroid_viz = X_viz_std.mean(axis=0)

    dist_cand_centroid_viz = float(
        np.linalg.norm(
            x_cand_std - centroid_viz
        )
    )

    dists_viz_centroid = np.linalg.norm(
        X_viz_std - centroid_viz,
        axis=1,
    )

    max_dist_viz_centroid = float(
        dists_viz_centroid.max()
    )

    mean_dist_viz_centroid = float(
        dists_viz_centroid.mean()
    )

    # Distância de Mahalanobis local regularizada.
    Xc = X_viz_std - centroid_viz

    cov = np.cov(
        Xc,
        rowvar=False,
        ddof=1,
    )

    reg = 1e-6 * np.eye(cov.shape[0])

    cov_inv = np.linalg.pinv(
        cov + reg
    )

    delta = x_cand_std - centroid_viz

    mahal_local = float(
        np.sqrt(
            delta @ cov_inv @ delta
        )
    )

    # Quantos parâmetros do candidato estão fora do envelope dos 10 vizinhos.
    viz_min = vizinhos[CST_COLS].min()
    viz_max = vizinhos[CST_COLS].max()

    cand_vals = pd.Series(
        {
            c: float(cand[c])
            for c in CST_COLS
        }
    )

    fora = (
        (cand_vals < viz_min)
        | (cand_vals > viz_max)
    )

    n_fora = int(fora.sum())

    # Extremidade global:
    z_abs = np.abs(x_cand_std)

    return pd.DataFrame([{
        "distance_candidato_ao_centroide_10_vizinhos":
            dist_cand_centroid_viz,
        "media_dist_vizinhos_ao_centroide":
            mean_dist_viz_centroid,
        "max_dist_vizinhos_ao_centroide":
            max_dist_viz_centroid,
        "mahalanobis_local_regularizada":
            mahal_local,
        "n_parametros_fora_envelope_10_vizinhos":
            n_fora,
        "max_abs_z_global_candidato":
            float(z_abs.max()),
        "mean_abs_z_global_candidato":
            float(z_abs.mean()),
    }])


# ======================================================================
# GRÁFICOS
# ======================================================================

def plotar_heatmap_parametros(
    cand: pd.Series,
    vizinhos: pd.DataFrame,
    scaler: StandardScaler,
):
    df = vizinhos[CST_COLS].astype(float).copy()

    X_viz_std = scaler.transform(df)

    x_cand_df = pd.DataFrame(
        [[float(cand[c]) for c in CST_COLS]],
        columns=CST_COLS,
    )

    x_cand_std = scaler.transform(
        x_cand_df
    )

    matriz = np.vstack(
        [x_cand_std, X_viz_std]
    )

    labels = ["Candidato 14"] + [
        f"Viz {int(r)}"
        for r in vizinhos["neighbor_rank"]
    ]

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    im = ax.imshow(
        matriz,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(len(CST_COLS))
    )
    ax.set_xticklabels(
        CST_COLS,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(labels))
    )
    ax.set_yticklabels(labels)

    ax.set_title(
        "CST padronizado: candidato 14 vs 10 vizinhos"
    )

    fig.colorbar(
        im,
        ax=ax,
        label="z-score global",
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "heatmap_cst_candidato_vs_vizinhos.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plotar_ranking_extremidade(
    tabela: pd.DataFrame,
):
    df = tabela.sort_values(
        "score_extremidade",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        df["parametro"],
        df["score_extremidade"],
    )

    ax.set_xlabel(
        "Score de extremidade"
    )

    ax.set_ylabel(
        "Parâmetro CST"
    )

    ax.set_title(
        "Parâmetros CST mais extremos do candidato 14"
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "ranking_extremidade_cst.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plotar_zscore_global(
    tabela: pd.DataFrame,
):
    df = tabela.copy()

    df = df.sort_values(
        "z_global_candidato"
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        df["parametro"],
        df["z_global_candidato"],
    )

    ax.axvline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "z-score global do candidato"
    )

    ax.set_ylabel(
        "Parâmetro CST"
    )

    ax.set_title(
        "Posição do candidato 14 no espaço CST do treino"
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / "zscore_global_candidato14.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 94)
    print("DIAGNÓSTICO CST DO CANDIDATO 14 VS VIZINHOS GEOMÉTRICOS")
    print("=" * 94)

    verificar_arquivos()

    train = pd.read_csv(TRAIN_CSV)
    candidates = pd.read_csv(CANDIDATES_CSV)

    cand = carregar_candidato(
        candidates
    )

    train_geom, scaler, tree, _ = preparar_geometrias_unicas(
        train
    )

    vizinhos, x_cand_std = encontrar_vizinhos(
        cand,
        train_geom,
        scaler,
        tree,
    )

    tabela = construir_tabela_parametros(
        cand,
        vizinhos,
        train_geom,
        scaler,
    )

    multi = diagnostico_multivariado(
        cand,
        vizinhos,
        train_geom,
        scaler,
    )

    # Salvar
    vizinhos.to_csv(
        OUT_DIR / "vizinhos_cst_candidato14.csv",
        index=False,
    )

    tabela.to_csv(
        OUT_DIR / "diagnostico_parametro_a_parametro_cst.csv",
        index=False,
    )

    multi.to_csv(
        OUT_DIR / "diagnostico_multivariado_cst.csv",
        index=False,
    )

    plotar_heatmap_parametros(
        cand,
        vizinhos,
        scaler,
    )

    plotar_ranking_extremidade(
        tabela
    )

    plotar_zscore_global(
        tabela
    )

    print(
        f"\nGeometrias únicas de treino: "
        f"{len(train_geom)}"
    )

    print("\n" + "=" * 94)
    print("10 VIZINHOS MAIS PRÓXIMOS")
    print("=" * 94)

    cols_viz = [
        "neighbor_rank",
        "perfil",
        "distance_cst_std",
    ]

    cols_viz = [
        c for c in cols_viz
        if c in vizinhos.columns
    ]

    print(
        vizinhos[cols_viz]
        .to_string(index=False)
    )

    print("\n" + "=" * 94)
    print("PARÂMETROS MAIS EXTREMOS")
    print("=" * 94)

    cols = [
        "parametro",
        "candidato",
        "viz_mean",
        "viz_min",
        "viz_max",
        "z_global_candidato",
        "z_local_vs_10_vizinhos",
        "fora_intervalo_10_vizinhos",
        "dist_fora_intervalo_norm",
        "score_extremidade",
    ]

    print(
        tabela[cols]
        .head(14)
        .to_string(index=False)
    )

    print("\n" + "=" * 94)
    print("DIAGNÓSTICO MULTIVARIADO")
    print("=" * 94)

    print(
        multi.to_string(index=False)
    )

    print("\n" + "=" * 94)
    print("RESUMO")
    print("=" * 94)

    fora = tabela[
        tabela["fora_intervalo_10_vizinhos"]
    ]

    print(
        f"Parâmetros fora do envelope dos 10 vizinhos: "
        f"{len(fora)} de {len(CST_COLS)}"
    )

    if len(fora):
        print(
            "Parâmetros fora do envelope: "
            + ", ".join(
                fora["parametro"].tolist()
            )
        )

    top5 = tabela.head(5)

    print(
        "\nTop 5 parâmetros mais extremos:"
    )

    for _, row in top5.iterrows():
        print(
            f" - {row['parametro']}: "
            f"score={row['score_extremidade']:.3f}, "
            f"z_global={row['z_global_candidato']:.3f}, "
            f"fora_envelope={row['fora_intervalo_10_vizinhos']}"
        )

    print(
        f"\nArquivos salvos em:\n{OUT_DIR}"
    )

    print(
        "\nArquivos principais:\n"
        " - diagnostico_parametro_a_parametro_cst.csv\n"
        " - diagnostico_multivariado_cst.csv\n"
        " - vizinhos_cst_candidato14.csv\n"
        " - heatmap_cst_candidato_vs_vizinhos.png\n"
        " - ranking_extremidade_cst.png\n"
        " - zscore_global_candidato14.png"
    )


if __name__ == "__main__":
    main()

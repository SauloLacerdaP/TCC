from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler


# ======================================================================
# CONFIGURAÇÕES
# ======================================================================

# Este script foi pensado para ficar em:
# C:\Repositorios\TCC\Otimizador\analise\diagnostico_candidato14_vizinhos.py
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "Output_dados" / "ml_preparado"
MODEL_DIR = ROOT / "Output_dados" / "resultados_xgboost"
OPT_DIR = ROOT / "Output_dados" / "otimizacao"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_CL = MODEL_DIR / "xgboost_CL.pkl"
MODEL_CD = MODEL_DIR / "xgboost_CD.pkl"
MODEL_CM = MODEL_DIR / "xgboost_CM.pkl"

CANDIDATES_CSV = OPT_DIR / "validacao_candidatos_xfoil.csv"

OUT_DIR = OPT_DIR / "diagnostico_candidato_14_vizinhos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_ID = 14
RE_TARGET = 250_000.0
ALPHA_WINDOW = 1.0
RE_WINDOW = 25_000.0

K_NEIGHBORS = 10

CST_COLS = (
    [f"Au{i}" for i in range(7)]
    + [f"Al{i}" for i in range(7)]
)


# ======================================================================
# UTILIDADES
# ======================================================================

def verificar_arquivos():
    arquivos = [
        TRAIN_CSV,
        MODEL_CL,
        MODEL_CD,
        MODEL_CM,
        CANDIDATES_CSV,
    ]

    faltantes = [p for p in arquivos if not p.exists()]

    if faltantes:
        raise FileNotFoundError(
            "Arquivos necessários não encontrados:\n"
            + "\n".join(str(p) for p in faltantes)
        )


def resolver_features_modelo(model):
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
            "Features exigidas pelo modelo não estão disponíveis:\n"
            + ", ".join(faltantes)
        )

    return df[features].copy()


def prever(model, df: pd.DataFrame):
    X = preparar_X(df, model)
    return np.asarray(model.predict(X), dtype=float).ravel()


def carregar_candidato(candidates: pd.DataFrame) -> pd.Series:
    id_candidates = [
        "candidate_id",
        "candidato_id",
        "id",
        "rank",
    ]

    id_col = None

    for col in id_candidates:
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
            f"Esperava exatamente 1 candidato {CANDIDATE_ID}; "
            f"encontrei {len(sel)}."
        )

    cand = sel.iloc[0].copy()

    for c in CST_COLS:
        if c not in cand.index:
            raise KeyError(
                f"O CSV de candidatos não contém a coluna {c}."
            )

    return cand


def resolver_alpha_candidato(cand: pd.Series) -> float:
    for col in [
        "alpha_pred",
        "alpha",
        "alpha_xfoil",
        "alpha_opt",
        "alpha_deg",
    ]:
        if col in cand.index and pd.notna(cand[col]):
            return float(cand[col])

    raise KeyError(
        "Não foi possível encontrar alpha do candidato."
    )


# ======================================================================
# GEOMETRIAS ÚNICAS E VIZINHOS
# ======================================================================

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

    X_geom = scaler.fit_transform(
        train_geom[CST_COLS].astype(float)
    )

    tree = cKDTree(X_geom)

    return train_geom, scaler, tree


def encontrar_vizinhos(
    cand: pd.Series,
    train_geom: pd.DataFrame,
    scaler: StandardScaler,
    tree: cKDTree,
):
    x_cand = np.array(
        [[float(cand[c]) for c in CST_COLS]],
        dtype=float,
    )

    x_cand_std = scaler.transform(x_cand)

    k = min(K_NEIGHBORS, len(train_geom))

    dist, idx = tree.query(
        x_cand_std,
        k=k,
    )

    dist = np.atleast_1d(dist).ravel()
    idx = np.atleast_1d(idx).ravel()

    viz = train_geom.iloc[idx].copy().reset_index(drop=True)

    viz.insert(0, "neighbor_rank", np.arange(1, len(viz) + 1))
    viz.insert(1, "distance_cst_std", dist)

    return viz


# ======================================================================
# ANÁLISE AERODINÂMICA DOS VIZINHOS
# ======================================================================

def selecionar_ponto_real_proximo(
    train: pd.DataFrame,
    geom_row: pd.Series,
    alpha_target: float,
):
    mask_geom = np.ones(len(train), dtype=bool)

    for c in CST_COLS:
        mask_geom &= np.isclose(
            train[c].astype(float).to_numpy(),
            float(geom_row[c]),
            rtol=0.0,
            atol=1e-12,
        )

    sub = train.loc[mask_geom].copy()

    if len(sub) == 0:
        return None

    if "Re" in sub.columns:
        sub["delta_Re"] = np.abs(
            sub["Re"].astype(float) - RE_TARGET
        )
    else:
        sub["delta_Re"] = 0.0

    sub["delta_alpha"] = np.abs(
        sub["alpha"].astype(float) - alpha_target
    )

    # Prioriza primeiro a região local desejada.
    local = sub[
        (sub["delta_alpha"] <= ALPHA_WINDOW)
        & (sub["delta_Re"] <= RE_WINDOW)
    ].copy()

    base = local if len(local) else sub

    # Distância operacional normalizada.
    base["score_operacional"] = (
        base["delta_alpha"] / max(ALPHA_WINDOW, 1e-9)
        + base["delta_Re"] / max(RE_WINDOW, 1e-9)
    )

    return base.sort_values(
        ["score_operacional", "delta_alpha", "delta_Re"]
    ).iloc[0]


def construir_ponto_mesmas_condicoes(
    geom_row: pd.Series,
    train: pd.DataFrame,
    alpha_target: float,
):
    """
    Cria uma linha da geometria vizinha exatamente em Re/alpha do candidato.
    Para features adicionais como DeltaTE_upper/lower, usa valores
    geométricos representativos do próprio perfil do treino.
    """
    mask_geom = np.ones(len(train), dtype=bool)

    for c in CST_COLS:
        mask_geom &= np.isclose(
            train[c].astype(float).to_numpy(),
            float(geom_row[c]),
            rtol=0.0,
            atol=1e-12,
        )

    sub = train.loc[mask_geom].copy()

    if len(sub) == 0:
        raise RuntimeError(
            "Não encontrei linhas da geometria vizinha no train.csv."
        )

    # Usa uma linha real do perfil como base para features geométricas extras.
    ref = sub.iloc[0].copy()

    for c in CST_COLS:
        ref[c] = float(geom_row[c])

    if "Re" in ref.index:
        ref["Re"] = RE_TARGET

    if "alpha" in ref.index:
        ref["alpha"] = alpha_target

    return pd.DataFrame([ref])


def analisar_vizinhos(
    train: pd.DataFrame,
    vizinhos: pd.DataFrame,
    alpha_target: float,
    model_cl,
    model_cd,
    model_cm,
):
    rows = []

    for _, geom in vizinhos.iterrows():
        ponto_real = selecionar_ponto_real_proximo(
            train,
            geom,
            alpha_target,
        )

        ponto_mesmas_cond = construir_ponto_mesmas_condicoes(
            geom,
            train,
            alpha_target,
        )

        cl_pred_target = prever(model_cl, ponto_mesmas_cond)[0]
        cd_pred_target = prever(model_cd, ponto_mesmas_cond)[0]
        cm_pred_target = prever(model_cm, ponto_mesmas_cond)[0]

        row = {
            "neighbor_rank": int(geom["neighbor_rank"]),
            "distance_cst_std": float(geom["distance_cst_std"]),
            "perfil": (
                str(geom["perfil"])
                if "perfil" in geom.index
                else ""
            ),
            "alpha_target": alpha_target,
            "Re_target": RE_TARGET,
            "CL_pred_no_alpha_candidato": cl_pred_target,
            "CD_pred_no_alpha_candidato": cd_pred_target,
            "CM_pred_no_alpha_candidato": cm_pred_target,
            "CL_CD_pred_no_alpha_candidato": (
                cl_pred_target / cd_pred_target
                if cd_pred_target > 0 else np.nan
            ),
        }

        if ponto_real is not None:
            alpha_real = float(ponto_real["alpha"])
            re_real = (
                float(ponto_real["Re"])
                if "Re" in ponto_real.index
                else np.nan
            )

            cl_real = float(ponto_real["CL"])
            cd_real = float(ponto_real["CD"])
            cm_real = float(ponto_real["CM"])

            # Predição exatamente no ponto real existente do treino.
            real_df = pd.DataFrame([ponto_real])

            cl_pred_real = prever(model_cl, real_df)[0]
            cd_pred_real = prever(model_cd, real_df)[0]
            cm_pred_real = prever(model_cm, real_df)[0]

            row.update({
                "alpha_real_mais_proximo": alpha_real,
                "Re_real_mais_proximo": re_real,
                "delta_alpha": abs(alpha_real - alpha_target),
                "delta_Re": abs(re_real - RE_TARGET)
                if np.isfinite(re_real) else np.nan,

                "CL_real": cl_real,
                "CD_real": cd_real,
                "CM_real": cm_real,
                "CL_CD_real": (
                    cl_real / cd_real
                    if cd_real > 0 else np.nan
                ),

                "CL_pred_no_ponto_real": cl_pred_real,
                "CD_pred_no_ponto_real": cd_pred_real,
                "CM_pred_no_ponto_real": cm_pred_real,
                "CL_CD_pred_no_ponto_real": (
                    cl_pred_real / cd_pred_real
                    if cd_pred_real > 0 else np.nan
                ),

                "erro_CL_abs_ponto_real": abs(cl_pred_real - cl_real),
                "erro_CD_abs_ponto_real": abs(cd_pred_real - cd_real),
                "erro_CM_abs_ponto_real": abs(cm_pred_real - cm_real),

                "erro_CL_pct_ponto_real": (
                    100 * abs(cl_pred_real - cl_real) / abs(cl_real)
                    if abs(cl_real) > 1e-12 else np.nan
                ),
                "erro_CD_pct_ponto_real": (
                    100 * abs(cd_pred_real - cd_real) / abs(cd_real)
                    if abs(cd_real) > 1e-12 else np.nan
                ),
                "erro_CM_pct_ponto_real": (
                    100 * abs(cm_pred_real - cm_real) / abs(cm_real)
                    if abs(cm_real) > 1e-12 else np.nan
                ),
            })

        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================
# CANDIDATO
# ======================================================================

def resumo_candidato(cand: pd.Series, alpha_target: float):
    def get_first(names):
        for name in names:
            if name in cand.index and pd.notna(cand[name]):
                return float(cand[name])
        return np.nan

    cl_pred = get_first(["CL_pred", "CL"])
    cd_pred = get_first(["CD_pred", "CD"])
    cm_pred = get_first(["CM_pred", "CM"])
    eff_pred = get_first(["CL_CD_pred", "CL_CD"])

    cl_xf = get_first(["CL_xfoil"])
    cd_xf = get_first(["CD_xfoil"])
    cm_xf = get_first(["CM_xfoil"])
    eff_xf = get_first(["CL_CD_xfoil"])

    return pd.DataFrame([{
        "candidate_id": CANDIDATE_ID,
        "alpha": alpha_target,
        "Re": RE_TARGET,
        "CL_pred": cl_pred,
        "CD_pred": cd_pred,
        "CM_pred": cm_pred,
        "CL_CD_pred": eff_pred,
        "CL_xfoil": cl_xf,
        "CD_xfoil": cd_xf,
        "CM_xfoil": cm_xf,
        "CL_CD_xfoil": eff_xf,
    }])


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 88)
    print("DIAGNÓSTICO LOCAL DO CANDIDATO 14 VS VIZINHOS GEOMÉTRICOS DO TREINO")
    print("=" * 88)

    verificar_arquivos()

    train = pd.read_csv(TRAIN_CSV)
    candidates = pd.read_csv(CANDIDATES_CSV)

    model_cl = joblib.load(MODEL_CL)
    model_cd = joblib.load(MODEL_CD)
    model_cm = joblib.load(MODEL_CM)

    cand = carregar_candidato(candidates)
    alpha_target = resolver_alpha_candidato(cand)

    print(f"\nCandidato: {CANDIDATE_ID}")
    print(f"Re alvo: {RE_TARGET:.0f}")
    print(f"alpha alvo: {alpha_target:.6f}°")

    train_geom, scaler, tree = preparar_geometrias_unicas(train)

    print(
        f"Geometrias únicas de treino: "
        f"{len(train_geom)}"
    )

    vizinhos = encontrar_vizinhos(
        cand,
        train_geom,
        scaler,
        tree,
    )

    analise = analisar_vizinhos(
        train,
        vizinhos,
        alpha_target,
        model_cl,
        model_cd,
        model_cm,
    )

    resumo = resumo_candidato(
        cand,
        alpha_target,
    )

    # Salva
    vizinhos.to_csv(
        OUT_DIR / "vizinhos_geometricos_candidato14.csv",
        index=False,
    )

    analise.to_csv(
        OUT_DIR / "comparacao_aerodinamica_vizinhos_candidato14.csv",
        index=False,
    )

    resumo.to_csv(
        OUT_DIR / "resumo_candidato14.csv",
        index=False,
    )

    # Tabela curta para inspeção rápida
    cols_print = [
        "neighbor_rank",
        "perfil",
        "distance_cst_std",
        "alpha_real_mais_proximo",
        "Re_real_mais_proximo",
        "CL_real",
        "CD_real",
        "CM_real",
        "CL_CD_real",
        "CL_pred_no_ponto_real",
        "CD_pred_no_ponto_real",
        "CL_CD_pred_no_ponto_real",
        "erro_CD_pct_ponto_real",
        "CL_pred_no_alpha_candidato",
        "CD_pred_no_alpha_candidato",
        "CL_CD_pred_no_alpha_candidato",
    ]

    cols_print = [
        c for c in cols_print
        if c in analise.columns
    ]

    print("\n" + "=" * 88)
    print("CANDIDATO 14")
    print("=" * 88)
    print(resumo.to_string(index=False))

    print("\n" + "=" * 88)
    print("VIZINHOS GEOMÉTRICOS E COMPORTAMENTO AERODINÂMICO")
    print("=" * 88)
    print(
        analise[cols_print]
        .to_string(index=False)
    )

    # Estatísticas dos vizinhos
    print("\n" + "=" * 88)
    print("RESUMO DOS VIZINHOS")
    print("=" * 88)

    if "CD_real" in analise.columns:
        print(
            f"CD real dos vizinhos: "
            f"min={analise['CD_real'].min():.6f}, "
            f"mediana={analise['CD_real'].median():.6f}, "
            f"max={analise['CD_real'].max():.6f}"
        )

    if "CD_pred_no_ponto_real" in analise.columns:
        print(
            f"CD previsto nos pontos reais dos vizinhos: "
            f"min={analise['CD_pred_no_ponto_real'].min():.6f}, "
            f"mediana={analise['CD_pred_no_ponto_real'].median():.6f}, "
            f"max={analise['CD_pred_no_ponto_real'].max():.6f}"
        )

    print(
        f"\nArquivos salvos em:\n{OUT_DIR}"
    )

    print(
        "\nArquivos principais:\n"
        " - vizinhos_geometricos_candidato14.csv\n"
        " - comparacao_aerodinamica_vizinhos_candidato14.csv\n"
        " - resumo_candidato14.csv"
    )


if __name__ == "__main__":
    main()

"""
Validação completa do CANDIDATO 14 encontrado pelo otimizador.

Fluxo:
1) lê validacao_candidatos_xfoil.csv;
2) seleciona explicitamente o candidato 14;
3) reconstrói o .dat a partir dos CST Au0..Au6 / Al0..Al6;
4) carrega os modelos XGBoost de CL/CD/CM;
5) roda XFOIL exatamente no alpha do candidato 14;
6) roda a polar XFOIL completa de 0° a 12°;
7) compara XFOIL x XGBoost em CL, CD, CM e CL/CD;
8) identifica o máximo CL/CD real no XFOIL;
9) salva CSVs e gráficos.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURAÇÕES
# ======================================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "Output_dados" / "resultados_xgboost"
OPT_DIR = ROOT / "Output_dados" / "otimizacao"

CANDIDATE_ID = 14
VAL_DIR = OPT_DIR / f"validacao_xfoil_candidato_{CANDIDATE_ID:02d}"
VAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CL = MODEL_DIR / "xgboost_CL.pkl"
MODEL_CD = MODEL_DIR / "xgboost_CD.pkl"
MODEL_CM = MODEL_DIR / "xgboost_CM.pkl"

CANDIDATE_VALIDATION_CSV = OPT_DIR / "validacao_candidatos_xfoil.csv"

XFOIL_EXE = Path(r"C:\Repositorios\TCC\xfoil\XFOIL\xfoil.exe")

MACH = 0.1
RE_DEFAULT = 250_000.0

ALPHA_MIN = 0.0
ALPHA_MAX = 12.0
ALPHA_STEP = 0.25

XFOIL_ITER = 300
XFOIL_TIMEOUT = 120
ALPHA_MATCH_TOL = 0.03

CST_N1 = 0.5
CST_N2 = 1.0
N_GEOM_POINTS = 301

CST_UPPER = [f"Au{i}" for i in range(7)]
CST_LOWER = [f"Al{i}" for i in range(7)]

MODEL_FEATURES_DEFAULT = (
    CST_UPPER
    + CST_LOWER
    + ["DeltaTE_upper", "DeltaTE_lower", "Re", "alpha"]
)


# ======================================================================
# LEITURA DO CANDIDATO 14
# ======================================================================

def carregar_candidato_14():
    if not CANDIDATE_VALIDATION_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{CANDIDATE_VALIDATION_CSV}"
        )

    df = pd.read_csv(CANDIDATE_VALIDATION_CSV)

    if df.empty:
        raise RuntimeError("validacao_candidatos_xfoil.csv está vazio.")

    # Detecta uma coluna explícita de ID; se não existir, usa a ordem 1..N.
    possiveis_ids = [
        "candidate_id",
        "xfoil_candidate_id",
        "candidate",
        "candidato",
        "id",
    ]

    id_col = next((c for c in possiveis_ids if c in df.columns), None)

    if id_col is None:
        df = df.copy()
        df["candidate_id"] = np.arange(1, len(df) + 1)
        id_col = "candidate_id"

    ids = pd.to_numeric(df[id_col], errors="coerce")
    selecao = df.loc[ids == CANDIDATE_ID]

    if selecao.empty:
        raise RuntimeError(
            f"Candidato {CANDIDATE_ID} não encontrado em "
            f"{CANDIDATE_VALIDATION_CSV}\n"
            f"Coluna usada como ID: {id_col}\n"
            f"IDs disponíveis: {ids.dropna().tolist()}"
        )

    row = selecao.iloc[0].copy()
    row["xfoil_candidate_id"] = CANDIDATE_ID

    # Aliases comuns de nomes de colunas.
    aliases = {
        "Re": ["Re", "re", "reynolds"],
        "alpha": ["alpha", "alpha_pred", "alpha_deg", "alpha_opt"],
        "CL_pred": ["CL_pred", "cl_pred", "surrogate_CL"],
        "CD_pred": ["CD_pred", "cd_pred", "surrogate_CD"],
        "CM_pred": ["CM_pred", "cm_pred", "surrogate_CM"],
        "CL_CD_pred": ["CL_CD_pred", "surrogate_CL_CD", "CL_CD"],
        "CL_xfoil": ["CL_xfoil", "xfoil_CL"],
        "CD_xfoil": ["CD_xfoil", "xfoil_CD"],
        "CM_xfoil": ["CM_xfoil", "xfoil_CM"],
        "CL_CD_xfoil": ["CL_CD_xfoil", "xfoil_CL_CD"],
    }

    for destino, opcoes in aliases.items():
        if destino in row.index and pd.notna(row[destino]):
            continue
        for origem in opcoes:
            if origem in row.index and pd.notna(row[origem]):
                row[destino] = row[origem]
                break

    obrigatorias = CST_UPPER + CST_LOWER + ["Re", "alpha"]
    faltantes = [
        c for c in obrigatorias
        if c not in row.index or pd.isna(row[c])
    ]

    if faltantes:
        raise KeyError(
            "O candidato 14 não possui todas as colunas necessárias:\n"
            + ", ".join(faltantes)
            + "\n\nColunas disponíveis no CSV:\n"
            + ", ".join(map(str, df.columns))
        )

    return df, row


# ======================================================================
# MODELOS E FEATURES
# ======================================================================

def localizar_xfoil() -> Path:
    if not XFOIL_EXE.exists():
        raise FileNotFoundError(
            f"XFOIL não encontrado:\n{XFOIL_EXE}"
        )
    return XFOIL_EXE


def carregar_modelos():
    for p in [MODEL_CL, MODEL_CD, MODEL_CM]:
        if not p.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {p}")

    return (
        joblib.load(MODEL_CL),
        joblib.load(MODEL_CD),
        joblib.load(MODEL_CM),
    )


def resolver_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "get_booster"):
        try:
            names = model.get_booster().feature_names
            if names:
                return list(names)
        except Exception:
            pass

    return MODEL_FEATURES_DEFAULT.copy()


def montar_X(model, row: pd.Series, alpha: float, re_value: float):
    values = {
        **{f"Au{i}": float(row[f"Au{i}"]) for i in range(7)},
        **{f"Al{i}": float(row[f"Al{i}"]) for i in range(7)},
        "DeltaTE_upper": float(row.get("DeltaTE_upper", 0.0)),
        "DeltaTE_lower": float(row.get("DeltaTE_lower", 0.0)),
        "Re": float(re_value),
        "alpha": float(alpha),
    }

    features = resolver_feature_names(model)
    faltantes = [f for f in features if f not in values]

    if faltantes:
        raise KeyError(
            "Features ausentes para o modelo: " + ", ".join(faltantes)
        )

    return pd.DataFrame(
        [[values[f] for f in features]],
        columns=features,
    )


def prever_surrogate(model_cl, model_cd, model_cm, row, alpha, re_value):
    Xcl = montar_X(model_cl, row, alpha, re_value)
    Xcd = montar_X(model_cd, row, alpha, re_value)
    Xcm = montar_X(model_cm, row, alpha, re_value)

    cl = float(np.asarray(model_cl.predict(Xcl)).ravel()[0])
    cd = float(np.asarray(model_cd.predict(Xcd)).ravel()[0])
    cm = float(np.asarray(model_cm.predict(Xcm)).ravel()[0])

    eff = cl / cd if cd > 0 else np.nan
    return cl, cd, cm, eff


# ======================================================================
# RECONSTRUÇÃO CST -> DAT
# ======================================================================

def bernstein_matrix(n: int, x: np.ndarray) -> np.ndarray:
    return np.column_stack([
        math.comb(n, i) * (x ** i) * ((1.0 - x) ** (n - i))
        for i in range(n + 1)
    ])


def reconstruir_dat(row: pd.Series) -> Path:
    x = np.linspace(0.0, 1.0, N_GEOM_POINTS)

    class_function = (
        np.power(x, CST_N1)
        * np.power(1.0 - x, CST_N2)
    )

    au = np.array(
        [float(row[f"Au{i}"]) for i in range(7)],
        dtype=float,
    )
    al = np.array(
        [float(row[f"Al{i}"]) for i in range(7)],
        dtype=float,
    )

    B = bernstein_matrix(6, x)

    yu = class_function * (B @ au)
    yl = class_function * (B @ al)

    dte_u = float(row.get("DeltaTE_upper", 0.0))
    dte_l = float(row.get("DeltaTE_lower", 0.0))

    yu = yu + x * dte_u
    yl = yl + x * dte_l

    # XFOIL: TE -> LE pelo extradorso, LE -> TE pelo intradorso.
    x_dat = np.concatenate([x[::-1], x[1:]])
    y_dat = np.concatenate([yu[::-1], yl[1:]])

    path = VAL_DIR / f"candidato_{CANDIDATE_ID:02d}.dat"

    with path.open("w", encoding="utf-8") as f:
        f.write(f"Candidato_{CANDIDATE_ID}\n")
        for xi, yi in zip(x_dat, y_dat):
            f.write(f"{xi:.10f} {yi:.10f}\n")

    return path


# ======================================================================
# XFOIL
# ======================================================================

def escrever_script_xfoil(
    airfoil_filename: str,
    polar_filename: str,
    re_value: float,
    mach: float,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
):
    lines = [
        f"LOAD {airfoil_filename}",
        "",
        "PANE",
        "OPER",
        f"VISC {re_value:.0f}",
        f"MACH {mach:.6f}",
        f"ITER {XFOIL_ITER}",
        "PACC",
        polar_filename,
        "",
        f"ASEQ {alpha_start:.6f} {alpha_end:.6f} {alpha_step:.6f}",
        "PACC",
        "",
        "QUIT",
        "",
    ]
    return "\n".join(lines)


def escrever_script_xfoil_alpha_unico(
    airfoil_filename: str,
    polar_filename: str,
    re_value: float,
    mach: float,
    alpha: float,
):
    lines = [
        f"LOAD {airfoil_filename}",
        "",
        "PANE",
        "OPER",
        f"VISC {re_value:.0f}",
        f"MACH {mach:.6f}",
        f"ITER {XFOIL_ITER}",
        "PACC",
        polar_filename,
        "",
        f"ALFA {alpha:.8f}",
        "PACC",
        "",
        "QUIT",
        "",
    ]
    return "\n".join(lines)


def executar_xfoil(
    xfoil_exe: Path,
    commands: str,
    stdout_name: str,
    stderr_name: str,
):
    try:
        proc = subprocess.run(
            [str(xfoil_exe)],
            input=commands,
            text=True,
            capture_output=True,
            timeout=XFOIL_TIMEOUT,
            cwd=str(VAL_DIR),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"XFOIL excedeu o timeout de {XFOIL_TIMEOUT} s."
        ) from exc

    (VAL_DIR / stdout_name).write_text(
        proc.stdout or "",
        encoding="utf-8",
        errors="ignore",
    )
    (VAL_DIR / stderr_name).write_text(
        proc.stderr or "",
        encoding="utf-8",
        errors="ignore",
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"XFOIL terminou com código {proc.returncode}. "
            f"Consulte {stdout_name} e {stderr_name}."
        )

    return proc.stdout, proc.stderr


def ler_polar_xfoil(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Polar não gerada: {path}")

    rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            try:
                vals = [float(v) for v in parts[:7]]
            except ValueError:
                continue

            rows.append(vals)

    if not rows:
        raise RuntimeError(
            f"Nenhuma linha numérica válida encontrada em {path}"
        )

    max_cols = max(len(r) for r in rows)

    columns = [
        "alpha", "CL", "CD", "CDp", "CM",
        "Top_Xtr", "Bot_Xtr"
    ][:max_cols]

    df = pd.DataFrame(
        [r[:max_cols] for r in rows],
        columns=columns,
    )

    df = (
        df
        .drop_duplicates(subset=["alpha"], keep="last")
        .sort_values("alpha")
        .reset_index(drop=True)
    )

    df["CL_CD"] = np.where(
        df["CD"] > 0,
        df["CL"] / df["CD"],
        np.nan,
    )

    return df


def rodar_alpha_unico_xfoil(
    xfoil_exe: Path,
    airfoil_dat: Path,
    re_value: float,
    alpha: float,
):
    local_airfoil = VAL_DIR / "aerofolio_validacao.dat"
    shutil.copy2(airfoil_dat, local_airfoil)

    polar_path = VAL_DIR / "xfoil_alpha_candidato14.txt"

    if polar_path.exists():
        polar_path.unlink()

    commands = escrever_script_xfoil_alpha_unico(
        airfoil_filename=local_airfoil.name,
        polar_filename=polar_path.name,
        re_value=re_value,
        mach=MACH,
        alpha=alpha,
    )

    (VAL_DIR / "comandos_xfoil_alpha_candidato14.txt").write_text(
        commands,
        encoding="utf-8",
    )

    executar_xfoil(
        xfoil_exe,
        commands,
        "xfoil_stdout_alpha_candidato14.txt",
        "xfoil_stderr_alpha_candidato14.txt",
    )

    return ler_polar_xfoil(polar_path)


def rodar_polar_xfoil(
    xfoil_exe: Path,
    airfoil_dat: Path,
    re_value: float,
):
    local_airfoil = VAL_DIR / "aerofolio_validacao.dat"
    shutil.copy2(airfoil_dat, local_airfoil)

    polar_path = VAL_DIR / "polar_xfoil_0a12.txt"

    if polar_path.exists():
        polar_path.unlink()

    commands = escrever_script_xfoil(
        airfoil_filename=local_airfoil.name,
        polar_filename=polar_path.name,
        re_value=re_value,
        mach=MACH,
        alpha_start=ALPHA_MIN,
        alpha_end=ALPHA_MAX,
        alpha_step=ALPHA_STEP,
    )

    (VAL_DIR / "comandos_xfoil_polar.txt").write_text(
        commands,
        encoding="utf-8",
    )

    executar_xfoil(
        xfoil_exe,
        commands,
        "xfoil_stdout_polar.txt",
        "xfoil_stderr_polar.txt",
    )

    return ler_polar_xfoil(polar_path)


# ======================================================================
# COMPARAÇÃO
# ======================================================================

def erro_percentual(pred: float, ref: float):
    if not np.isfinite(ref) or abs(ref) < 1e-12:
        return np.nan
    return 100.0 * abs(pred - ref) / abs(ref)


def comparar_ponto(
    row_opt,
    xfoil_point,
    model_cl,
    model_cd,
    model_cm,
    re_value,
    alpha_opt,
):
    cl_ml, cd_ml, cm_ml, eff_ml = prever_surrogate(
        model_cl,
        model_cd,
        model_cm,
        row_opt,
        alpha_opt,
        re_value,
    )

    cl_xf = float(xfoil_point["CL"])
    cd_xf = float(xfoil_point["CD"])
    cm_xf = float(xfoil_point["CM"])
    eff_xf = float(xfoil_point["CL_CD"])

    rows = []

    for nome, ml, xf in [
        ("CL", cl_ml, cl_xf),
        ("CD", cd_ml, cd_xf),
        ("CM", cm_ml, cm_xf),
        ("CL/CD", eff_ml, eff_xf),
    ]:
        rows.append({
            "variavel": nome,
            "XGBoost": ml,
            "XFOIL": xf,
            "erro_absoluto": abs(ml - xf),
            "erro_percentual": erro_percentual(ml, xf),
        })

    return pd.DataFrame(rows)


def gerar_curva_surrogate(
    row_opt,
    model_cl,
    model_cd,
    model_cm,
    re_value,
    alphas,
):
    rows = []

    for alpha in alphas:
        cl, cd, cm, eff = prever_surrogate(
            model_cl,
            model_cd,
            model_cm,
            row_opt,
            float(alpha),
            re_value,
        )

        rows.append({
            "alpha": float(alpha),
            "CL_ML": cl,
            "CD_ML": cd,
            "CM_ML": cm,
            "CL_CD_ML": eff,
        })

    return pd.DataFrame(rows)


# ======================================================================
# GRÁFICOS
# ======================================================================

def salvar_grafico(
    df,
    x_col,
    y_xfoil,
    y_ml,
    ylabel,
    filename,
):
    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df[y_xfoil], marker="o", label="XFOIL")
    plt.plot(df[x_col], df[y_ml], marker="x", label="XGBoost")
    plt.xlabel("Ângulo de ataque α (°)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(VAL_DIR / filename, dpi=200)
    plt.close()


def gerar_graficos(df):
    salvar_grafico(
        df, "alpha", "CL", "CL_ML",
        "CL", "comparacao_CL.png",
    )

    salvar_grafico(
        df, "alpha", "CD", "CD_ML",
        "CD", "comparacao_CD.png",
    )

    salvar_grafico(
        df, "alpha", "CM", "CM_ML",
        "CM", "comparacao_CM.png",
    )

    salvar_grafico(
        df, "alpha", "CL_CD", "CL_CD_ML",
        "CL/CD", "comparacao_CL_CD.png",
    )

    plt.figure(figsize=(8, 5))
    plt.plot(df["CD"], df["CL"], marker="o", label="XFOIL")
    plt.plot(df["CD_ML"], df["CL_ML"], marker="x", label="XGBoost")
    plt.xlabel("CD")
    plt.ylabel("CL")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(VAL_DIR / "polar_CL_CD.png", dpi=200)
    plt.close()


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 78)
    print(f"VALIDAÇÃO DO CANDIDATO {CANDIDATE_ID}: XGBOOST x XFOIL")
    print("=" * 78)

    xfoil_exe = localizar_xfoil()
    print(f"\nXFOIL:\n{xfoil_exe}")

    _, row = carregar_candidato_14()

    re_value = float(row.get("Re", RE_DEFAULT))
    alpha_opt = float(row["alpha"])

    print(f"\nCandidato selecionado: {CANDIDATE_ID}")
    print(f"Re = {re_value:.0f}")
    print(f"Mach = {MACH}")
    print(f"alpha surrogate = {alpha_opt:.6f}°")

    if "CL_CD_pred" in row.index and pd.notna(row["CL_CD_pred"]):
        print(f"CL/CD surrogate salvo = {float(row['CL_CD_pred']):.8f}")

    if "CL_CD_xfoil" in row.index and pd.notna(row["CL_CD_xfoil"]):
        print(f"CL/CD XFOIL salvo = {float(row['CL_CD_xfoil']):.8f}")

    dat_path = reconstruir_dat(row)
    print(f"\nDAT reconstruído:\n{dat_path}")

    model_cl, model_cd, model_cm = carregar_modelos()

    # ------------------------------------------------------------------
    # 1) Revalidação no alpha do candidato 14
    # ------------------------------------------------------------------

    print("\n" + "-" * 78)
    print("1) XFOIL NO ALPHA DO CANDIDATO 14")
    print("-" * 78)

    df_alpha = rodar_alpha_unico_xfoil(
        xfoil_exe,
        dat_path,
        re_value,
        alpha_opt,
    )

    idx = (df_alpha["alpha"] - alpha_opt).abs().idxmin()
    xfoil_point = df_alpha.loc[idx]

    if abs(float(xfoil_point["alpha"]) - alpha_opt) > ALPHA_MATCH_TOL:
        warnings.warn(
            "O alpha registrado pelo XFOIL difere do solicitado "
            "mais que ALPHA_MATCH_TOL."
        )

    comparison = comparar_ponto(
        row,
        xfoil_point,
        model_cl,
        model_cd,
        model_cm,
        re_value,
        alpha_opt,
    )

    print("\nComparação no alpha do candidato 14:")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        VAL_DIR / "comparacao_alpha_candidato14.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 2) Polar completa
    # ------------------------------------------------------------------

    print("\n" + "-" * 78)
    print("2) POLAR XFOIL COMPLETA DE 0° A 12°")
    print("-" * 78)

    xfoil_polar = rodar_polar_xfoil(
        xfoil_exe,
        dat_path,
        re_value,
    )

    print(
        f"Pontos XFOIL convergidos: {len(xfoil_polar)} "
        f"de aproximadamente "
        f"{int(round((ALPHA_MAX - ALPHA_MIN) / ALPHA_STEP)) + 1}"
    )

    alphas = xfoil_polar["alpha"].to_numpy(dtype=float)

    surrogate_polar = gerar_curva_surrogate(
        row,
        model_cl,
        model_cd,
        model_cm,
        re_value,
        alphas,
    )

    combined = pd.merge(
        xfoil_polar,
        surrogate_polar,
        on="alpha",
        how="left",
    )

    combined["erro_CL_pct"] = np.where(
        np.abs(combined["CL"]) > 1e-12,
        100.0 * np.abs(combined["CL_ML"] - combined["CL"]) / np.abs(combined["CL"]),
        np.nan,
    )

    combined["erro_CD_pct"] = np.where(
        np.abs(combined["CD"]) > 1e-12,
        100.0 * np.abs(combined["CD_ML"] - combined["CD"]) / np.abs(combined["CD"]),
        np.nan,
    )

    combined["erro_CM_pct"] = np.where(
        np.abs(combined["CM"]) > 1e-12,
        100.0 * np.abs(combined["CM_ML"] - combined["CM"]) / np.abs(combined["CM"]),
        np.nan,
    )

    combined["erro_CL_CD_pct"] = np.where(
        np.abs(combined["CL_CD"]) > 1e-12,
        100.0 * np.abs(combined["CL_CD_ML"] - combined["CL_CD"]) / np.abs(combined["CL_CD"]),
        np.nan,
    )

    xfoil_polar.to_csv(
        VAL_DIR / "polar_xfoil.csv",
        index=False,
    )

    surrogate_polar.to_csv(
        VAL_DIR / "polar_xgboost.csv",
        index=False,
    )

    combined.to_csv(
        VAL_DIR / "polar_comparacao_xfoil_xgboost.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 3) Máximos
    # ------------------------------------------------------------------

    xf_valid = combined[np.isfinite(combined["CL_CD"])].copy()
    ml_valid = combined[np.isfinite(combined["CL_CD_ML"])].copy()

    best_xf = None
    best_ml = None

    if not xf_valid.empty:
        best_xf = combined.loc[xf_valid["CL_CD"].idxmax()]

    if not ml_valid.empty:
        best_ml = combined.loc[ml_valid["CL_CD_ML"].idxmax()]

    print("\n" + "=" * 78)
    print("RESUMO")
    print("=" * 78)

    print("\nNo alpha escolhido pelo surrogate:")
    print(comparison.to_string(index=False))

    if best_xf is not None:
        print("\nMáximo CL/CD segundo XFOIL:")
        print(f"alpha = {best_xf['alpha']:.4f}°")
        print(f"CL = {best_xf['CL']:.8f}")
        print(f"CD = {best_xf['CD']:.8f}")
        print(f"CM = {best_xf['CM']:.8f}")
        print(f"CL/CD = {best_xf['CL_CD']:.8f}")

    if best_ml is not None:
        print("\nMáximo CL/CD do XGBoost nos mesmos pontos:")
        print(f"alpha = {best_ml['alpha']:.4f}°")
        print(f"CL = {best_ml['CL_ML']:.8f}")
        print(f"CD = {best_ml['CD_ML']:.8f}")
        print(f"CM = {best_ml['CM_ML']:.8f}")
        print(f"CL/CD = {best_ml['CL_CD_ML']:.8f}")

    metrics = {
        "candidate_id": CANDIDATE_ID,
        "n_pontos_xfoil": len(combined),
        "alpha_surrogate": alpha_opt,
        "MAE_CL": float(np.nanmean(np.abs(combined["CL_ML"] - combined["CL"]))),
        "MAE_CD": float(np.nanmean(np.abs(combined["CD_ML"] - combined["CD"]))),
        "MAE_CM": float(np.nanmean(np.abs(combined["CM_ML"] - combined["CM"]))),
        "MAE_CL_CD": float(np.nanmean(np.abs(combined["CL_CD_ML"] - combined["CL_CD"]))),
        "MAPE_CL_pct": float(np.nanmean(combined["erro_CL_pct"])),
        "MAPE_CD_pct": float(np.nanmean(combined["erro_CD_pct"])),
        "MAPE_CM_pct": float(np.nanmean(combined["erro_CM_pct"])),
        "MAPE_CL_CD_pct": float(np.nanmean(combined["erro_CL_CD_pct"])),
    }

    if best_xf is not None:
        metrics["alpha_max_CL_CD_XFOIL"] = float(best_xf["alpha"])
        metrics["max_CL_CD_XFOIL"] = float(best_xf["CL_CD"])

    if best_ml is not None:
        metrics["alpha_max_CL_CD_XGBoost"] = float(best_ml["alpha"])
        metrics["max_CL_CD_XGBoost"] = float(best_ml["CL_CD_ML"])

    pd.DataFrame([metrics]).to_csv(
        VAL_DIR / "metricas_validacao.csv",
        index=False,
    )

    gerar_graficos(combined)

    print("\nArquivos salvos em:")
    print(VAL_DIR)


if __name__ == "__main__":
    main()

"""

plot_curvas_otimizado.py

Gera gráficos dos aerofólios otimizados usando polares XFOIL já existentes:

1) geometria do perfil otimizado para CL;

2) geometria do perfil otimizado para CD;

3) geometria do perfil otimizado para CL/CD;

4) curva CL x alpha do perfil otimizado para CL;

5) curva CD x alpha do perfil otimizado para CD.

Nos gráficos aerodinâmicos, o ponto em alpha = 6 graus é destacado

com um círculo.

Condições XFOIL:

    Reynolds = 250000

    Mach     = 0.1

    alpha    = 0 a 12 graus

    passo    = 0.25 grau

"""

from __future__ import annotations

import json

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

MODEL_CL = MODEL_DIR / "xgboost_CL.pkl"

MODEL_CD = MODEL_DIR / "xgboost_CD.pkl"

RESULT_CL_JSON = (

    ROOT / "Output_dados" / "otimizacao_CL" / "resultado_otimizacao.json"

)

RESULT_CD_JSON = (

    ROOT / "Output_dados" / "otimizacao_CD" / "resultado_otimizacao.json"

)

DAT_CL = (

    ROOT

    / "Output_dados"

    / "otimizacao_CL"

    / "aerofolio_otimizado.dat"

)

DAT_CD = (

    ROOT

    / "Output_dados"

    / "otimizacao_CD"

    / "aerofolio_otimizado.dat"

)

DAT_CL_CD = (

    ROOT

    / "Output_dados"

    / "otimizacao_CL_CD"

    / "aerofolio_otimizado.dat"

)

OUTPUT_DIR = ROOT / "Output_dados" / "plots_otimizados"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Polares XFOIL já calculadas pelos otimizadores.
POLAR_CL = OUTPUT_DIR / "xfoil_otimizado_CL" / "polar.txt"
POLAR_CD = OUTPUT_DIR / "xfoil_otimizado_CD" / "polar.txt"
POLAR_CL_CD = (
    ROOT
    / "Output_dados"
    / "otimizacao_CL_CD"
    / "validacao_candidatos_xfoil"
    / "candidato_005"
    / "polar.txt"
)

REYNOLDS = 250_000.0

MACH = 0.1

ALPHA_MIN = 0.0

ALPHA_MAX = 12.0

ALPHA_STEP = 0.25

ALPHA_DESTAQUE = 6.0



# ======================================================================

# LEITURA / PLOT DA GEOMETRIA

# ======================================================================

def ler_dat(path: Path) -> np.ndarray:

    """

    Lê um arquivo .dat de aerofólio.

    Ignora cabeçalhos/textos e mantém apenas linhas com duas colunas

    numéricas x, y.

    """

    if not path.exists():

        raise FileNotFoundError(f"Arquivo .dat não encontrado:\n{path}")

    pontos = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            parts = line.strip().replace(",", ".").split()

            if len(parts) < 2:

                continue

            try:

                x = float(parts[0])

                y = float(parts[1])

            except ValueError:

                continue

            if np.isfinite(x) and np.isfinite(y):

                pontos.append((x, y))

    if len(pontos) < 10:

        raise RuntimeError(

            f"Poucos pontos válidos encontrados no arquivo:\n{path}"

        )

    return np.asarray(pontos, dtype=float)



def separar_superficies(coords: np.ndarray):

    """

    Separa extradorso e intradorso usando o ponto de menor x

    como bordo de ataque.

    """

    i_le = int(np.argmin(coords[:, 0]))

    extradorso = coords[: i_le + 1]

    intradorso = coords[i_le:]

    return extradorso, intradorso



def resolver_features_modelo(model):

    """Recupera os nomes das features do modelo XGBoost."""

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

        "Não foi possível recuperar os nomes das features do modelo XGBoost."

    )



def carregar_cst_json(path: Path) -> dict:

    """Carrega o JSON com os coeficientes CST do perfil otimizado."""

    if not path.exists():

        raise FileNotFoundError(f"Arquivo de resultado não encontrado:\n{path}")

    with path.open("r", encoding="utf-8") as f:

        data = json.load(f)

    return data



def gerar_previsao_xgboost(

    model_path: Path,

    json_path: Path,

    target: str,

) -> pd.DataFrame:

    """

    Gera a curva de previsão do XGBoost para o mesmo intervalo de alpha

    usado no XFOIL e usa os coeficientes CST salvos no JSON do otimizador.

    """

    if not model_path.exists():

        raise FileNotFoundError(f"Modelo XGBoost não encontrado:\n{model_path}")

    if not json_path.exists():

        raise FileNotFoundError(f"JSON da otimização não encontrado:\n{json_path}")

    model = joblib.load(model_path)

    features = resolver_features_modelo(model)

    cst = carregar_cst_json(json_path)

    # Coeficientes CST e parâmetros físicos do modelo.

    row_template = {}

    for feature in features:

        if feature in cst:

            row_template[feature] = float(cst[feature])

        elif feature == "Re":

            row_template[feature] = REYNOLDS

        elif feature == "alpha":

            row_template[feature] = ALPHA_MIN

        elif feature == "DeltaTE_upper":

            row_template[feature] = float(cst.get("DeltaTE_upper", 0.0))

        elif feature == "DeltaTE_lower":

            row_template[feature] = float(cst.get("DeltaTE_lower", 0.0))

    # Gera pontos em alpha de mesma grade do XFOIL.

    alphas = np.arange(

        ALPHA_MIN,

        ALPHA_MAX + 0.5 * ALPHA_STEP,

        ALPHA_STEP,

        dtype=float,

    )

    rows = []

    for alpha in alphas:

        row = row_template.copy()

        row["alpha"] = float(alpha)

        row["Re"] = REYNOLDS

        rows.append(row)

    X_pred = pd.DataFrame(rows, columns=features)

    y_pred = np.asarray(model.predict(X_pred), dtype=float).ravel()

    return pd.DataFrame(

        {

            "alpha": alphas,

            target: y_pred,

        }

    )



def plotar_geometria(dat_path: Path, titulo: str, output_path: Path):

    coords = ler_dat(dat_path)

    extradorso, intradorso = separar_superficies(coords)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(

        extradorso[:, 0],

        extradorso[:, 1],

        linewidth=2.2,

        label="Extradorso",

    )

    ax.plot(

        intradorso[:, 0],

        intradorso[:, 1],

        linewidth=2.2,

        label="Intradorso",

    )

    ax.set_title(titulo)

    ax.set_xlabel("x/c")

    ax.set_ylabel("y/c")

    ax.grid(True, alpha=0.30)

    ax.legend()

    ax.set_xlim(0.0, 1.0)

    # Mantém a geometria visualmente proporcional.

    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)



# ======================================================================

# XFOIL

# ======================================================================

def ler_polar_xfoil(path: Path) -> pd.DataFrame:

    """Lê o arquivo de polar criado pelo XFOIL."""

    if not path.exists():

        raise RuntimeError(f"Polar XFOIL não encontrada:\n{path}")

    rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:

                continue

            try:

                values = [float(v) for v in parts[:5]]

            except ValueError:

                continue

            rows.append(values)

    if not rows:

        raise RuntimeError(

            f"Nenhum ponto numérico válido encontrado na polar:\n{path}"

        )

    df = pd.DataFrame(

        rows,

        columns=["alpha", "CL", "CD", "CDp", "CM"],

    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(

        subset=["alpha", "CL", "CD", "CM"]

    )

    df = (

        df.sort_values("alpha")

        .drop_duplicates(subset=["alpha"], keep="last")

        .reset_index(drop=True)

    )

    df["CL_CD"] = df["CL"] / df["CD"]

    return df



def executar_xfoil(

    dat_path: Path,

    nome: str,

) -> pd.DataFrame:

    """

    Executa polar completa no XFOIL para um aerofólio.

    """

    if not XFOIL_EXE.exists():

        raise FileNotFoundError(

            "XFOIL não encontrado.\n"

            f"Caminho esperado:\n{XFOIL_EXE}"

        )

    if not dat_path.exists():

        raise FileNotFoundError(

            f"Aerofólio não encontrado:\n{dat_path}"

        )

    work_dir = OUTPUT_DIR / f"xfoil_{nome}"

    work_dir.mkdir(parents=True, exist_ok=True)

    local_dat = work_dir / "aerofolio.dat"

    polar_path = work_dir / "polar.txt"

    shutil.copy2(dat_path, local_dat)

    if polar_path.exists():

        polar_path.unlink()

    commands = "\n".join(

        [

            f"LOAD {local_dat.name}",

            "",

            "PANE",

            "OPER",

            f"VISC {REYNOLDS:.0f}",

            f"MACH {MACH:.6f}",

            f"ITER {XFOIL_ITER}",

            "PACC",

            polar_path.name,

            "",

            (

                f"ASEQ {ALPHA_MIN:.6f} "

                f"{ALPHA_MAX:.6f} "

                f"{ALPHA_STEP:.6f}"

            ),

            "PACC",

            "",

            "QUIT",

            "",

        ]

    )

    command_path = work_dir / "comandos_xfoil.txt"

    command_path.write_text(commands, encoding="utf-8")

    print(f"\nExecutando XFOIL para: {nome}")

    print(f"Arquivo: {dat_path}")

    print(

        f"Re={REYNOLDS:.0f} | Mach={MACH:.2f} | "

        f"alpha={ALPHA_MIN:.2f} a {ALPHA_MAX:.2f} "

        f"(passo {ALPHA_STEP:.2f})"

    )

    try:

        proc = subprocess.run(

            [str(XFOIL_EXE)],

            input=commands,

            text=True,

            capture_output=True,

            timeout=XFOIL_TIMEOUT,

            cwd=str(work_dir),

        )

    except subprocess.TimeoutExpired as exc:

        raise RuntimeError(

            f"XFOIL excedeu o timeout de {XFOIL_TIMEOUT} s "

            f"para {nome}."

        ) from exc

    (work_dir / "stdout.txt").write_text(

        proc.stdout or "",

        encoding="utf-8",

        errors="ignore",

    )

    (work_dir / "stderr.txt").write_text(

        proc.stderr or "",

        encoding="utf-8",

        errors="ignore",

    )

    if proc.returncode != 0:

        raise RuntimeError(

            f"XFOIL retornou código {proc.returncode} para {nome}. "

            f"Consulte:\n{work_dir / 'stdout.txt'}"

        )

    df = ler_polar_xfoil(polar_path)

    csv_path = OUTPUT_DIR / f"polar_xfoil_{nome}.csv"

    df.to_csv(csv_path, index=False)

    print(f"Pontos XFOIL válidos: {len(df)}")

    print(f"Polar salva em: {csv_path}")

    return df



# ======================================================================

# PLOTS AERODINÂMICOS

# ======================================================================

def obter_ponto_alpha(

    df: pd.DataFrame,

    alpha_target: float,

):

    """

    Retorna a linha exatamente em alpha_target.

    Se houver pequena diferença numérica, aceita tolerância de 1e-6.

    """

    mask = np.isclose(

        df["alpha"].to_numpy(dtype=float),

        alpha_target,

        atol=1e-6,

        rtol=0.0,

    )

    if not np.any(mask):

        raise RuntimeError(

            f"A polar não possui ponto convergido em alpha={alpha_target:.2f}°."

        )

    return df.loc[mask].iloc[-1]



def plotar_cl_alpha(

    df: pd.DataFrame,

    output_path: Path,

    titulo: str,

    previsao_xgb: pd.DataFrame | None = None,

):

    ponto = obter_ponto_alpha(df, ALPHA_DESTAQUE)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(

        df["alpha"],

        df["CL"],

        linewidth=2.2,

        label="XFOIL",

    )

    if previsao_xgb is not None:

        ax.plot(

            previsao_xgb["alpha"],

            previsao_xgb["CL"],

            linewidth=2.0,

            linestyle=":",

            color="tab:orange",

            label="XGBoost (surrogate)",

        )

    # Círculo no ponto alpha = 6°.

    ax.scatter(

        [ponto["alpha"]],

        [ponto["CL"]],

        s=130,

        facecolors="none",

        edgecolors="black",

        linewidths=2.0,

        zorder=5,

        label=(

            rf"$\alpha={ALPHA_DESTAQUE:.0f}^\circ$"

            rf"  |  $C_L={ponto['CL']:.4f}$"

        ),

    )

    ax.set_title(

        f"{titulo}" "\n"

        f"Re={REYNOLDS:.0f}, Mach={MACH:.1f}"

    )

    ax.set_xlabel(r"$\alpha$ (°)")

    ax.set_ylabel(r"$C_L$")

    ax.grid(True, alpha=0.30)

    ax.legend()

    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return ponto



def plotar_cd_alpha(

    df: pd.DataFrame,

    output_path: Path,

    titulo: str,

    previsao_xgb: pd.DataFrame | None = None,

):

    ponto = obter_ponto_alpha(df, ALPHA_DESTAQUE)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(

        df["alpha"],

        df["CD"],

        linewidth=2.2,

        label="XFOIL",

    )

    if previsao_xgb is not None:

        ax.plot(

            previsao_xgb["alpha"],

            previsao_xgb["CD"],

            linewidth=2.0,

            linestyle=":",

            color="tab:orange",

            label="XGBoost (surrogate)",

        )

    # Círculo no ponto alpha = 6°.

    ax.scatter(

        [ponto["alpha"]],

        [ponto["CD"]],

        s=130,

        facecolors="none",

        edgecolors="black",

        linewidths=2.0,

        zorder=5,

        label=(

            rf"$\alpha={ALPHA_DESTAQUE:.0f}^\circ$"

            rf"  |  $C_D={ponto['CD']:.5f}$"

        ),

    )

    ax.set_title(

        f"{titulo}" "\n"

        f"Re={REYNOLDS:.0f}, Mach={MACH:.1f}"

    )

    ax.set_xlabel(r"$\alpha$ (°)")

    ax.set_ylabel(r"$C_D$")

    ax.grid(True, alpha=0.30)

    ax.legend()

    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return ponto



# ======================================================================

# MAIN

# ======================================================================

def main():

    print("=" * 78)

    print("PLOTS DOS AEROFÓLIOS OTIMIZADOS + POLARES XFOIL EXISTENTES")

    print("=" * 78)

    print(f"Perfil CL: {DAT_CL}")

    print(f"Perfil CD: {DAT_CD}")

    print(f"Perfil CL/CD: {DAT_CL_CD}")

    print(f"Saída: {OUTPUT_DIR}")

    # --------------------------------------------------------------

    # 1) Plots geométricos

    # --------------------------------------------------------------

    plotar_geometria(

        DAT_CL,

        "Perfil otimizado — CL (Re=250k, alpha=6°)",

        OUTPUT_DIR / "perfil_otimizado_CL.png",

    )

    plotar_geometria(

        DAT_CD,

        "Perfil otimizado — CD (Re=250k, alpha=6°)",

        OUTPUT_DIR / "perfil_otimizado_CD.png",

    )

    plotar_geometria(

        DAT_CL_CD,

        "Perfil otimizado — CL/CD (Re=250k, alpha=6°)",

        OUTPUT_DIR / "perfil_otimizado_CL_CD.png",

    )

    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # 2) Leitura das polares XFOIL já existentes
    # --------------------------------------------------------------

    print("\nLendo polares XFOIL já existentes:")
    print(f"Polar CL: {POLAR_CL}")
    print(f"Polar CD: {POLAR_CD}")
    print(f"Polar CL/CD: {POLAR_CL_CD}")

    polar_cl = ler_polar_xfoil(POLAR_CL)
    polar_cd = ler_polar_xfoil(POLAR_CD)
    polar_cl_cd = ler_polar_xfoil(POLAR_CL_CD)

    # Mantém os CSVs consolidados na pasta de plots.
    polar_cl.to_csv(
        OUTPUT_DIR / "polar_xfoil_otimizado_CL.csv",
        index=False,
    )
    polar_cd.to_csv(
        OUTPUT_DIR / "polar_xfoil_otimizado_CD.csv",
        index=False,
    )
    polar_cl_cd.to_csv(
        OUTPUT_DIR / "polar_xfoil_otimizado_CL_CD.csv",
        index=False,
    )

    # 3) Linhas de previsão XGBoost

    # --------------------------------------------------------------

    previsao_cl = gerar_previsao_xgboost(

        MODEL_CL,

        RESULT_CL_JSON,

        target="CL",

    )

    previsao_cd = gerar_previsao_xgboost(

        MODEL_CD,

        RESULT_CD_JSON,

        target="CD",

    )

    previsao_cl_cl_cd = gerar_previsao_xgboost(

        MODEL_CL,

        ROOT / "Output_dados" / "otimizacao_CL_CD" / "resultado_otimizacao.json",

        target="CL",

    )

    previsao_cd_cl_cd = gerar_previsao_xgboost(

        MODEL_CD,

        ROOT / "Output_dados" / "otimizacao_CL_CD" / "resultado_otimizacao.json",

        target="CD",

    )

    # --------------------------------------------------------------

    # 4) Novos plots

    # --------------------------------------------------------------

    ponto_cl = plotar_cl_alpha(

        polar_cl,

        OUTPUT_DIR / "curva_CL_alpha_otimizado_CL.png",

        r"Perfil otimizado em $C_L$ — curva $C_L$ × $\alpha$",

        previsao_xgb=previsao_cl,

    )

    plotar_cl_alpha(

        polar_cd[polar_cd["alpha"] <= 7.5].copy(),

        OUTPUT_DIR / "curva_CL_alpha_otimizado_CD.png",

        r"Perfil otimizado em $C_L$ para CD — curva $C_L$ × $\alpha$",

        previsao_xgb=previsao_cl[previsao_cl["alpha"] <= 7.5].copy(),

    )

    ponto_cd = plotar_cd_alpha(

        polar_cd[polar_cd["alpha"] <= 7.5].copy(),

        OUTPUT_DIR / "curva_CD_alpha_otimizado_CD.png",

        r"Perfil otimizado em $C_D$ — curva $C_D$ × $\alpha$",

        previsao_xgb=previsao_cd[previsao_cd["alpha"] <= 7.5].copy(),

    )

    plotar_cd_alpha(

        polar_cl,

        OUTPUT_DIR / "curva_CD_alpha_otimizado_CL.png",

        r"Perfil otimizado em $C_D$ para CL — curva $C_D$ × $\alpha$",

        previsao_xgb=previsao_cd,

    )

    ponto_cl_cd = obter_ponto_alpha(polar_cl_cd, ALPHA_DESTAQUE)

    plotar_cl_alpha(

        polar_cl_cd[polar_cl_cd["alpha"] <= 9.0].copy(),

        OUTPUT_DIR / "curva_CL_alpha_otimizado_CL_CD.png",

        r"Perfil otimizado em $C_L$ para CL/CD — curva $C_L$ × $\alpha$",

        previsao_xgb=previsao_cl_cl_cd[previsao_cl_cl_cd["alpha"] <= 9.0].copy(),

    )

    plotar_cd_alpha(

        polar_cl_cd[polar_cl_cd["alpha"] <= 9.0].copy(),

        OUTPUT_DIR / "curva_CD_alpha_otimizado_CL_CD.png",

        r"Perfil otimizado em $C_D$ para CL/CD — curva $C_D$ × $\alpha$",

        previsao_xgb=previsao_cd_cl_cd[previsao_cd_cl_cd["alpha"] <= 9.0].copy(),

    )

    # --------------------------------------------------------------

    resumo = pd.DataFrame(

        [

            {

                "perfil": "otimizado_CL",

                "Re": REYNOLDS,

                "Mach": MACH,

                "alpha": float(ponto_cl["alpha"]),

                "CL": float(ponto_cl["CL"]),

                "CD": float(ponto_cl["CD"]),

                "CM": float(ponto_cl["CM"]),

                "CL_CD": float(ponto_cl["CL_CD"]),

            },

            {

                "perfil": "otimizado_CD",

                "Re": REYNOLDS,

                "Mach": MACH,

                "alpha": float(ponto_cd["alpha"]),

                "CL": float(ponto_cd["CL"]),

                "CD": float(ponto_cd["CD"]),

                "CM": float(ponto_cd["CM"]),

                "CL_CD": float(ponto_cd["CL_CD"]),

            },

        ]

    )

    resumo_path = OUTPUT_DIR / "resumo_alpha6_otimizados.csv"

    resumo.to_csv(resumo_path, index=False)

    print("\n" + "=" * 78)

    print("RESULTADOS EM ALPHA = 6°")

    print("=" * 78)

    print(

        f"Perfil otimizado CL: "

        f"CL={ponto_cl['CL']:.6f} | "

        f"CD={ponto_cl['CD']:.6f} | "

        f"CM={ponto_cl['CM']:.6f} | "

        f"CL/CD={ponto_cl['CL_CD']:.3f}"

    )

    print(

        f"Perfil otimizado CD: "

        f"CL={ponto_cd['CL']:.6f} | "

        f"CD={ponto_cd['CD']:.6f} | "

        f"CM={ponto_cd['CM']:.6f} | "

        f"CL/CD={ponto_cd['CL_CD']:.3f}"

    )

    print("\nArquivos gerados:")

    print(" - perfil_otimizado_CL.png")

    print(" - perfil_otimizado_CD.png")

    print(" - perfil_otimizado_CL_CD.png")

    print(" - curva_CL_alpha_otimizado_CL.png")

    print(" - curva_CL_alpha_otimizado_CD.png")

    print(" - curva_CD_alpha_otimizado_CD.png")

    print(" - curva_CD_alpha_otimizado_CL.png")

    print(" - curva_CL_alpha_otimizado_CL_CD.png")

    print(" - curva_CD_alpha_otimizado_CL_CD.png")

    print(" - polar_xfoil_otimizado_CL.csv")

    print(" - polar_xfoil_otimizado_CD.csv")

    print(" - polar_xfoil_otimizado_CL_CD.csv")

    print(" - resumo_alpha6_otimizados.csv")

    print(f"\nPasta: {OUTPUT_DIR}")



if __name__ == "__main__":

    main()
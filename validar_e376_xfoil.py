"""
validar_e376_xfoil.py

Validação isolada do perfil e376 no XFOIL.

Objetivo:
- Rodar novamente o perfil e376 em Re = 250.000, Mach = 0,1;
- Avaliar alpha de 0° a 12° com passo de 0,25°;
- Salvar a nova polar;
- Comparar automaticamente os pontos inteiros com o database_ml.csv,
  quando esse arquivo estiver disponível no repositório.

Saídas:
Output_dados/validacao_e376/
    e376_polar_xfoil_025.csv
    e376_comparacao_database.csv
    comandos_xfoil.txt
    stdout.txt
    stderr.txt
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# CONFIGURAÇÕES
# ======================================================================

ROOT = Path(__file__).resolve().parent

XFOIL_EXE = ROOT / "xfoil" / "XFOIL" / "xfoil.exe"

AIRFOIL_DIR = ROOT / "lednicer_to_selig" / "Airfoils_Selig"

# Tenta localizar o perfil com tolerância a maiúsculas/minúsculas.
PROFILE_NAME = "e376"

RE = 250_000.0
MACH = 0.1
XFOIL_ITER = 500
TIMEOUT = 300

ALPHA_MIN = 0.0
ALPHA_MAX = 12.0
ALPHA_STEP = 0.25

OUTPUT_DIR = ROOT / "Output_dados" / "validacao_e376"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORK_DIR = OUTPUT_DIR / "_xfoil_temp"
WORK_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_CANDIDATES = [
    ROOT / "Output_dados" / "database_ml.csv",
    ROOT / "database_ml.csv",
]


# ======================================================================
# UTILIDADES
# ======================================================================

def localizar_perfil() -> Path:
    candidatos = list(AIRFOIL_DIR.glob("*.dat"))

    for p in candidatos:
        if p.stem.lower() == PROFILE_NAME.lower():
            return p

    raise FileNotFoundError(
        f"Não encontrei {PROFILE_NAME}.dat em:\n{AIRFOIL_DIR}"
    )


def localizar_database() -> Path | None:
    for p in DATABASE_CANDIDATES:
        if p.exists():
            return p
    return None


def ler_polar_xfoil(path: Path) -> pd.DataFrame:
    rows = []

    if not path.exists():
        raise FileNotFoundError(f"Polar não encontrada: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()

            # Formato típico:
            # alpha CL CD CDp CM Top_Xtr Bot_Xtr
            if len(parts) < 5:
                continue

            try:
                vals = [float(v) for v in parts[:7]]
            except ValueError:
                continue

            while len(vals) < 7:
                vals.append(np.nan)

            rows.append(vals[:7])

    if not rows:
        raise RuntimeError("Nenhum ponto numérico válido encontrado na polar XFOIL.")

    df = pd.DataFrame(
        rows,
        columns=[
            "alpha",
            "CL",
            "CD",
            "CDp",
            "CM",
            "Top_Xtr",
            "Bot_Xtr",
        ],
    )

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["alpha", "CL", "CD", "CM"])
        .sort_values("alpha")
        .drop_duplicates(subset=["alpha"], keep="last")
        .reset_index(drop=True)
    )

    df = df[df["CD"] > 0].copy()
    df["CL_CD"] = df["CL"] / df["CD"]

    return df


def executar_xfoil(profile_path: Path) -> pd.DataFrame:
    local_dat = WORK_DIR / f"{PROFILE_NAME}.dat"
    polar_path = WORK_DIR / "polar_e376.txt"

    shutil.copy2(profile_path, local_dat)

    if polar_path.exists():
        polar_path.unlink()

    alphas = np.arange(
        ALPHA_MIN,
        ALPHA_MAX + 0.5 * ALPHA_STEP,
        ALPHA_STEP,
        dtype=float,
    )

    # ALFA individual em vez de ASEQ:
    # facilita identificar falhas pontuais de convergência.
    alfa_commands = [f"ALFA {a:.6f}" for a in alphas]

    commands = "\n".join([
        f"LOAD {local_dat.name}",
        "",
        "PANE",
        "OPER",
        f"VISC {RE:.0f}",
        f"MACH {MACH:.6f}",
        f"ITER {XFOIL_ITER}",
        "PACC",
        polar_path.name,
        "",
        *alfa_commands,
        "PACC",
        "",
        "QUIT",
        "",
    ])

    (OUTPUT_DIR / "comandos_xfoil.txt").write_text(
        commands,
        encoding="utf-8",
    )

    proc = subprocess.run(
        [str(XFOIL_EXE)],
        input=commands,
        text=True,
        capture_output=True,
        timeout=TIMEOUT,
        cwd=str(WORK_DIR),
    )

    (OUTPUT_DIR / "stdout.txt").write_text(
        proc.stdout or "",
        encoding="utf-8",
        errors="ignore",
    )

    (OUTPUT_DIR / "stderr.txt").write_text(
        proc.stderr or "",
        encoding="utf-8",
        errors="ignore",
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"XFOIL retornou código {proc.returncode}. "
            f"Consulte stdout.txt e stderr.txt em {OUTPUT_DIR}"
        )

    return ler_polar_xfoil(polar_path)


def comparar_com_database(df_novo: pd.DataFrame) -> pd.DataFrame | None:
    db_path = localizar_database()

    if db_path is None:
        print("\nAVISO: database_ml.csv não encontrado.")
        print("A nova polar será salva normalmente, mas sem comparação automática.")
        return None

    db = pd.read_csv(db_path)

    perfil_col = next(
        (c for c in ("perfil", "Perfil", "profile", "Profile") if c in db.columns),
        None,
    )
    re_col = next(
        (c for c in ("Re", "RE", "re", "Reynolds") if c in db.columns),
        None,
    )
    alpha_col = next(
        (c for c in ("alpha", "Alpha", "ALPHA") if c in db.columns),
        None,
    )

    obrigatorias = {"CL", "CD", "CM"}
    if (
        perfil_col is None
        or re_col is None
        or alpha_col is None
        or not obrigatorias.issubset(db.columns)
    ):
        print("\nAVISO: database_ml.csv encontrado, mas colunas esperadas não foram identificadas.")
        return None

    db[perfil_col] = db[perfil_col].astype(str)

    ref = db[
        (db[perfil_col].str.lower() == PROFILE_NAME.lower())
        & np.isclose(pd.to_numeric(db[re_col], errors="coerce"), RE)
    ].copy()

    if ref.empty:
        print(f"\nAVISO: não encontrei {PROFILE_NAME} em Re={RE:.0f} no database.")
        return None

    ref[alpha_col] = pd.to_numeric(ref[alpha_col], errors="coerce")
    ref["CL"] = pd.to_numeric(ref["CL"], errors="coerce")
    ref["CD"] = pd.to_numeric(ref["CD"], errors="coerce")
    ref["CM"] = pd.to_numeric(ref["CM"], errors="coerce")

    ref = ref.dropna(subset=[alpha_col, "CL", "CD", "CM"])

    # O banco original usa alpha inteiro.
    ref = ref[
        (ref[alpha_col] >= ALPHA_MIN)
        & (ref[alpha_col] <= ALPHA_MAX)
    ].copy()

    novo = df_novo.copy()
    novo["alpha_round"] = novo["alpha"].round(8)

    ref = ref.rename(
        columns={
            alpha_col: "alpha",
            "CL": "CL_database",
            "CD": "CD_database",
            "CM": "CM_database",
        }
    )
    ref["alpha_round"] = ref["alpha"].round(8)

    merged = ref[
        ["alpha_round", "alpha", "CL_database", "CD_database", "CM_database"]
    ].merge(
        novo[
            ["alpha_round", "CL", "CD", "CM", "CL_CD"]
        ],
        on="alpha_round",
        how="left",
    )

    merged = merged.rename(
        columns={
            "CL": "CL_novo",
            "CD": "CD_novo",
            "CM": "CM_novo",
            "CL_CD": "CL_CD_novo",
        }
    )

    merged["CL_CD_database"] = (
        merged["CL_database"] / merged["CD_database"]
    )

    merged["delta_CL"] = merged["CL_novo"] - merged["CL_database"]
    merged["delta_CD"] = merged["CD_novo"] - merged["CD_database"]
    merged["delta_CM"] = merged["CM_novo"] - merged["CM_database"]
    merged["delta_CL_CD"] = (
        merged["CL_CD_novo"] - merged["CL_CD_database"]
    )

    merged["erro_CD_pct"] = (
        100.0
        * np.abs(merged["delta_CD"])
        / np.maximum(np.abs(merged["CD_database"]), 1e-12)
    )

    return merged


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 88)
    print("VALIDAÇÃO ISOLADA XFOIL — PERFIL e376")
    print("=" * 88)

    if not XFOIL_EXE.exists():
        raise FileNotFoundError(f"XFOIL não encontrado em:\n{XFOIL_EXE}")

    profile_path = localizar_perfil()

    print(f"\nPerfil: {profile_path}")
    print(f"Re: {RE:.0f}")
    print(f"Mach: {MACH:.3f}")
    print(
        f"Alpha: {ALPHA_MIN:.2f}° -> {ALPHA_MAX:.2f}° "
        f"passo {ALPHA_STEP:.2f}°"
    )

    df = executar_xfoil(profile_path)

    output_polar = OUTPUT_DIR / "e376_polar_xfoil_025.csv"
    df.to_csv(output_polar, index=False)

    print("\n" + "=" * 88)
    print("RESULTADO DA NOVA POLAR")
    print("=" * 88)
    print(f"Pontos convergidos: {len(df)}")

    if not df.empty:
        best_idx = df["CL_CD"].idxmax()
        best = df.loc[best_idx]

        print(f"CL/CD máximo: {best['CL_CD']:.6f}")
        print(f"Alpha ótimo: {best['alpha']:.2f}°")
        print(f"CL: {best['CL']:.6f}")
        print(f"CD: {best['CD']:.8f}")
        print(f"CM: {best['CM']:.6f}")

        print("\nRegião de interesse 4°–9°:")
        cols = ["alpha", "CL", "CD", "CM", "CL_CD"]
        trecho = df[(df["alpha"] >= 4.0) & (df["alpha"] <= 9.0)][cols]
        print(trecho.to_string(index=False))

    comparacao = comparar_com_database(df)

    if comparacao is not None:
        output_comp = OUTPUT_DIR / "e376_comparacao_database.csv"
        comparacao.to_csv(output_comp, index=False)

        print("\n" + "=" * 88)
        print("COMPARAÇÃO COM O DATABASE ORIGINAL — PONTOS INTEIROS")
        print("=" * 88)

        cols_show = [
            "alpha",
            "CL_database",
            "CL_novo",
            "CD_database",
            "CD_novo",
            "CL_CD_database",
            "CL_CD_novo",
            "erro_CD_pct",
        ]

        print(
            comparacao[cols_show]
            .sort_values("alpha")
            .to_string(index=False)
        )

    print("\nArquivos salvos em:")
    print(OUTPUT_DIR)
    print(" - e376_polar_xfoil_025.csv")
    if comparacao is not None:
        print(" - e376_comparacao_database.csv")
    print(" - comandos_xfoil.txt")
    print(" - stdout.txt")
    print(" - stderr.txt")


if __name__ == "__main__":
    main()

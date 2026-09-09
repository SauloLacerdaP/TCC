"""
diagnostico_geometrico_curvatura_suavidade.py

Diagnóstico geométrico complementar para a base CST do TCC.

Mede, SEM impor automaticamente novas restrições:
1) curvatura/suavidade do extradorso;
2) curvatura/suavidade do intradorso;
3) número de inversões relevantes de curvatura;
4) comportamento no trecho interno, excluindo BA/BF;
5) distribuição de camber e espessura ao longo da corda.

A análise principal de curvatura usa 0.02 <= x/c <= 0.98 para evitar que
a singularidade/alta curvatura natural do bordo de ataque e o fechamento
do bordo de fuga dominem as métricas.

Saídas:
- metricas_geometricas_curvatura.csv
- resumo_distribuicao_curvatura.csv
- top_curvatura_extradorso.csv
- top_curvatura_intradorso.csv
- top_inversoes_curvatura.csv
- distribuicao_camber_espessura_estacoes.csv
- top20_CL_metricas_geometricas.csv
- perfil_s1223_metricas_geometricas.csv
- graficos PNG de diagnóstico
"""

from __future__ import annotations

import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CSV_CANDIDATES = [
    ROOT / "Output_dados" / "database_ml.csv",
    ROOT / "database_ml.csv",
]

OUTPUT_DIR = ROOT / "Output_dados" / "diagnostico_geometrico_curvatura"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N1 = 0.5
N2 = 1.0

# Alta resolução para derivadas numéricas.
N_GEOM_POINTS = 2001

# Região usada nas métricas principais: exclui BA/BF.
X_INTERIOR_MIN = 0.02
X_INTERIOR_MAX = 0.98

# Estações para estudar a distribuição longitudinal.
X_STATIONS = np.array([0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98])

CST_UPPER = [f"Au{i}" for i in range(7)]
CST_LOWER = [f"Al{i}" for i in range(7)]
CST_FEATURES = CST_UPPER + CST_LOWER


def localizar_csv():
    for p in REFERENCE_CSV_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Não encontrei database_ml.csv. Ajuste REFERENCE_CSV_CANDIDATES."
    )


def bernstein(n, i, x):
    return math.comb(n, i) * x**i * (1.0 - x)**(n-i)


def class_function(x):
    return x**N1 * (1.0-x)**N2


def cst_surface(x, coeffs):
    coeffs = np.asarray(coeffs, dtype=float)
    n = len(coeffs)-1
    S = np.zeros_like(x)
    for i, a in enumerate(coeffs):
        S += a * bernstein(n, i, x)
    return class_function(x) * S


def reconstruir_geometria(au, al, n_points=N_GEOM_POINTS):
    # Uniforme em x é preferível aqui para derivadas numéricas e integração.
    x = np.linspace(0.0, 1.0, n_points)
    yu = cst_surface(x, au)
    yl = cst_surface(x, al)
    return x, yu, yl


def derivadas_curvatura(x, y):
    """Retorna dy/dx, d2y/dx2 e curvatura geométrica assinada."""
    dy = np.gradient(y, x, edge_order=2)
    d2y = np.gradient(dy, x, edge_order=2)
    kappa = d2y / np.power(1.0 + dy**2, 1.5)
    return dy, d2y, kappa


def contar_inversoes_relevantes(kappa, threshold):
    """
    Conta mudanças de sinal após zerar curvaturas muito pequenas.
    threshold é definido relativamente à magnitude da própria curva,
    evitando contar ruído numérico perto de zero.
    """
    k = np.asarray(kappa, dtype=float).copy()
    k[np.abs(k) < threshold] = 0.0
    nz = k[k != 0.0]
    if len(nz) < 2:
        return 0
    return int(np.sum(np.sign(nz[1:]) != np.sign(nz[:-1])))


def metricas_superficie(x, y, prefix):
    dy, d2y, kappa = derivadas_curvatura(x, y)
    mask = (x >= X_INTERIOR_MIN) & (x <= X_INTERIOR_MAX)

    xi = x[mask]
    ki = kappa[mask]
    d2i = d2y[mask]

    abs_k = np.abs(ki)
    abs_d2 = np.abs(d2i)

    # L1 da variação da curvatura: mede "ondulação"/mudanças ao longo da corda.
    total_variation_kappa = float(np.sum(np.abs(np.diff(ki))))

    # RMS e percentis são mais robustos que somente o máximo.
    rms_kappa = float(np.sqrt(np.mean(ki**2)))
    p95_abs_kappa = float(np.percentile(abs_k, 95))
    p99_abs_kappa = float(np.percentile(abs_k, 99))
    max_abs_kappa = float(np.max(abs_k))
    rms_d2 = float(np.sqrt(np.mean(d2i**2)))

    # Limiar relevante: 2% do P95 de |kappa|, com piso numérico.
    sign_threshold = max(0.02 * p95_abs_kappa, 1e-8)
    inversions = contar_inversoes_relevantes(ki, sign_threshold)

    idx = int(np.argmax(abs_k))
    return {
        f"{prefix}_kappa_rms_interior": rms_kappa,
        f"{prefix}_kappa_abs_p95_interior": p95_abs_kappa,
        f"{prefix}_kappa_abs_p99_interior": p99_abs_kappa,
        f"{prefix}_kappa_abs_max_interior": max_abs_kappa,
        f"{prefix}_x_kappa_abs_max_interior": float(xi[idx]),
        f"{prefix}_kappa_total_variation_interior": total_variation_kappa,
        f"{prefix}_d2y_rms_interior": rms_d2,
        f"{prefix}_curvature_sign_threshold": float(sign_threshold),
        f"{prefix}_curvature_inversions_interior": inversions,
    }


def metricas_geometria(row):
    vals = row[CST_FEATURES].to_numpy(dtype=float)
    au, al = vals[:7], vals[7:]
    x, yu, yl = reconstruir_geometria(au, al)

    thickness = yu - yl
    camber = 0.5 * (yu + yl)

    out = {}
    out.update(metricas_superficie(x, yu, "upper"))
    out.update(metricas_superficie(x, yl, "lower"))

    i_t = int(np.argmax(thickness))
    i_c = int(np.argmax(np.abs(camber)))

    out.update({
        "t_max": float(thickness[i_t]),
        "t_max_pct": float(100*thickness[i_t]),
        "x_t_max": float(x[i_t]),
        "camber_abs_max": float(abs(camber[i_c])),
        "camber_abs_max_pct": float(100*abs(camber[i_c])),
        "x_camber_abs_max": float(x[i_c]),
        "min_thickness": float(np.min(thickness)),
    })

    # Distribuição longitudinal em estações fixas.
    for xs in X_STATIONS:
        yu_s = float(np.interp(xs, x, yu))
        yl_s = float(np.interp(xs, x, yl))
        t_s = yu_s - yl_s
        c_s = 0.5*(yu_s + yl_s)
        tag = f"x{int(round(xs*100)):02d}"
        out[f"yu_{tag}"] = yu_s
        out[f"yl_{tag}"] = yl_s
        out[f"thickness_{tag}"] = t_s
        out[f"thickness_pct_{tag}"] = 100*t_s
        out[f"camber_{tag}"] = c_s
        out[f"camber_pct_{tag}"] = 100*c_s

    return out


def resumo_serie(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "n": len(s),
        "min": s.min(),
        "p01": s.quantile(.01),
        "p05": s.quantile(.05),
        "p10": s.quantile(.10),
        "p25": s.quantile(.25),
        "mediana": s.quantile(.50),
        "p75": s.quantile(.75),
        "p90": s.quantile(.90),
        "p95": s.quantile(.95),
        "p99": s.quantile(.99),
        "max": s.max(),
        "media": s.mean(),
        "desvio_padrao": s.std(),
    }


def main():
    csv = localizar_csv()
    df = pd.read_csv(csv)

    missing = [c for c in CST_FEATURES if c not in df.columns]
    if missing:
        raise KeyError("Faltam colunas CST: " + ", ".join(missing))

    # Uma geometria por perfil/conjunto CST.
    keep = ["perfil"] if "perfil" in df.columns else []
    geom = df[keep + CST_FEATURES].dropna(subset=CST_FEATURES).drop_duplicates(
        subset=CST_FEATURES
    ).reset_index(drop=True)

    print("="*88)
    print("DIAGNÓSTICO DE CURVATURA / SUAVIDADE DA BASE CST")
    print("="*88)
    print(f"Arquivo: {csv}")
    print(f"Linhas no banco: {len(df)}")
    print(f"Geometrias únicas: {len(geom)}")
    print(
        f"Região principal de análise: "
        f"{X_INTERIOR_MIN:.2f} <= x/c <= {X_INTERIOR_MAX:.2f}"
    )

    rows = []
    for _, r in geom.iterrows():
        ident = {"perfil": r["perfil"]} if "perfil" in r.index else {}
        ident.update({c: float(r[c]) for c in CST_FEATURES})
        ident.update(metricas_geometria(r))
        rows.append(ident)

    metrics = pd.DataFrame(rows)

    # Acrescenta aerodinâmica na condição de referência, se disponível.
    if {"Re", "alpha"}.issubset(df.columns):
        ref = df[
            np.isclose(pd.to_numeric(df["Re"], errors="coerce"), 250000.0)
            & np.isclose(pd.to_numeric(df["alpha"], errors="coerce"), 6.0)
        ].copy()
        aero_cols = [c for c in ["perfil", "CL", "CD", "CM"] if c in ref.columns]
        if "perfil" in metrics.columns and "perfil" in aero_cols:
            ref = ref[aero_cols].drop_duplicates("perfil")
            metrics = metrics.merge(ref, on="perfil", how="left")

    metrics.to_csv(OUTPUT_DIR / "metricas_geometricas_curvatura.csv", index=False)

    # Resumo das métricas centrais.
    metric_cols = [
        "upper_kappa_rms_interior",
        "upper_kappa_abs_p95_interior",
        "upper_kappa_abs_p99_interior",
        "upper_kappa_abs_max_interior",
        "upper_kappa_total_variation_interior",
        "upper_d2y_rms_interior",
        "upper_curvature_inversions_interior",
        "lower_kappa_rms_interior",
        "lower_kappa_abs_p95_interior",
        "lower_kappa_abs_p99_interior",
        "lower_kappa_abs_max_interior",
        "lower_kappa_total_variation_interior",
        "lower_d2y_rms_interior",
        "lower_curvature_inversions_interior",
        "camber_abs_max_pct",
        "t_max_pct",
    ]

    summary_rows = []
    for c in metric_cols:
        if c in metrics:
            d = {"metrica": c}
            d.update(resumo_serie(metrics[c]))
            summary_rows.append(d)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "resumo_distribuicao_curvatura.csv", index=False)

    # Rankings de diagnóstico.
    metrics.sort_values(
        "upper_kappa_total_variation_interior", ascending=False
    ).head(30).to_csv(OUTPUT_DIR / "top_curvatura_extradorso.csv", index=False)

    metrics.sort_values(
        "lower_kappa_total_variation_interior", ascending=False
    ).head(30).to_csv(OUTPUT_DIR / "top_curvatura_intradorso.csv", index=False)

    inv = metrics.copy()
    inv["curvature_inversions_total"] = (
        inv["upper_curvature_inversions_interior"]
        + inv["lower_curvature_inversions_interior"]
    )
    inv.sort_values(
        ["curvature_inversions_total",
         "upper_kappa_total_variation_interior",
         "lower_kappa_total_variation_interior"],
        ascending=False
    ).head(30).to_csv(OUTPUT_DIR / "top_inversoes_curvatura.csv", index=False)

    # Tabela longa: distribuição de camber/espessura em cada estação.
    station_rows = []
    for _, r in metrics.iterrows():
        for xs in X_STATIONS:
            tag = f"x{int(round(xs*100)):02d}"
            station_rows.append({
                "perfil": r.get("perfil", ""),
                "x_c": xs,
                "camber_pct": r[f"camber_pct_{tag}"],
                "thickness_pct": r[f"thickness_pct_{tag}"],
                "yu": r[f"yu_{tag}"],
                "yl": r[f"yl_{tag}"],
            })
    stations = pd.DataFrame(station_rows)
    stations.to_csv(
        OUTPUT_DIR / "distribuicao_camber_espessura_estacoes.csv", index=False
    )

    # Destaques: s1223 e Top-20 CL.
    if "perfil" in metrics.columns:
        metrics[metrics["perfil"].astype(str).str.lower() == "s1223"].to_csv(
            OUTPUT_DIR / "perfil_s1223_metricas_geometricas.csv", index=False
        )

    if "CL" in metrics.columns:
        top20 = metrics.dropna(subset=["CL"]).sort_values("CL", ascending=False).head(20)
        top20.to_csv(OUTPUT_DIR / "top20_CL_metricas_geometricas.csv", index=False)

    # Impressão objetiva para o terminal.
    print("\n" + "="*88)
    print("RESUMO — SUAVIDADE/CURVATURA NO TRECHO INTERNO")
    print("="*88)
    show = [
        "upper_kappa_total_variation_interior",
        "lower_kappa_total_variation_interior",
        "upper_curvature_inversions_interior",
        "lower_curvature_inversions_interior",
        "upper_kappa_abs_p99_interior",
        "lower_kappa_abs_p99_interior",
    ]
    for c in show:
        s = resumo_serie(metrics[c])
        print(f"\n{c}")
        print(
            f"  P95={s['p95']:.6g} | P99={s['p99']:.6g} | "
            f"MAX={s['max']:.6g} | mediana={s['mediana']:.6g}"
        )

    if "perfil" in metrics.columns:
        s1223 = metrics[metrics["perfil"].astype(str).str.lower() == "s1223"]
        if not s1223.empty:
            r = s1223.iloc[0]
            print("\n" + "="*88)
            print("s1223 — MÉTRICAS DE REFERÊNCIA")
            print("="*88)
            for c in show + ["camber_abs_max_pct", "t_max_pct"]:
                print(f"{c}: {r[c]}")

    # Gráficos simples para inspeção.
    plt.figure(figsize=(8, 5))
    plt.hist(metrics["upper_kappa_total_variation_interior"].dropna(), bins=30)
    plt.xlabel("Variação total da curvatura — extradorso")
    plt.ylabel("Número de perfis")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hist_variacao_curvatura_extradorso.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(metrics["lower_kappa_total_variation_interior"].dropna(), bins=30)
    plt.xlabel("Variação total da curvatura — intradorso")
    plt.ylabel("Número de perfis")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hist_variacao_curvatura_intradorso.png", dpi=180)
    plt.close()

    if "CL" in metrics.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            metrics["upper_kappa_total_variation_interior"],
            metrics["CL"],
            alpha=0.7,
        )
        plt.xlabel("Variação total da curvatura — extradorso")
        plt.ylabel("CL @ Re=250k, alpha=6°")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "curvatura_extradorso_vs_CL.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.scatter(
            metrics["lower_kappa_total_variation_interior"],
            metrics["CL"],
            alpha=0.7,
        )
        plt.xlabel("Variação total da curvatura — intradorso")
        plt.ylabel("CL @ Re=250k, alpha=6°")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "curvatura_intradorso_vs_CL.png", dpi=180)
        plt.close()

    print("\n" + "="*88)
    print("ARQUIVOS GERADOS")
    print("="*88)
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(" -", p.name)

    print(
        "\nIMPORTANTE: este script é diagnóstico. "
        "Não aplique P95/P99 automaticamente como restrição. "
        "Compare primeiro os candidatos problemáticos, o s1223 e os Top-CL."
    )


if __name__ == "__main__":
    main()

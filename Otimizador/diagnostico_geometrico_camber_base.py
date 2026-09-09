import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

# Banco com parâmetros CST e coluna "perfil".
# Ajuste se necessário.
CAMINHOS_CANDIDATOS = [
    Path(r"C:\Repositorios\TCC\Output_dados\database_ml.csv"),
    Path(r"C:\Repositorios\TCC\database_ml.csv"),
    Path(r"C:\Repositorios\TCC\Dados\database_ml.csv"),
]

ARQUIVO_DATABASE = next((p for p in CAMINHOS_CANDIDATOS if p.exists()), CAMINHOS_CANDIDATOS[0])

PASTA_SAIDA = Path(r"C:\Repositorios\TCC\Output_dados\diagnostico_geometrico")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

COLUNAS_AU = [f"Au{i}" for i in range(7)]
COLUNAS_AL = [f"Al{i}" for i in range(7)]

# Discretização para reconstrução CST
N_X = 2001
X_EPS = 1e-6

# CST padrão para aerofólios
N1 = 0.5
N2 = 1.0

# Se database_ml possuir DeltaTE_upper/lower, serão usados.
COL_DTE_U = "DeltaTE_upper"
COL_DTE_L = "DeltaTE_lower"

# Condição usada apenas para ranking aerodinâmico de referência.
RE_REF = 250000
ALPHA_REF = 6.0
MACH_REF = 0.1
TOP_N_CL = 20

# Perfis especiais que queremos destacar
PERFIS_DESTAQUE = {"s1223"}

# ============================================================
# FUNÇÕES CST
# ============================================================

def bernstein(n, i, x):
    from math import comb
    return comb(n, i) * (x ** i) * ((1.0 - x) ** (n - i))


def superficie_cst(x, A, delta_te=0.0):
    """
    y(x) = C(x) * S(x) + x * DeltaTE
    C(x) = x^N1 * (1-x)^N2
    """
    A = np.asarray(A, dtype=float)
    n = len(A) - 1

    C = (x ** N1) * ((1.0 - x) ** N2)

    S = np.zeros_like(x, dtype=float)
    for i, ai in enumerate(A):
        S += ai * bernstein(n, i, x)

    return C * S + x * float(delta_te)


def reconstruir_geometria(row):
    x = np.linspace(X_EPS, 1.0 - X_EPS, N_X)

    Au = row[COLUNAS_AU].to_numpy(dtype=float)
    Al = row[COLUNAS_AL].to_numpy(dtype=float)

    dte_u = float(row[COL_DTE_U]) if COL_DTE_U in row.index and pd.notna(row[COL_DTE_U]) else 0.0
    dte_l = float(row[COL_DTE_L]) if COL_DTE_L in row.index and pd.notna(row[COL_DTE_L]) else 0.0

    yu = superficie_cst(x, Au, dte_u)
    yl = superficie_cst(x, Al, dte_l)

    espessura = yu - yl
    camber = 0.5 * (yu + yl)

    return x, yu, yl, espessura, camber


def metricas_geometricas(row):
    x, yu, yl, t, camber = reconstruir_geometria(row)

    i_tmax = int(np.nanargmax(t))
    i_cmax = int(np.nanargmax(camber))
    i_cmin = int(np.nanargmin(camber))
    i_abs = int(np.nanargmax(np.abs(camber)))

    return {
        "t_max": float(t[i_tmax]),
        "t_max_pct": float(100.0 * t[i_tmax]),
        "x_t_max": float(x[i_tmax]),

        "camber_max": float(camber[i_cmax]),
        "camber_max_pct": float(100.0 * camber[i_cmax]),
        "x_camber_max": float(x[i_cmax]),

        "camber_min": float(camber[i_cmin]),
        "camber_min_pct": float(100.0 * camber[i_cmin]),
        "x_camber_min": float(x[i_cmin]),

        "camber_abs_max": float(abs(camber[i_abs])),
        "camber_abs_max_pct": float(100.0 * abs(camber[i_abs])),
        "x_camber_abs_max": float(x[i_abs]),

        "espessura_min": float(np.nanmin(t)),
        "superficies_cruzam": bool(np.nanmin(t) < -1e-7),
    }


# ============================================================
# CARREGAMENTO E REDUÇÃO PARA 1 LINHA POR PERFIL
# ============================================================

if not ARQUIVO_DATABASE.exists():
    raise FileNotFoundError(
        f"database_ml.csv não encontrado.\n"
        f"Ajuste ARQUIVO_DATABASE no início do script.\n"
        f"Caminho tentado: {ARQUIVO_DATABASE}"
    )

df = pd.read_csv(ARQUIVO_DATABASE)

obrigatorias = ["perfil"] + COLUNAS_AU + COLUNAS_AL
faltando = [c for c in obrigatorias if c not in df.columns]
if faltando:
    raise ValueError(f"Colunas obrigatórias ausentes: {faltando}")

print("=" * 88)
print("DIAGNÓSTICO GEOMÉTRICO DA BASE CST")
print("=" * 88)
print(f"Arquivo: {ARQUIVO_DATABASE}")
print(f"Linhas no banco: {len(df)}")
print(f"Perfis únicos: {df['perfil'].nunique()}")

# CST é constante por perfil; usamos uma linha representativa.
cols_geom = ["perfil"] + COLUNAS_AU + COLUNAS_AL
for c in [COL_DTE_U, COL_DTE_L]:
    if c in df.columns:
        cols_geom.append(c)

df_geom = df[cols_geom].drop_duplicates(subset=["perfil"]).copy()

# ============================================================
# CÁLCULO DAS MÉTRICAS
# ============================================================

registros = []
for idx, row in df_geom.iterrows():
    m = metricas_geometricas(row)
    m["perfil"] = row["perfil"]
    registros.append(m)

geo = pd.DataFrame(registros)

# Adiciona CST ao arquivo final para auditoria
geo = geo.merge(df_geom, on="perfil", how="left")

# ============================================================
# DADOS AERODINÂMICOS DE REFERÊNCIA (Re=250k, alpha=6°)
# ============================================================

df_ref = df.copy()

if "Re" in df_ref.columns:
    df_ref = df_ref[np.isclose(pd.to_numeric(df_ref["Re"], errors="coerce"), RE_REF)]
if "alpha" in df_ref.columns:
    df_ref = df_ref[np.isclose(pd.to_numeric(df_ref["alpha"], errors="coerce"), ALPHA_REF)]
if "Mach" in df_ref.columns:
    df_ref = df_ref[np.isclose(pd.to_numeric(df_ref["Mach"], errors="coerce"), MACH_REF)]

aero_cols = [c for c in ["perfil", "CL", "CD", "CM"] if c in df_ref.columns]
if len(aero_cols) > 1:
    aero = df_ref[aero_cols].drop_duplicates(subset=["perfil"])
    geo = geo.merge(aero, on="perfil", how="left")

# ============================================================
# ESTATÍSTICAS DA DISTRIBUIÇÃO
# ============================================================

def resumo_serie(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    qs = s.quantile([0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1.0])
    return pd.Series({
        "n": len(s),
        "min": qs.loc[0.0],
        "p01": qs.loc[0.01],
        "p05": qs.loc[0.05],
        "p10": qs.loc[0.10],
        "p25": qs.loc[0.25],
        "mediana": qs.loc[0.50],
        "p75": qs.loc[0.75],
        "p90": qs.loc[0.90],
        "p95": qs.loc[0.95],
        "p99": qs.loc[0.99],
        "max": qs.loc[1.0],
        "media": s.mean(),
        "desvio_padrao": s.std(),
    })


metricas_resumo = [
    "t_max_pct",
    "x_t_max",
    "camber_max_pct",
    "x_camber_max",
    "camber_abs_max_pct",
    "x_camber_abs_max",
]

resumo = pd.DataFrame({c: resumo_serie(geo[c]) for c in metricas_resumo}).T
resumo.to_csv(PASTA_SAIDA / "resumo_distribuicao_geometrica.csv", encoding="utf-8-sig")

# ============================================================
# RANKINGS IMPORTANTES
# ============================================================

rank_camber = geo.sort_values("camber_abs_max_pct", ascending=False).copy()
rank_camber.to_csv(PASTA_SAIDA / "ranking_maior_camber.csv", index=False, encoding="utf-8-sig")

rank_espessura = geo.sort_values("t_max_pct", ascending=False).copy()
rank_espessura.to_csv(PASTA_SAIDA / "ranking_maior_espessura.csv", index=False, encoding="utf-8-sig")

# Top-N CL na condição de referência
if "CL" in geo.columns:
    top_cl = geo.dropna(subset=["CL"]).sort_values("CL", ascending=False).head(TOP_N_CL).copy()
    top_cl.to_csv(
        PASTA_SAIDA / f"top_{TOP_N_CL}_CL_Re{RE_REF}_alpha{ALPHA_REF:g}.csv",
        index=False,
        encoding="utf-8-sig"
    )
else:
    top_cl = pd.DataFrame()

# Destaques
destaques = geo[geo["perfil"].astype(str).str.lower().isin({p.lower() for p in PERFIS_DESTAQUE})].copy()
destaques.to_csv(PASTA_SAIDA / "perfis_destaque.csv", index=False, encoding="utf-8-sig")

# ============================================================
# SUGESTÃO EMPÍRICA DE LIMITES
# ============================================================

# Não aplicamos automaticamente estes limites ao otimizador.
# O objetivo é fornecer valores defensáveis para inspeção.
camber_p95 = float(geo["camber_abs_max_pct"].quantile(0.95))
camber_p99 = float(geo["camber_abs_max_pct"].quantile(0.99))
camber_max_base = float(geo["camber_abs_max_pct"].max())

t_p01 = float(geo["t_max_pct"].quantile(0.01))
t_p99 = float(geo["t_max_pct"].quantile(0.99))

sugestoes = pd.DataFrame([
    {
        "criterio": "camber_abs_max_pct_P95",
        "valor": camber_p95,
        "interpretacao": "95% dos perfis da base possuem camber absoluto máximo <= este valor"
    },
    {
        "criterio": "camber_abs_max_pct_P99",
        "valor": camber_p99,
        "interpretacao": "99% dos perfis da base possuem camber absoluto máximo <= este valor"
    },
    {
        "criterio": "camber_abs_max_pct_MAX_BASE",
        "valor": camber_max_base,
        "interpretacao": "Maior camber absoluto máximo observado em qualquer perfil real da base"
    },
    {
        "criterio": "t_max_pct_P01",
        "valor": t_p01,
        "interpretacao": "Percentil 1 da espessura máxima"
    },
    {
        "criterio": "t_max_pct_P99",
        "valor": t_p99,
        "interpretacao": "Percentil 99 da espessura máxima"
    },
])
sugestoes.to_csv(PASTA_SAIDA / "limites_geometricos_candidatos.csv", index=False, encoding="utf-8-sig")

# ============================================================
# RELATÓRIO NO TERMINAL
# ============================================================

print("\n" + "=" * 88)
print("DISTRIBUIÇÃO DE CAMBER ABSOLUTO MÁXIMO (%c)")
print("=" * 88)
print(resumo.loc["camber_abs_max_pct"].to_string())

print("\n" + "=" * 88)
print("DISTRIBUIÇÃO DE ESPESSURA MÁXIMA (%c)")
print("=" * 88)
print(resumo.loc["t_max_pct"].to_string())

print("\n" + "=" * 88)
print("TOP 15 — MAIOR CAMBER ABSOLUTO DA BASE")
print("=" * 88)
cols_print = ["perfil", "camber_abs_max_pct", "x_camber_abs_max", "t_max_pct", "x_t_max"]
for c in ["CL", "CD", "CM"]:
    if c in rank_camber.columns:
        cols_print.append(c)
print(rank_camber[cols_print].head(15).to_string(index=False))

if not destaques.empty:
    print("\n" + "=" * 88)
    print("PERFIS DE DESTAQUE")
    print("=" * 88)
    cols = ["perfil", "camber_max_pct", "camber_abs_max_pct", "x_camber_abs_max", "t_max_pct", "x_t_max"]
    for c in ["CL", "CD", "CM"]:
        if c in destaques.columns:
            cols.append(c)
    print(destaques[cols].to_string(index=False))

if not top_cl.empty:
    print("\n" + "=" * 88)
    print(f"TOP {TOP_N_CL} CL — Re={RE_REF}, alpha={ALPHA_REF}°")
    print("=" * 88)
    cols = ["perfil", "CL"]
    for c in ["CD", "CM", "camber_abs_max_pct", "x_camber_abs_max", "t_max_pct"]:
        if c in top_cl.columns:
            cols.append(c)
    print(top_cl[cols].to_string(index=False))

print("\n" + "=" * 88)
print("VALORES PARA DISCUSSÃO DE NOVAS RESTRIÇÕES")
print("=" * 88)
print(f"Camber abs. máx. P95 : {camber_p95:.4f}% c")
print(f"Camber abs. máx. P99 : {camber_p99:.4f}% c")
print(f"Camber abs. máx. BASE: {camber_max_base:.4f}% c")
print(f"Espessura máx. P01   : {t_p01:.4f}% c")
print(f"Espessura máx. P99   : {t_p99:.4f}% c")

print("\nIMPORTANTE:")
print("Não use automaticamente P95/P99 como restrição antes de conferir o s1223")
print("e os perfis Top-CL. O objetivo deste script é medir a distribuição real primeiro.")

print("\nArquivos salvos em:")
print(PASTA_SAIDA)

# ============================================================
# GRÁFICOS
# ============================================================

plt.figure(figsize=(8, 5))
plt.hist(geo["camber_abs_max_pct"].dropna(), bins=30)
plt.axvline(camber_p95, linestyle="--", label=f"P95 = {camber_p95:.2f}%")
plt.axvline(camber_p99, linestyle="--", label=f"P99 = {camber_p99:.2f}%")
plt.xlabel("Camber absoluto máximo (% da corda)")
plt.ylabel("Número de perfis")
plt.title("Distribuição do camber máximo da base")
plt.legend()
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "histograma_camber_maximo.png", dpi=200)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(geo["camber_abs_max_pct"], geo["t_max_pct"], s=20)
plt.xlabel("Camber absoluto máximo (% da corda)")
plt.ylabel("Espessura máxima (% da corda)")
plt.title("Camber máximo × espessura máxima")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "camber_vs_espessura.png", dpi=200)
plt.close()

if "CL" in geo.columns:
    tmp = geo.dropna(subset=["CL", "camber_abs_max_pct"])
    plt.figure(figsize=(8, 5))
    plt.scatter(tmp["camber_abs_max_pct"], tmp["CL"], s=20)
    plt.xlabel("Camber absoluto máximo (% da corda)")
    plt.ylabel(f"CL em Re={RE_REF}, alpha={ALPHA_REF}°")
    plt.title("Camber máximo × CL")
    plt.tight_layout()
    plt.savefig(PASTA_SAIDA / "camber_vs_CL.png", dpi=200)
    plt.close()

print("\nDiagnóstico concluído.")

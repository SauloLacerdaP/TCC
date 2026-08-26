import pandas as pd
from pathlib import Path


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVO_XFOIL = Path(
    r"C:\Repositorios\TCC\Output_dados\banco_dados_xfoil.csv"
)

ARQUIVO_CST = Path(
    r"C:\Repositorios\TCC\Output_dados\database_cst.csv"
)

ARQUIVO_SAIDA = Path(
    r"C:\Repositorios\TCC\Output_dados\database_ml.csv"
)


# ============================================================
# EXTRAÇÃO
# ============================================================

print("Carregando banco de dados do XFOIL...")
df_xfoil = pd.read_csv(ARQUIVO_XFOIL)

print("Carregando banco de dados CST...")
df_cst = pd.read_csv(ARQUIVO_CST)


# ============================================================
# INFORMAÇÕES INICIAIS
# ============================================================

print("\n--- Dados originais ---")
print(f"Linhas XFOIL: {len(df_xfoil)}")
print(f"Perfis XFOIL: {df_xfoil['perfil'].nunique()}")

print(f"Linhas CST: {len(df_cst)}")
print(f"Perfis CST: {df_cst['airfoil_id'].nunique()}")


# ============================================================
# TRANSFORMAÇÃO
# ============================================================

# ------------------------------------------------------------
# 1. Padronizar nomes dos perfis
# ------------------------------------------------------------

df_xfoil["perfil_join"] = (
    df_xfoil["perfil"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df_cst["perfil_join"] = (
    df_cst["airfoil_id"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ------------------------------------------------------------
# 2. Garantir que alpha seja numérico
# ------------------------------------------------------------

df_xfoil["alpha"] = pd.to_numeric(
    df_xfoil["alpha"],
    errors="coerce"
)


# ------------------------------------------------------------
# 3. Remover linhas inválidas de alpha
# ------------------------------------------------------------

df_xfoil = df_xfoil.dropna(subset=["alpha"])


# ------------------------------------------------------------
# 4. Filtrar alpha entre 0 e 12 graus
# ------------------------------------------------------------

df_xfoil = df_xfoil[
    df_xfoil["alpha"].between(0, 12, inclusive="both")
].copy()


print("\n--- Após filtro de alpha ---")
print(f"Linhas restantes: {len(df_xfoil)}")
print(
    f"Alpha mínimo: {df_xfoil['alpha'].min()}"
)
print(
    f"Alpha máximo: {df_xfoil['alpha'].max()}"
)


# ------------------------------------------------------------
# 5. Verificar perfis sem correspondência
# ------------------------------------------------------------

perfis_xfoil = set(df_xfoil["perfil_join"])
perfis_cst = set(df_cst["perfil_join"])

sem_cst = sorted(perfis_xfoil - perfis_cst)
sem_xfoil = sorted(perfis_cst - perfis_xfoil)

if sem_cst:
    print("\nATENÇÃO: perfis XFOIL sem CST:")
    for perfil in sem_cst:
        print(f"  - {perfil}")
else:
    print("\nTodos os perfis do XFOIL possuem dados CST.")

if sem_xfoil:
    print("\nPerfis CST sem resultados no XFOIL:")
    for perfil in sem_xfoil:
        print(f"  - {perfil}")


# ------------------------------------------------------------
# 6. Fazer JOIN
# ------------------------------------------------------------

df_final = pd.merge(
    df_xfoil,
    df_cst,
    on="perfil_join",
    how="inner",
    validate="many_to_one"
)


# ============================================================
# LIMPEZA DO DATASET FINAL
# ============================================================

# A coluna auxiliar não é mais necessária
df_final = df_final.drop(
    columns=["perfil_join"]
)

# Como airfoil_id e perfil representam a mesma informação,
# podemos remover airfoil_id
df_final = df_final.drop(
    columns=["airfoil_id"],
    errors="ignore"
)


# ============================================================
# ORGANIZAÇÃO DAS COLUNAS
# ============================================================

# Colunas aerodinâmicas
colunas_xfoil = [
    "perfil",
    "Re",
    "Mach",
    "alpha",
    "CL",
    "CD",
    "CDp",
    "CM",
    "Top_Xtr",
    "Bot_Xtr"
]

# Coeficientes CST
colunas_cst = [
    "Au0",
    "Au1",
    "Au2",
    "Au3",
    "Au4",
    "Au5",
    "Au6",
    "Al0",
    "Al1",
    "Al2",
    "Al3",
    "Al4",
    "Al5",
    "Al6",
    "DeltaTE_upper",
    "DeltaTE_lower"
]

# Outras informações do CST que podem ser úteis
colunas_qualidade_cst = [
    "RMSE_upper",
    "RMSE_lower",
    "MaxError_upper",
    "MaxError_lower"
]


colunas_finais = (
    colunas_xfoil
    + colunas_cst
    + colunas_qualidade_cst
)


# Mantém apenas colunas existentes
colunas_finais = [
    coluna
    for coluna in colunas_finais
    if coluna in df_final.columns
]

df_final = df_final[colunas_finais]


# ============================================================
# ORDENAÇÃO
# ============================================================

df_final = df_final.sort_values(
    by=["perfil", "Re", "Mach", "alpha"]
).reset_index(drop=True)


# ============================================================
# LOAD
# ============================================================

ARQUIVO_SAIDA.parent.mkdir(
    parents=True,
    exist_ok=True
)

df_final.to_csv(
    ARQUIVO_SAIDA,
    index=False,
    encoding="utf-8"
)


# ============================================================
# RESUMO FINAL
# ============================================================

print("\n========================================")
print("ETL FINALIZADO")
print("========================================")

print(f"Arquivo salvo em:")
print(ARQUIVO_SAIDA)

print(f"\nNúmero de linhas: {len(df_final)}")
print(
    f"Número de perfis: "
    f"{df_final['perfil'].nunique()}"
)

print(
    f"Reynolds encontrados: "
    f"{sorted(df_final['Re'].unique())}"
)

print(
    f"Mach encontrados: "
    f"{sorted(df_final['Mach'].unique())}"
)

print(
    f"Alphas encontrados: "
    f"{sorted(df_final['alpha'].unique())}"
)

print("\nColunas finais:")
for coluna in df_final.columns:
    print(f"  - {coluna}")
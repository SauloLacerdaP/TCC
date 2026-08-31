import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DADOS = Path(
    r"C:\Repositorios\TCC\Output_dados\ml_preparado"
)

PASTA_SAIDA = Path(
    r"C:\Repositorios\TCC\Output_dados\resultados_xgboost"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)

TARGETS = ["CL", "CD", "CM"]

# Intervalos de alpha
ALPHA_RANGES = [
    (0, 6, "0-6°"),
    (7, 9, "7-9°"),
    (10, 12, "10-12°"),
    (13, 15, "13-15°"),
    (-6, -1, "-6 a -1°"),
]


# ============================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================

print("=" * 80)
print("ANÁLISE XGBOOST POR INTERVALO DE ALPHA")
print("=" * 80)

# Carregar dados de teste com alpha originais
test_completo = pd.read_csv(PASTA_DADOS / "test.csv")
X_test = pd.read_csv(PASTA_DADOS / "X_test.csv")
y_test = pd.read_csv(PASTA_DADOS / "y_test.csv")

print(f"\n✓ Dados carregados: {X_test.shape[0]} amostras")

# Extração de alpha (coluna original)
alpha_values = test_completo["alpha"].values


# ============================================================
# 2. CARREGAMENTO DOS MODELOS
# ============================================================

modelos = {}
for target in TARGETS:
    caminho_modelo = PASTA_SAIDA / f"xgboost_{target}.pkl"
    if caminho_modelo.exists():
        modelos[target] = joblib.load(caminho_modelo)
        print(f"✓ Modelo {target} carregado")
    else:
        print(f"✗ Modelo {target} não encontrado em {caminho_modelo}")

if len(modelos) != len(TARGETS):
    print("\n⚠ Nem todos os modelos foram encontrados!")


# ============================================================
# 3. FUNÇÃO DE CÁLCULO DE MÉTRICAS
# ============================================================

def calcular_metricas(y_real, y_pred):
    """Calcula métricas de desempenho"""
    return {
        "R²": r2_score(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "MAE": mean_absolute_error(y_real, y_pred),
        "MSE": mean_squared_error(y_real, y_pred),
    }


# ============================================================
# 4. PREDIÇÃO E ANÁLISE POR ALPHA
# ============================================================

print("\n" + "=" * 80)
print("ANÁLISE POR INTERVALO DE ALPHA")
print("=" * 80)

resultados_por_intervalo = {}

for target in TARGETS:
    print(f"\n{'─' * 80}")
    print(f"TARGET: {target}")
    print(f"{'─' * 80}")
    
    if target not in modelos:
        continue
    
    modelo = modelos[target]
    y_pred = modelo.predict(X_test)
    y_real = y_test[target].values
    
    resultados_por_intervalo[target] = {}
    
    for alpha_min, alpha_max, label_alpha in ALPHA_RANGES:
        # Filtrar dados por intervalo de alpha
        mascara = (alpha_values >= alpha_min) & (alpha_values <= alpha_max)
        
        if mascara.sum() == 0:
            print(f"\n  {label_alpha}: Sem dados")
            continue
        
        y_real_intervalo = y_real[mascara]
        y_pred_intervalo = y_pred[mascara]
        
        metricas = calcular_metricas(y_real_intervalo, y_pred_intervalo)
        resultados_por_intervalo[target][label_alpha] = metricas
        
        # Exibição formatada
        print(f"\n  {label_alpha:>10} (n={mascara.sum():>4})")
        print(f"    R²   = {metricas['R²']:>8.4f}")
        print(f"    RMSE = {metricas['RMSE']:>8.4f}")
        print(f"    MAE  = {metricas['MAE']:>8.4f}")


# ============================================================
# 5. TABELA RESUMIDA
# ============================================================

print("\n" + "=" * 80)
print("RESUMO DAS MÉTRICAS POR INTERVALO")
print("=" * 80)

for target in TARGETS:
    print(f"\n{'─' * 80}")
    print(f"TARGET: {target} (R² por Intervalo)")
    print(f"{'─' * 80}")
    
    if target not in resultados_por_intervalo:
        continue
    
    df_r2 = pd.DataFrame({
        label: metricas['R²']
        for label, metricas in resultados_por_intervalo[target].items()
    }, index=['R²']).T
    
    print(df_r2.to_string())


# ============================================================
# 6. VISUALIZAÇÃO - R² POR INTERVALO
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('R² por Intervalo de Alpha', fontsize=14, fontweight='bold')

for idx, target in enumerate(TARGETS):
    if target not in resultados_por_intervalo:
        continue
    
    intervalos = list(resultados_por_intervalo[target].keys())
    r2_valores = [
        resultados_por_intervalo[target][label]['R²']
        for label in intervalos
    ]
    
    cores = ['#2ecc71' if r2 > 0.9 else '#f39c12' if r2 > 0.8 else '#e74c3c'
             for r2 in r2_valores]
    
    axes[idx].bar(intervalos, r2_valores, color=cores, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{target}', fontweight='bold')
    axes[idx].set_ylabel('R²')
    axes[idx].set_ylim([0, 1])
    axes[idx].grid(axis='y', alpha=0.3)
    axes[idx].tick_params(axis='x', rotation=45)
    
    # Adicionar valores nas barras
    for i, (intervalo, r2) in enumerate(zip(intervalos, r2_valores)):
        axes[idx].text(i, r2 + 0.02, f'{r2:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'r2_por_alpha_intervalo.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico R² salvo: r2_por_alpha_intervalo.png")


# ============================================================
# 7. VISUALIZAÇÃO - RMSE POR INTERVALO
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('RMSE por Intervalo de Alpha', fontsize=14, fontweight='bold')

for idx, target in enumerate(TARGETS):
    if target not in resultados_por_intervalo:
        continue
    
    intervalos = list(resultados_por_intervalo[target].keys())
    rmse_valores = [
        resultados_por_intervalo[target][label]['RMSE']
        for label in intervalos
    ]
    
    axes[idx].bar(intervalos, rmse_valores, color='#3498db', alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{target}', fontweight='bold')
    axes[idx].set_ylabel('RMSE')
    axes[idx].grid(axis='y', alpha=0.3)
    axes[idx].tick_params(axis='x', rotation=45)
    
    # Adicionar valores nas barras
    for i, (intervalo, rmse) in enumerate(zip(intervalos, rmse_valores)):
        axes[idx].text(i, rmse + 0.005, f'{rmse:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PASTA_SAIDA / 'rmse_por_alpha_intervalo.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico RMSE salvo: rmse_por_alpha_intervalo.png")


# ============================================================
# 8. TABELA DE MÉTRICAS COMPLETA
# ============================================================

for target in TARGETS:
    if target not in resultados_por_intervalo:
        continue
    
    df_metricas = pd.DataFrame(resultados_por_intervalo[target]).T
    df_metricas.to_csv(PASTA_SAIDA / f'metricas_por_alpha_{target}.csv')
    print(f"✓ Tabela de métricas salva: metricas_por_alpha_{target}.csv")


# ============================================================
# 9. RESUMO FINAL
# ============================================================

print("\n" + "=" * 80)
print("ANÁLISE CONCLUÍDA")
print("=" * 80)

for target in TARGETS:
    if target in resultados_por_intervalo:
        r2_medio = np.mean([m['R²'] for m in resultados_por_intervalo[target].values()])
        rmse_medio = np.mean([m['RMSE'] for m in resultados_por_intervalo[target].values()])
        print(f"\n{target}:")
        print(f"  R² médio:   {r2_medio:.4f}")
        print(f"  RMSE médio: {rmse_medio:.4f}")

print("\n" + "=" * 80)


# ============================================================
# 10. PREDIÇÃO PARA CASO ESPECÍFICO: Re=250.000, alpha=5°
# ============================================================

print("\n" + "=" * 80)
print("PREDIÇÃO PARA CASO ESPECÍFICO")
print("=" * 80)

Re_target = 250000
alpha_target = 5.0

print(f"\nParâmetros:")
print(f"  Reynolds: {Re_target}")
print(f"  Alpha:    {alpha_target}°")

# Buscar dados para Re=250k e alpha=5° no dataset completo
mask_caso = (test_completo["Re"].values == Re_target) & (test_completo["alpha"].values == alpha_target)
indices_caso = np.where(mask_caso)[0]

if len(indices_caso) > 0:
    print(f"\n✓ Encontrado {len(indices_caso)} caso(s) com Re=250k e alpha=5°")
    
    # Extrair features CST correspondentes
    X_caso = X_test.iloc[indices_caso].values
    y_real_caso = y_test.iloc[indices_caso]
    test_caso = test_completo.iloc[indices_caso]
    
    # Fazer predições para cada target
    print("\n" + "─" * 80)
    print("RESULTADOS DAS PREDIÇÕES")
    print("─" * 80)
    
    for target in TARGETS:
        if target not in modelos:
            continue
        
        modelo = modelos[target]
        y_pred_caso = modelo.predict(X_caso)
        y_real_valor = y_real_caso[target].values
        
        print(f"\n{target}:")
        
        for i, (pred, real) in enumerate(zip(y_pred_caso, y_real_valor)):
            erro_abs = abs(pred - real)
            erro_pct = (erro_abs / abs(real)) * 100 if real != 0 else 0
            
            print(f"  Amostra {i+1}:")
            print(f"    Valor Real:      {real:.6f}")
            print(f"    Valor Predito:   {pred:.6f}")
            print(f"    Erro Absoluto:   {erro_abs:.6f}")
            print(f"    Erro %:          {erro_pct:.2f}%")
    
    # Salvar resultados em CSV
    df_predicoes_caso = pd.DataFrame({
        "perfil": test_caso["perfil"].values,
        "Re": Re_target,
        "Mach": test_caso["Mach"].values,
        "alpha": alpha_target
    })
    
    for target in TARGETS:
        if target not in modelos:
            continue
        modelo = modelos[target]
        y_pred = modelo.predict(X_caso)
        y_real = y_real_caso[target].values
        
        df_predicoes_caso[f"{target}_real"] = y_real
        df_predicoes_caso[f"{target}_pred"] = y_pred
        df_predicoes_caso[f"{target}_erro"] = np.abs(y_pred - y_real)
    
    df_predicoes_caso.to_csv(
        PASTA_SAIDA / f"predicoes_Re{Re_target}_alpha{alpha_target}.csv",
        index=False
    )
    print(f"\n✓ Resultados salvos: predicoes_Re{Re_target}_alpha{alpha_target}.csv")

else:
    print(f"\n✗ Nenhum caso encontrado com Re={Re_target} e alpha={alpha_target}°")

print("\n" + "=" * 80)

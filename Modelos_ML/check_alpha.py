import pandas as pd
from pathlib import Path

PASTA_DADOS = Path(r'C:\Repositorios\TCC\Output_dados\ml_preparado')

# Carregar dados
train = pd.read_csv(PASTA_DADOS / 'train.csv')
X_test = pd.read_csv(PASTA_DADOS / 'X_test.csv')
test = pd.read_csv(PASTA_DADOS / 'test.csv')

print("Colunas do X_test:", X_test.columns.tolist()[:10])
print("\nColunas do test.csv:", test.columns.tolist()[:10])
print("\nPrimeiras linhas do test.csv:")
print(test.head())

# Pasta `Prep_ML`

A pasta `Prep_ML` organiza a preparação do dataset para uso em aprendizagem de máquina.

## O que foi feito

- O script `prep_ml.py` carrega a base de dados integrada, proveniente da etapa de ETL, e cria a estrutura de particionamento para treino, validação e teste.
- Define a semente `RANDOM_STATE = 42` e usa `GroupShuffleSplit` para preservar o agrupamento por perfil, evitando vazamento de geometria entre subconjuntos.
- Aplica padronização com `StandardScaler`, separa os alvos aerodinâmicos e salva os arquivos prontos em `Output_dados/ml_preparado/`.

## Papel no projeto

A preparação do ML converte a base de dados bruta em uma representação estruturada, balanceada e transformada para treinar regressões e modelos de árvore com métricas comparáveis.

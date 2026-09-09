# Pasta `etl`

A pasta `etl` é o módulo de preparação e integração de dados do projeto.

## O que foi feito

- O arquivo `etl.py` lê as saídas geradas pelo XFOIL e pelo pipeline CST a partir de arquivos CSV já produzidos em `Output_dados/`.
- Ele normaliza nomes de perfis, padroniza chaves de junção entre as bases, remove registros com qualidade numérica problemática e aplica regras de cobertura mínima.
- A etapa final produz uma base unificada armazenada em `Output_dados/database_ml.csv`, pronta para alimentar os modelos de machine learning.

## Papel no projeto

A ETL junta, limpa, valida e transforma os dados aerodinâmicos em um dataset analítico consistente. Sem essa etapa, a base de treinamento dos modelos ficaria incompatível entre as fontes de XFOIL e CST.

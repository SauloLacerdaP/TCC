# Pasta `xfoil`

A pasta `xfoil` concentra o fluxo de execução do XFOIL para obtenção das curvas aerodinâmicas dos perfis.

## O que foi feito

- O script `run_xfoil.py` configura a chamada ao executável `xfoil.exe`, define a pasta de perfis `.dat`, os pontos de alfa e os valores de Reynolds e Mach.
- A execução usa paralelização com `ProcessPoolExecutor` para rodar várias simulações em paralelo, com controle de timeout e organização automática dos resultados em `Output_dados/`.
- A pasta `XFOIL/` contém o executável e os recursos do solver, enquanto `arquivos_xfoil/` e `_temp_xfoil/` guardam arquivos temporários e resultados intermediários.

## Papel no projeto

Essa pasta é a interface de simulação da aerodinâmica dos perfis. Ela produz as respostas de coeficientes aerodinâmicos como `CL`, `CD` e `CM`, que se tornam a fonte de verdade para validação, ETL e treinamento dos modelos.

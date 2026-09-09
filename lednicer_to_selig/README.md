# Pasta `lednicer_to_selig`

Esta pasta implementa a conversão de arquivos de aerofólio no formato Lednicer para o formato Selig/GeoDat, adotado pelo fluxo de simulação do projeto.

## O que foi feito

- O script `lednicer_convert.py` recebe arquivos `.dat` no formato Lednicer, realiza limpeza de nomes, análise de contagem de pontos e leitura estruturada dos pontos do perfil.
- A função `limpar_nome()` normaliza os nomes dos perfis para evitar caracteres inválidos em nomes de arquivos.
- A rotina `ler_dat()` e o conjunto de auxiliares identificam padrões do formato Lednicer e transformam a geometria para um diretório de saída compatível com Selig.
- A pasta `Airfoils_Selig/` contém as geometrias convertidas e o relatório `relatorio_conversao.csv` com o resultado da conversão.

## Papel no projeto

Essa etapa cria a base de perfis em um formato uniforme e legível pelo XFOIL e pelo restante do pipeline, permitindo que os aerofólios sejam avaliados de forma consistente.

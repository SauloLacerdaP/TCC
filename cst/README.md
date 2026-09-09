# Pasta `cst`

Esta pasta concentra a implementação do método de reconstrução geométrica de perfis aerodinâmicos por funções CST (Class-Shape Transformation).

## O que foi feito

- O script `cst_automatizado.py` organiza a geração e reconstrução de perfis a partir de coeficientes CST, com foco em validar a forma do aerofólio e no uso de matrizes de Bernstein e funções de classe.
- O script `cst_automatizado_UIUC.py` atua como uma extensão ou variação automatizada do fluxo CST, com suporte ao conjunto de perfis UIUC e ao processamento de arquivos `.dat` para geração de formas reconstruídas.
- A subpasta `cst_output/` guarda os resultados de geometria reconstruída e os arquivos derivados da conversão CST.

## Papel no projeto

A pasta `cst` é responsável por transformar a descrição paramétrica dos perfis em coordenadas geométricas reais, produzindo representações físicas que podem ser avaliadas em XFOIL e usadas como base para otimização, treinamento de modelos e diagnósticos geométricos.

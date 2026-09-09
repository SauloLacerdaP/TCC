# Pasta `Otimizador`

A pasta `Otimizador` concentra o fluxo de otimização geométrica e de análise dos aerofólios candidatos.

## O que foi feito

- Os scripts `otimizacao_CL_alpha6_Re250k_ALTO.py` e `otimizacao_CD_alpha6_Re250k_BAIXO.py` usam modelos substitutos para buscar geometrias CST promissoras com objetivos específicos de alto `CL` ou baixo `CD`.
- O arquivo `diagnostico_geometrico_camber_base.py` e os scripts de diagnóstico em `analise/` avaliam resíduos geométricos, curvatura, suavidade e qualidade das soluções otimizadas.
- Os scripts de plotagem, como `plot_curvas_otimizado.py`, `plot_otimizados.py`, e demais arquivos em `analise/`, geram visualizações das curvas, comparação de soluções e diagnóstico final.
- A pasta `Output_dados/` armazena os resultados gerados pela otimização, métricas e diagnósticos.

## Papel no projeto

A otimização aproveita a base de modelos de machine learning e a simulação em XFOIL para explorar geometrias de aerofólios sob restrições geométricas e aerodinâmicas, com o objetivo de descobrir perfis com desempenho superior em condições específicas de Reynolds e ângulo de ataque.

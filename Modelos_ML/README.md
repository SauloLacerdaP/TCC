# Pasta `Modelos_ML`

A pasta `Modelos_ML` reúne os modelos de regressão e aprendizagem de máquina usados para prever coeficientes aerodinâmicos a partir da base de perfis parametrizados.

## O que foi feito

- Os scripts `reg_linear.py`, `random_forest.py`, `xgboost_model.py`, `xgboost_model_groupkfold.py`, `xgb_alpha.py`, `check_alpha.py` e `val_split.py` implementam diferentes abordagens de modelagem.
- Em geral, os arquivos contemplam carregamento do dataset preparado, treinamento de regressores, avaliação de métricas como RMSE, R² e erro por perfil, além de salvamento de artefatos com `joblib`.
- O conjunto inclui comparações de regressão linear, Random Forest, XGBoost e estratégias de validação por grupo, bem como análise de desempenho por faixa de `alpha` e por divisão de dados.

## Papel no projeto

A pasta é o núcleo preditivo do trabalho: ela converte as variáveis geométricas e aerodinâmicas em modelos capazes de estimar `CL`, `CD` e `CM` para novos perfis e condições de voo.

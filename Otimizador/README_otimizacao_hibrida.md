# Otimizador híbrido de aerofólio CST — XGBoost + Differential Evolution + XFOIL

## 1. Objetivo da nova versão

Esta versão altera a arquitetura da otimização após a análise do candidato 14.

O problema identificado foi que o modelo surrogate apresentou bom desempenho global e bom comportamento nos perfis de treinamento próximos, mas o Differential Evolution encontrou uma combinação nova de coeficientes CST para a qual o XGBoost simultaneamente superestimou `CL` e subestimou `CD`. Como a função objetivo é `CL/CD`, esses dois erros atuaram na mesma direção e produziram uma eficiência prevista artificialmente elevada.

A nova versão, portanto, deixa de tratar o resultado de máximo `CL/CD` do XGBoost como o ótimo aerodinâmico final. O surrogate passa a ser utilizado como ferramenta de busca e triagem, enquanto o XFOIL determina o ranking físico final entre os candidatos selecionados.

O fluxo passa a ser:

```text
Base XFOIL de treinamento
        ↓
Modelos XGBoost de CL, CD e CM
        ↓
Differential Evolution otimiza somente 14 CST
        ↓
Para cada geometria: polar surrogate de 0° a 12°
        ↓
max(CL/CD) previsto pelo surrogate
        ↓
Pool das melhores geometrias
        ↓
Seleção Top-N de geometrias distintas
        ↓
Polar completa no XFOIL de 0° a 12°
        ↓
max(CL/CD) XFOIL de cada geometria
        ↓
Ranking físico final
        ↓
Aerofólio otimizado
```

---

## 2. Mudança principal: alpha deixa de ser variável do Differential Evolution

Na versão anterior, o vetor de decisão era composto por 15 variáveis:

```text
Au0 ... Au6
Al0 ... Al6
alpha
```

Na nova versão, o Differential Evolution otimiza somente:

```text
Au0 ... Au6
Al0 ... Al6
```

Logo, o vetor de decisão possui 14 dimensões.

O ângulo de ataque não é fixado. Para cada geometria avaliada pelo Differential Evolution, o XGBoost calcula uma polar surrogate completa na grade:

```text
alpha = 0.00° até 12.00°
passo = 0.25°
```

Isso corresponde a 49 condições de alpha por geometria.

A função objetivo utilizada pelo Differential Evolution passa a ser:

$$
\max_{\mathbf{A}}
\left[
\max_{\alpha \in [0,12]}
\frac{C_L(\mathbf{A},\alpha)}{C_D(\mathbf{A},\alpha)}
\right]
$$

em que:

$$
\mathbf{A} =
[Au_0,\ldots,Au_6,Al_0,\ldots,Al_6]
$$

Assim, o `alpha` ótimo previsto é uma **saída da avaliação da geometria**, e não uma variável independente do Differential Evolution.

---

## 3. Polar surrogate vetorizada

Para evitar 49 chamadas Python independentes por geometria, a nova versão monta todas as condições de alpha em um único `DataFrame` e faz uma chamada vetorizada para cada modelo:

- uma chamada ao XGBoost de `CL`;
- uma chamada ao XGBoost de `CD`;
- uma chamada ao XGBoost de `CM`.

O resultado é uma polar surrogate contendo, para cada alpha:

- `CL_pred`;
- `CD_pred`;
- `CM_pred`;
- `CL_CD_pred`;
- limite local de `CD`;
- limite inferior local de `CM`;
- limite superior local de `CM`;
- indicação se o ponto é admissível.

Apenas os pontos que passam pelas restrições locais entram na seleção do máximo `CL/CD` surrogate.

---

## 4. Restrições preservadas

A nova lógica **não remove as restrições já implementadas**. Permanecem:

1. fechamento do bordo de ataque pela formulação CST;
2. fechamento do bordo de fuga com `DeltaTE_upper = DeltaTE_lower = 0`;
3. não cruzamento entre extradorso e intradorso;
4. espessura máxima dentro da faixa observada no treinamento;
5. limites individuais dos 14 coeficientes CST;
6. distância ao vizinho geométrico mais próximo do conjunto de treinamento (`d1`) no espaço CST padronizado;
7. Reynolds dentro do domínio utilizado, com `Re = 250000` nesta configuração;
8. alpha avaliado somente entre 0° e 12°;
9. piso local de `CD` condicionado a Reynolds e alpha;
10. faixa local admissível de `CM` condicionada a Reynolds e alpha;
11. população inicial construída a partir de geometrias reais do treinamento;
12. penalização gradual para candidatos geometricamente inválidos.

A distância geométrica continua sendo calculada somente a partir das **geometrias CST únicas** do treinamento.

---

## 5. Restrições que não foram adicionadas

A análise de erro e densidade não mostrou benefício em substituir ou complementar `d1` por métricas como:

- `d5`;
- `d10`;
- `d20`;
- média dos k vizinhos;
- Mahalanobis local.

Também não foram criados limites especiais para `Au6`, `Al3` ou `Al5` apenas porque esses parâmetros apareceram como mais extremos no candidato 14.

Essas características foram úteis para diagnosticar a falha local do surrogate, mas não constituíram evidência suficiente para criar novas restrições físicas ou geométricas.

---

## 6. Pré-cálculo dos limites locais de CD e CM

Como Reynolds é fixo e a grade de alpha é conhecida antes da otimização, os limites locais de `CD` e `CM` são calculados apenas uma vez na inicialização do problema.

Para cada um dos 49 valores de alpha são armazenados:

```text
CD_local_min
CM_local_min
CM_local_max
```

Durante as milhares de avaliações do Differential Evolution, esses valores são reutilizados. Isso evita repetir buscas no `train.csv` para cada geometria e reduz o custo computacional da nova polar surrogate.

---

## 7. Registro dos melhores candidatos

Durante o Differential Evolution, o código mantém um pool das melhores geometrias encontradas pelo surrogate.

Configuração padrão:

```python
CANDIDATE_POOL_SIZE = 100
TOP_SURROGATE_TO_VALIDATE = 20
```

Para cada geometria armazenada são registrados:

```text
14 coeficientes CST
alpha_pred_opt
CL_pred_opt
CD_pred_opt
CM_pred_opt
CL_CD_pred_opt
```

O `alpha_pred_opt` representa o ponto da polar surrogate onde aquela geometria atingiu o maior `CL/CD` admissível.

---

## 8. Seleção de geometrias distintas

Na versão anterior, uma combinação `geometria + alpha` podia ser considerada um candidato diferente.

Na nova versão, o candidato é somente a geometria.

Antes da validação XFOIL, as geometrias são comparadas no mesmo espaço CST padronizado utilizado na análise de domínio.

A configuração:

```python
CANDIDATE_DISTINCT_DISTANCE_STD = 0.10
```

serve apenas para evitar enviar ao XFOIL várias cópias praticamente idênticas da mesma solução.

Esse parâmetro **não é uma nova restrição física do espaço de busca**. Ele é aplicado somente na formação da lista Top-N para validação.

---

## 9. Validação XFOIL por polar completa

Cada uma das geometrias Top-N é exportada para `.dat` e avaliada no XFOIL utilizando:

```text
Re = 250000
Mach = 0.1
alpha = 0° até 12°
passo = 0.25°
```

O comando principal enviado ao XFOIL é equivalente a:

```text
ASEQ 0.0 12.0 0.25
```

A polar inteira é lida e o código calcula:

$$
\left(\frac{C_L}{C_D}\right)_{XFOIL,max}
$$

para cada geometria.

São armazenados:

```text
alpha_xfoil_opt
CL_xfoil_opt
CD_xfoil_opt
CM_xfoil_opt
CL_CD_xfoil_opt
n_xfoil_points
```

Por padrão, uma geometria precisa possuir pelo menos:

```python
XFOIL_MIN_CONVERGED_POINTS = 20
```

pontos válidos na polar para entrar no ranking final.

---

## 10. Ranking final

O ranking final deixa de ser determinado pelo `CL/CD` do XGBoost.

O critério passa a ser:

$$
i^* = \arg\max_i
\left[
\max_{\alpha}
\left(\frac{C_L}{C_D}\right)_{XFOIL,i}
\right]
$$

Portanto:

- o XGBoost determina quais geometrias merecem ser verificadas;
- o XFOIL determina qual delas é efetivamente a melhor;
- o alpha ótimo final também é determinado pela polar XFOIL.

O arquivo `aerofolio_otimizado.dat` corresponde à geometria vencedora desse ranking físico.

---

## 11. Comparação entre ótimo do surrogate e ótimo do XFOIL

A tabela de validação registra tanto o ótimo previsto quanto o ótimo confirmado:

```text
alpha_pred_opt
CL_CD_pred_opt
alpha_xfoil_opt
CL_CD_xfoil_opt
delta_alpha_opt_deg
erro_CL_CD_max_pct
```

Além disso, o XGBoost é novamente avaliado no alpha ótimo encontrado pelo XFOIL, permitindo separar dois tipos de erro:

1. erro na escolha do alpha ótimo;
2. erro na previsão aerodinâmica da geometria naquele alpha.

São registrados, por exemplo:

```text
CL_pred_at_xfoil_opt
CD_pred_at_xfoil_opt
CM_pred_at_xfoil_opt
CL_CD_pred_at_xfoil_opt
erro_CL_no_alpha_xfoil_pct
erro_CD_no_alpha_xfoil_pct
```

Isso fornece uma análise mais completa da qualidade do surrogate durante a otimização.

---

## 12. Principais arquivos gerados

### Resultado final

```text
Output_dados/otimizacao/resultado_otimizacao.csv
Output_dados/otimizacao/resultado_otimizacao.json
Output_dados/otimizacao/aerofolio_otimizado.dat
```

### Polar surrogate

```text
polar_surrogate_melhor_de.csv
polar_surrogate_aerofolio_otimizado.csv
```

### Validação e ranking XFOIL

```text
validacao_candidatos_xfoil_polar.csv
ranking_candidatos_xfoil.csv
```

### Arquivos individuais de cada candidato

Dentro de:

```text
Output_dados/otimizacao/validacao_candidatos_xfoil/candidato_XXX/
```

são salvos:

```text
aerofolio.dat
polar.txt
polar_xfoil_processada.csv
comandos_xfoil.txt
stdout.txt
stderr.txt
```

---

## 13. Parâmetros principais que podem ser ajustados

### Grade surrogate

```python
ALPHA_MIN = 0.0
ALPHA_MAX = 12.0
SURROGATE_ALPHA_STEP = 0.25
```

### Polar XFOIL

```python
XFOIL_ALPHA_MIN = 0.0
XFOIL_ALPHA_MAX = 12.0
XFOIL_ALPHA_STEP = 0.25
```

### Quantidade de candidatos

```python
CANDIDATE_POOL_SIZE = 100
TOP_SURROGATE_TO_VALIDATE = 20
```

### Diversidade geométrica do Top-N

```python
CANDIDATE_DISTINCT_DISTANCE_STD = 0.10
```

### Differential Evolution

```python
POP_SIZE = 20
MAX_ITER = 500
TOL = 1e-7
SEED = 42
```

---

## 14. Interpretação metodológica

A principal mudança conceitual pode ser resumida da seguinte forma.

### Versão anterior

O surrogate era utilizado para responder diretamente:

> Qual geometria e qual alpha maximizam `CL/CD`?

O resultado era então verificado no XFOIL naquele alpha selecionado pelo próprio modelo.

### Nova versão

O surrogate responde:

> Quais geometrias parecem mais promissoras considerando sua melhor condição prevista entre 0° e 12°?

O XFOIL responde posteriormente:

> Entre essas geometrias, qual possui realmente o maior `CL/CD` e em qual alpha isso ocorre?

Essa separação reduz a dependência do resultado final de erros locais do surrogate e preserva a principal vantagem computacional do modelo de aprendizado de máquina: realizar rapidamente a triagem de um grande número de geometrias.

---

## 15. Resumo da arquitetura final

```text
14 CST
  ↓
Validação geométrica
  ↓
Distância d1 ao treinamento
  ↓
Polar XGBoost vetorizada — 49 alphas
  ↓
Filtros locais de CD e CM
  ↓
max(CL/CD) surrogate por geometria
  ↓
Differential Evolution
  ↓
Pool Top-100
  ↓
Top-20 geometrias distintas
  ↓
Polar XFOIL completa — 49 alphas
  ↓
max(CL/CD) XFOIL por geometria
  ↓
Ranking XFOIL
  ↓
Ótimo final confirmado
```

A implementação correspondente está no arquivo:

```text
otimizacao_aerofolio_hibrida.py
```

# Resumo da Análise do Candidato 14

A análise teve como objetivo entender por que o **candidato 14**, selecionado pelo algoritmo de otimização com XGBoost, apresentou desempenho previsto muito superior ao obtido posteriormente no XFOIL.

O XGBoost previa, em aproximadamente **α = 3,93° e Re = 250.000**:

$$
C_L = 1,3799,\qquad
C_D = 0,00857,\qquad
\frac{C_L}{C_D} = 161,01
$$

Enquanto a validação no XFOIL resultou em:

$$
C_L = 1,0068,\qquad
C_D = 0,01602,\qquad
\frac{C_L}{C_D} = 62,85
$$

Portanto, o surrogate **superestimou $C_L$ e, principalmente, subestimou $C_D$**, provocando uma grande superestimação da eficiência aerodinâmica.

## 1. Distância ao domínio de treinamento

Foi inicialmente investigado se o erro estava associado à distância geométrica entre o candidato e os perfis utilizados no treinamento.

A análise do conjunto de teste mostrou que a distância ao vizinho mais próximo (`d1`) possui correlação com o erro, principalmente para $C_L$, porém apenas uma relação moderada para $C_D$:

$$
\rho_{\text{Spearman}}(d_1, erro_{C_D}) = 0,373
$$

Também foram avaliadas métricas de densidade baseadas nos 3, 5, 10 e 20 vizinhos geométricos distintos. Nenhuma apresentou relação com o erro de $C_D$ superior à obtida com `d1`. Por exemplo:

$$
\rho(d_{10}, erro_{C_D}) = 0,286
$$

$$
\rho(\overline{d}_{10}, erro_{C_D}) = 0,314
$$

Assim, **a baixa densidade geométrica local não explica adequadamente o erro do candidato 14**.

## 2. Análise dos vizinhos do candidato

Foram então identificados os 10 perfis CST mais próximos do candidato.

O vizinho mais próximo foi o `sg6043`, com distância padronizada:

$$
d_1 = 2,249
$$

Para esse perfil, em **α = 4° e Re = 250.000**:

$$
C_{D,\text{XFOIL}} = 0,01099
$$

$$
C_{D,\text{ML}} = 0,01119
$$

correspondendo a um erro de apenas **1,79%**.

De maneira geral, os 10 vizinhos apresentaram:

$$
C_{D,\text{real}} = 0,00972 \text{ a } 0,01390
$$

com mediana de aproximadamente:

$$
C_D = 0,01181
$$

O modelo também reproduziu satisfatoriamente esses perfis conhecidos. Isso mostra que **o XGBoost consegue prever adequadamente geometrias próximas**, apesar de apresentar grande erro para a nova geometria produzida pelo otimizador.

## 3. Combinação dos parâmetros CST

A comparação dos 14 coeficientes CST do candidato com seus 10 vizinhos mostrou que apenas **3 parâmetros estavam fora do envelope local**:

- `Au6`: 0,5531, com $z = 2,62$;
- `Al3`: 0,1903, com $z = 1,62$;
- `Al5`: 0,1267, com $z = 1,39$.

O `Au6` foi o parâmetro individualmente mais extremo. Outros coeficientes permaneceram dentro dos intervalos dos vizinhos, embora alguns apresentassem deslocamentos em relação às médias locais.

Portanto, o candidato **não constitui uma geometria completamente isolada ou fora do domínio global de treinamento**. O que se observa é uma **combinação localmente incomum de parâmetros CST**, ainda dentro das restrições impostas ao otimizador.

## 4. Conclusão

A análise indica que o problema do candidato 14 **não pode ser atribuído exclusivamente à distância ao conjunto de treinamento nem à baixa densidade de perfis próximos**.

O resultado é mais consistente com a exploração, pelo algoritmo de otimização, de uma **região local de erro do surrogate**.

O Differential Evolution encontrou uma nova combinação de parâmetros CST para a qual o XGBoost simultaneamente:

- **superestimou $C_L$**;
- **subestimou $C_D$**.

Como a função objetivo é:

$$
\max\left(\frac{C_L}{C_D}\right)
$$

esses dois erros atuaram na mesma direção, produzindo um valor artificialmente elevado de $C_L/C_D$.

Isso **não invalida o XGBoost como modelo surrogate**, mas demonstra que um bom desempenho global no conjunto de teste não garante que o ótimo encontrado pelo surrogate seja também o ótimo físico.

Por esse motivo, a estratégia mais robusta é utilizar o ML para realizar a busca computacionalmente eficiente e, posteriormente, **validar os melhores candidatos com uma polar completa no XFOIL**, utilizando o solver aerodinâmico para determinar o ranking e o ótimo final.

A metodologia resultante pode ser representada por:

$$
\boxed{
\text{Base XFOIL}
\rightarrow
\text{XGBoost}
\rightarrow
\text{Differential Evolution}
\rightarrow
\text{Top-N candidatos}
\rightarrow
\text{XFOIL full polar}
\rightarrow
\text{Ótimo final}
}
$$
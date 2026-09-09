r"""

otimizacao_aerofolio.py

Otimização de aerofólio CST usando modelos substitutos XGBoost.

Objetivo:

    buscar geometrias CST promissoras com XGBoost usando uma única execução ampla

    do Differential Evolution e confirmar o ótimo final

    com polar completa no XFOIL. Para cada geometria, o surrogate avalia exclusivamente alpha = 6 graus e Re = 250.000, usando CL como função objetivo.

Modelos:

    C:\Repositorios\TCC\Output_dados\resultados_xgboost\xgboost_CL.pkl

    C:\Repositorios\TCC\Output_dados\resultados_xgboost\xgboost_CD.pkl

    C:\Repositorios\TCC\Output_dados\resultados_xgboost\xgboost_CM.pkl

Restrições implementadas:

1) fechamento do bordo de ataque (inerente à classe CST N1=0.5, N2=1.0);

2) fechamento do bordo de fuga (DeltaTE_upper = DeltaTE_lower = 0);

3) não cruzamento entre extradorso e intradorso;

4) espessura máxima dentro da faixa observada na base de referência;

5) limites individuais dos coeficientes CST conforme base de referência;

6) distância geométrica ao domínio CST do treinamento;

7) Reynolds entre 200.000 e 300.000 (por padrão fixo em 250.000);

8) alpha fixado em 6 graus durante toda a otimização surrogate;

9) CD previsto não inferior ao menor CD positivo observado na base;

10) predição de CL somente para candidatos válidos;

11) população inicial baseada em geometrias reais do treino;

12) penalização gradual para candidatos inválidos.

A restrição de distância geométrica é calculada no espaço CST padronizado.

O limite é obtido automaticamente pela distribuição das distâncias de

cada perfil de referência ao seu vizinho mais próximo (percentil configurável).

IMPORTANTE:

- Para a restrição de domínio ser rigorosamente "distância ao TREINO",

  informe TRAIN_REFERENCE_CSV apontando para um CSV contendo apenas os

  perfis usados no treinamento do XGBoost.

- Se esse arquivo não for encontrado, o código tenta usar database_ml.csv

  e emite um aviso.

"""

from __future__ import annotations

import json

import math

import subprocess

import shutil

import warnings

from pathlib import Path

import joblib

import numpy as np

import pandas as pd

from scipy.optimize import differential_evolution

from scipy.spatial import cKDTree

from sklearn.preprocessing import StandardScaler



# ======================================================================

# CONFIGURAÇÕES

# ======================================================================

# Resolve a raiz do repositório pelo próprio local do arquivo do script.

# Isso mantém o código portátil de máquina para máquina.

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "Output_dados" / "resultados_xgboost"

MODEL_CL = MODEL_DIR / "xgboost_CL.pkl"

MODEL_CD = MODEL_DIR / "xgboost_CD.pkl"

MODEL_CM = MODEL_DIR / "xgboost_CM.pkl"

# Idealmente: CSV contendo SOMENTE os perfis do conjunto de treinamento.

TRAIN_REFERENCE_CSV: Path | None = None

# Alternativas procuradas automaticamente caso TRAIN_REFERENCE_CSV = None.

# Base de referência disponível no repositório.

REFERENCE_CSV_CANDIDATES = [

    ROOT / "Output_dados" / "ml_preparado" / "train.csv",

    ROOT / "Output_dados" / "ml_preparado" / "perfis_train.csv",

    ROOT / "Output_dados" / "database_train.csv",

    ROOT / "Output_dados" / "dados_treino.csv",

    ROOT / "Output_dados" / "database_ml.csv",

    ROOT / "database_ml.csv",

]

# Cria a subpasta dentro de Output_dados para os arquivos do otimizador.

OUTPUT_DIR = ROOT / "Output_dados" / "otimizacao_CL"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Condição aerodinâmica

RE_FIXED = 250_000.0

RE_MIN = 200_000.0

RE_MAX = 300_000.0

# Alpha NÃO é mais uma variável do Differential Evolution.

# O DE otimiza somente os 14 coeficientes CST. Para cada geometria,

# o XGBoost calcula uma polar surrogate completa nesta grade.
# A grade do surrogate usa apenas alphas inteiros, iguais aos usados no treinamento.

OPTIMIZE_ALPHA = False

ALPHA_FIXED = 6.0

ALPHA_MIN = 0.0

ALPHA_MAX = 12.0

SURROGATE_ALPHA_STEP = 1.0

SURROGATE_ALPHA_GRID = np.arange(

    ALPHA_MIN,

    ALPHA_MAX + 0.5 * SURROGATE_ALPHA_STEP,

    SURROGATE_ALPHA_STEP,

    dtype=float,

)

# CST clássico para aerofólio

N1 = 0.5

N2 = 1.0

# Discretização geométrica

N_GEOM_POINTS = 301

# Distância geométrica:

# percentil das distâncias "perfil de treino -> vizinho de treino mais próximo".

# 95% é conservador sem ser excessivamente restritivo.

DISTANCE_PERCENTILE = 80.0

# O P80 continua sendo a referência estatística do conjunto de treino.
# O s1223 está a aproximadamente 4.015 unidades padronizadas, enquanto
# P80 ~= 2.364. O multiplicador 1.80 fornece limite ~= 4.255:
# suficiente para manter a região do s1223 acessível, mas elimina parte
# da extrapolação excessiva observada com 2.25 x P80 (~5.319).
DISTANCE_LIMIT_MULTIPLIER = 1.80

# Expansão dos bounds individuais CST em relação à amplitude observada
# no conjunto de treino. O s1223 NÃO é usado como seed.
CST_BOUND_EXPANSION = 0.75

# Pequena tolerância apenas numérica.

GEOM_TOL = 1e-8

# Restrição morfológica de camber.
# Diagnóstico da base (193 perfis):
#   P95 = 6.6147% c
#   P99 = 8.7737% c
#   máximo observado = 10.2175% c
#   s1223 = 8.5971% c
#
# Usamos 10.3% c: ligeiramente acima do maior valor observado na base,
# preservando s1223/e423 e rejeitando geometrias artificiais como a
# solução exploratória anterior (~14.7% c).
MAX_ABS_CAMBER = 0.09

# Regularidade geométrica por curvatura.
# Avalia somente o trecho interno da corda para evitar efeitos naturais do BA/BF.
CURVATURE_X_MIN = 0.02
CURVATURE_X_MAX = 0.98
CURVATURE_N_POINTS = 2001
CURVATURE_SIGN_THRESHOLD_FRACTION = 0.02

# Limites derivados do diagnóstico geométrico e mantidos iguais aos do CD.
MAX_CURVATURE_INVERSIONS_UPPER = 2
MAX_CURVATURE_INVERSIONS_LOWER = 3

# ======================================================================
# RESTRIÇÕES GEOMÉTRICAS — VERSÃO FINAL
# ======================================================================
# 1) Fechamento BA/BF:
#    inerente ao CST N1=0.5, N2=1.0 e DeltaTE=0.
#
# 2) Não cruzamento:
#    exige yu(x) >= yl(x) ao longo de toda a corda.
#
# 3) Espessura máxima:
#    deve permanecer dentro da faixa efetivamente observada na base de
#    referência. Não é escolhido um limite aerodinâmico arbitrário.
#
# 4) Limites CST expandidos:
#    os limites individuais do treino são ampliados para permitir explorar
#    regiões legítimas próximas de perfis de alto CL que ficaram fora do
#    hipercubo original. A expansão NÃO substitui a validação geométrica.
#
# 5) Distância ao domínio:
#    limite = P80 da distância ao vizinho de treino * 1.80. Esse valor mantém
#    acessível a região do s1223 (~4.015 no espaço padronizado), mas impede
#    extrapolação multivariada ainda mais distante (~4.255 para a base atual).
#
# 6) Camber máximo absoluto:
#    |camber|max <= 9% c. O valor deriva do máximo observado na base
#    (~10.22% c), acrescido apenas de pequena tolerância numérica.
#
# 7) NÃO foram adicionadas restrições rígidas de curvatura, número de
#    inversões ou envelopes locais de camber/espessura. O diagnóstico mostrou
#    que elas não separam de forma consistente candidatos que convergem e não
#    convergem no XFOIL e poderiam excluir candidatos fisicamente válidos.
#
# 8) CD e CM:
#    permanecem SEM restrição nesta otimização de CL. O objetivo é puro:
#    maximizar CL em alpha=6 graus e Re=250000. CD e CM são diagnósticos.
#
# 9) XFOIL:
#    não é uma restrição durante o DE. É a validação física final dos Top-N.
#    Candidatos sem polar válida são descartados do ranking final.
# ======================================================================

# Otimizador

SEED = 42

# ÚNICA execução deliberadamente mais ampla e exploratória.
POP_SIZE = 20
MAX_ITER = 100
TOL = 1e-7

POLISH = True

WORKERS = 1  

# Inicialização factível do Differential Evolution

# A população nasce próxima de geometrias reais da base de treino.

INIT_PERTURBATION_STD = 0.18

# Fração da população inicial gerada globalmente nos bounds ampliados.
INIT_GLOBAL_FRACTION = 0.30
INIT_HIGH_CL_FRACTION = 0.70
HIGH_CL_TOP_N_PROFILES = 20
HIGH_CL_LOCAL_STD = 0.12
HIGH_CL_WIDE_STD = 0.35
HIGH_CL_WIDE_FRACTION = 0.35

INIT_MAX_ATTEMPTS_PER_MEMBER = 300

# Penalizações graduais para guiar o algoritmo de volta à região factível.

PENALTY_BASE = 1e5

PENALTY_SCALE = 1e4

# Penalização SUAVE por proximidade da fronteira do domínio CST.
# Até 70% do limite de distância não há penalização. Acima disso,
# o custo cresce quadraticamente e chega a DISTANCE_SOFT_WEIGHT
# exatamente no limite duro. O limite P80 continua sendo uma restrição
# rígida em validar_geometria().
# Sem penalização suave nesta versão exploratória.
# Apenas o limite duro ampliado permanece.
DISTANCE_SOFT_START = 1.0
DISTANCE_SOFT_WEIGHT = 0.0
DISTANCE_SOFT_POWER = 2.0


# Na otimização isolada de CL, CD e CM permanecem como diagnóstico,
# mas NÃO invalidam candidatos.
USE_CD_FLOOR_CONSTRAINT = False
USE_CM_CONSTRAINT = False

# Proteção local contra exploração artificial de CD

CD_LOCAL_ALPHA_WINDOW = 1.0

CD_LOCAL_RE_WINDOW = 25_000.0

CD_LOCAL_PERCENTILE = 5.0

CD_LOCAL_MIN_POINTS = 20

# Limites locais de CM para evitar momentos muito fora do comportamento

# observado na base em condições aerodinâmicas semelhantes.

CM_LOCAL_ALPHA_WINDOW = 1.0

CM_LOCAL_RE_WINDOW = 25_000.0

CM_LOCAL_LOWER_PERCENTILE = 5.0

CM_LOCAL_UPPER_PERCENTILE = 95.0

CM_LOCAL_MIN_POINTS = 20

# Validação final dos melhores candidatos no XFOIL

XFOIL_EXE = Path(r"C:\Repositorios\TCC\xfoil\XFOIL\xfoil.exe")

MACH = 0.1

XFOIL_ITER = 300

# Polar completa de cada candidato final no XFOIL.
# Aqui mantemos passo fino de 0.25° para localizar melhor o ótimo físico.

XFOIL_TIMEOUT = 60
XFOIL_ALPHA_MIN = 0.0

XFOIL_ALPHA_MAX = 12.0

XFOIL_ALPHA_STEP = 0.25

XFOIL_MIN_CONVERGED_POINTS = 20

TOP_SURROGATE_TO_VALIDATE = 20

CANDIDATE_POOL_SIZE_PER_SEED = 2000

# Distância mínima, no espaço CST padronizado, para considerar duas

# geometrias diferentes durante a seleção Top-N. Não é restrição física;

# serve apenas para evitar validar cópias praticamente idênticas.

FARTHEST_POINT_MIN_DISTANCE_STD = 1e-8

XFOIL_VALIDATION_DIR = OUTPUT_DIR / "validacao_candidatos_xfoil"

XFOIL_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)



CST_UPPER = [f"Au{i}" for i in range(7)]

CST_LOWER = [f"Al{i}" for i in range(7)]

CST_FEATURES = CST_UPPER + CST_LOWER

MODEL_FEATURES_DEFAULT = (

    CST_UPPER

    + CST_LOWER

    + ["DeltaTE_upper", "DeltaTE_lower", "Re", "alpha"]

)



# ======================================================================

# UTILIDADES

# ======================================================================

def localizar_csv_referencia() -> Path:

    if TRAIN_REFERENCE_CSV is not None:

        p = Path(TRAIN_REFERENCE_CSV)

        if not p.exists():

            raise FileNotFoundError(

                f"TRAIN_REFERENCE_CSV foi informado, mas não existe:\n{p}"

            )

        return p

    for p in REFERENCE_CSV_CANDIDATES:

        if p.exists():

            if p.name.lower() == "database_ml.csv":

                warnings.warn(

                    "\nATENÇÃO: não encontrei um CSV explicitamente contendo apenas "

                    "o conjunto de treinamento. Estou usando database_ml.csv como "

                    "referência geométrica. Para a versão final do TCC, prefira "

                    "apontar TRAIN_REFERENCE_CSV para os perfis efetivamente usados "

                    "no TREINO do XGBoost.\n"

                )

            return p

    raise FileNotFoundError(

        "Não encontrei o CSV de referência.\n"

        "Defina TRAIN_REFERENCE_CSV no início do script apontando para o "

        "arquivo que contém os perfis do conjunto de treinamento."

    )



def carregar_modelos():

    for p in (MODEL_CL, MODEL_CD, MODEL_CM):

        if not p.exists():

            raise FileNotFoundError(f"Modelo não encontrado: {p}")

    return (

        joblib.load(MODEL_CL),

        joblib.load(MODEL_CD),

        joblib.load(MODEL_CM),

    )



def bernstein(n: int, i: int, x: np.ndarray) -> np.ndarray:

    return math.comb(n, i) * (x ** i) * ((1.0 - x) ** (n - i))



def class_function(x: np.ndarray) -> np.ndarray:

    return (x ** N1) * ((1.0 - x) ** N2)



def cst_surface(

    x: np.ndarray,

    coeffs: np.ndarray,

    delta_te: float = 0.0,

) -> np.ndarray:

    """

    y(x) = C(x) * S(x) + x * DeltaTE

    """

    coeffs = np.asarray(coeffs, dtype=float)

    n = len(coeffs) - 1

    S = np.zeros_like(x, dtype=float)

    for i, a_i in enumerate(coeffs):

        S += a_i * bernstein(n, i, x)

    return class_function(x) * S + x * float(delta_te)



def reconstruir_geometria(

    au: np.ndarray,

    al: np.ndarray,

    n_points: int = N_GEOM_POINTS,

):

    # distribuição cosine-spaced: maior resolução perto de BA/BF

    beta = np.linspace(0.0, np.pi, n_points)

    x = 0.5 * (1.0 - np.cos(beta))

    # BF fechado explicitamente

    yu = cst_surface(x, au, delta_te=0.0)

    yl = cst_surface(x, al, delta_te=0.0)

    return x, yu, yl



def resolver_feature_names(model):

    if hasattr(model, "feature_names_in_"):

        return list(model.feature_names_in_)

    # Alguns objetos XGBoost guardam isso no booster

    if hasattr(model, "get_booster"):

        try:

            names = model.get_booster().feature_names

            if names:

                return list(names)

        except Exception:

            pass

    return MODEL_FEATURES_DEFAULT.copy()



def montar_dataframe_modelo(

    model,

    au: np.ndarray,

    al: np.ndarray,

    re_value: float,

    alpha: float,

) -> pd.DataFrame:

    values = {

        **{f"Au{i}": float(au[i]) for i in range(7)},

        **{f"Al{i}": float(al[i]) for i in range(7)},

        "DeltaTE_upper": 0.0,

        "DeltaTE_lower": 0.0,

        "Re": float(re_value),

        "alpha": float(alpha),

    }

    feature_names = resolver_feature_names(model)

    faltantes = [f for f in feature_names if f not in values]

    if faltantes:

        raise KeyError(

            "O modelo solicita features que o otimizador não reconheceu: "

            + ", ".join(faltantes)

        )

    return pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)



# ======================================================================

# DOMÍNIO GEOMÉTRICO DO TREINAMENTO

# ======================================================================

class DominioGeometrico:

    def __init__(self, df: pd.DataFrame):

        missing = [c for c in CST_FEATURES if c not in df.columns]

        if missing:

            raise KeyError(

                "CSV de referência não contém todas as colunas CST: "

                + ", ".join(missing)

            )

        # Um perfil aparece em vários alpha/Re. A geometria só deve entrar uma vez.

        geom = (

            df[CST_FEATURES]

            .replace([np.inf, -np.inf], np.nan)

            .dropna()

            .drop_duplicates()

            .reset_index(drop=True)

        )

        if len(geom) < 3:

            raise ValueError("Poucos perfis geométricos válidos na referência.")

        self.geom = geom

        # Bounds observados no conjunto de treino, preservados para diagnóstico.
        self.observed_min_cst = geom.min()
        self.observed_max_cst = geom.max()

        # Bounds de BUSCA ampliados. O scaler/KDTree continuam ajustados
        # somente com as geometrias reais do treino.
        amplitude = self.observed_max_cst - self.observed_min_cst
        self.min_cst = self.observed_min_cst - CST_BOUND_EXPANSION * amplitude
        self.max_cst = self.observed_max_cst + CST_BOUND_EXPANSION * amplitude

        # Distância multivariada:

        # padroniza cada dimensão antes de calcular distância.

        self.scaler = StandardScaler()

        z = self.scaler.fit_transform(geom[CST_FEATURES].to_numpy(dtype=float))

        self.tree = cKDTree(z)

        # k=2: o primeiro vizinho é o próprio ponto (distância zero);

        # o segundo é o perfil geométrico distinto mais próximo.

        dists, _ = self.tree.query(z, k=2)

        nearest = dists[:, 1]

        self.nn_train = nearest

        self.distance_limit_base = float(
            np.percentile(nearest, DISTANCE_PERCENTILE)
        )
        self.distance_limit = float(
            self.distance_limit_base * DISTANCE_LIMIT_MULTIPLIER
        )

    def distancia_ao_treino(self, cst_vector: np.ndarray) -> float:

        z = self.scaler.transform(

            np.asarray(cst_vector, dtype=float).reshape(1, -1)

        )

        d, _ = self.tree.query(z, k=1)

        return float(d[0])

    def dentro_limites_individuais(self, cst_vector: np.ndarray) -> bool:

        v = np.asarray(cst_vector, dtype=float)

        mins = self.min_cst[CST_FEATURES].to_numpy(dtype=float)

        maxs = self.max_cst[CST_FEATURES].to_numpy(dtype=float)

        return bool(np.all(v >= mins - GEOM_TOL) and np.all(v <= maxs + GEOM_TOL))

    def bounds_scipy(self):

        return [

            (float(self.min_cst[c]), float(self.max_cst[c]))

            for c in CST_FEATURES

        ]



def calcular_limites_espessura(df: pd.DataFrame):

    """

    Faixa obtida DIRETAMENTE das geometrias observadas na referência.

    Não escolhe arbitrariamente 6%, 12%, 20% etc.

    """

    unique_geom = df[CST_FEATURES].drop_duplicates()

    thicknesses = []

    for row in unique_geom.itertuples(index=False):

        vals = np.asarray(row, dtype=float)

        au = vals[:7]

        al = vals[7:]

        _, yu, yl = reconstruir_geometria(au, al)

        t = yu - yl

        if np.all(np.isfinite(t)):

            thicknesses.append(float(np.max(t)))

    if not thicknesses:

        raise ValueError("Não consegui calcular espessuras na base de referência.")

    return (

        float(np.min(thicknesses)),

        float(np.max(thicknesses)),

        np.asarray(thicknesses),

    )



def calcular_cd_minimo_global(df: pd.DataFrame) -> float:

    """Retorna o menor CD positivo da base apenas como fallback global."""

    if "CD" not in df.columns:

        warnings.warn(

            "Coluna CD não encontrada. Usando CD_MIN_GLOBAL = 1e-6."

        )

        return 1e-6

    cd = pd.to_numeric(df["CD"], errors="coerce").to_numpy(dtype=float)

    cd = cd[np.isfinite(cd) & (cd > 0)]

    if len(cd) == 0:

        warnings.warn(

            "Nenhum CD positivo válido encontrado. Usando 1e-6."

        )

        return 1e-6

    return float(np.min(cd))



def calcular_cd_minimo_local(

    df: pd.DataFrame,

    re_value: float,

    alpha: float,

    cd_global_min: float,

) -> float:

    """

    Piso de CD condicionado à vizinhança de Reynolds e alpha.

    Usa um percentil baixo da distribuição local em vez do mínimo global.

    """

    if "CD" not in df.columns:

        return float(cd_global_min)

    re_col = next(

        (c for c in ("Re", "RE", "re", "Reynolds") if c in df.columns),

        None,

    )

    alpha_col = next(

        (c for c in ("alpha", "Alpha", "ALPHA") if c in df.columns),

        None,

    )

    if re_col is None or alpha_col is None:

        return float(cd_global_min)

    work = df[[re_col, alpha_col, "CD"]].copy()

    for c in (re_col, alpha_col, "CD"):

        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    work = work[work["CD"] > 0]

    if work.empty:

        return float(cd_global_min)

    selected = pd.DataFrame()

    for factor in (1.0, 1.5, 2.0, 3.0, 4.0):

        selected = work[

            (np.abs(work[alpha_col] - alpha)

             <= CD_LOCAL_ALPHA_WINDOW * factor)

            &

            (np.abs(work[re_col] - re_value)

             <= CD_LOCAL_RE_WINDOW * factor)

        ]

        if len(selected) >= CD_LOCAL_MIN_POINTS:

            break

    if selected.empty:

        return float(cd_global_min)

    local_floor = float(

        np.percentile(

            selected["CD"].to_numpy(dtype=float),

            CD_LOCAL_PERCENTILE,

        )

    )

    if not np.isfinite(local_floor) or local_floor <= 0:

        return float(cd_global_min)

    return max(float(cd_global_min), local_floor)



def calcular_limites_cm_local(

    df: pd.DataFrame,

    re_value: float,

    alpha: float,

):

    """

    Retorna (cm_min_local, cm_max_local) a partir da distribuição de CM

    em pontos próximos de Reynolds e alpha.

    A janela é ampliada gradualmente se houver poucos pontos. Os limites

    são definidos pelos percentis configurados, reduzindo a influência

    de extremos isolados da base.

    """

    if "CM" not in df.columns:

        return None, None

    re_col = next(

        (c for c in ("Re", "RE", "re", "Reynolds") if c in df.columns),

        None,

    )

    alpha_col = next(

        (c for c in ("alpha", "Alpha", "ALPHA") if c in df.columns),

        None,

    )

    if re_col is None or alpha_col is None:

        return None, None

    work = df[[re_col, alpha_col, "CM"]].copy()

    for c in (re_col, alpha_col, "CM"):

        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    if work.empty:

        return None, None

    selected = pd.DataFrame()

    for factor in (1.0, 1.5, 2.0, 3.0, 4.0):

        selected = work[

            (np.abs(work[alpha_col] - alpha)

             <= CM_LOCAL_ALPHA_WINDOW * factor)

            &

            (np.abs(work[re_col] - re_value)

             <= CM_LOCAL_RE_WINDOW * factor)

        ]

        if len(selected) >= CM_LOCAL_MIN_POINTS:

            break

    if selected.empty:

        return None, None

    cm_values = selected["CM"].to_numpy(dtype=float)

    cm_min = float(

        np.percentile(cm_values, CM_LOCAL_LOWER_PERCENTILE)

    )

    cm_max = float(

        np.percentile(cm_values, CM_LOCAL_UPPER_PERCENTILE)

    )

    if not (np.isfinite(cm_min) and np.isfinite(cm_max)):

        return None, None

    if cm_min > cm_max:

        cm_min, cm_max = cm_max, cm_min

    return cm_min, cm_max



# ======================================================================

# VALIDADE GEOMÉTRICA

# ======================================================================


# ======================================================================
# REGULARIDADE GEOMÉTRICA POR CURVATURA
# ======================================================================

def derivadas_curvatura(x: np.ndarray, y: np.ndarray):
    """Retorna dy/dx, d2y/dx2 e curvatura geométrica assinada."""
    dy = np.gradient(y, x, edge_order=2)
    d2y = np.gradient(dy, x, edge_order=2)
    kappa = d2y / np.power(1.0 + dy**2, 1.5)
    return dy, d2y, kappa


def contar_inversoes_relevantes_curvatura(x: np.ndarray, y: np.ndarray):
    """Conta inversões relevantes de curvatura em 0.02 <= x/c <= 0.98."""
    _, _, kappa = derivadas_curvatura(x, y)
    mask = (
        (x >= CURVATURE_X_MIN)
        & (x <= CURVATURE_X_MAX)
        & np.isfinite(kappa)
    )
    ki = np.asarray(kappa[mask], dtype=float)

    if ki.size < 3:
        return 0, np.nan, np.nan, np.nan

    abs_k = np.abs(ki)
    p95_abs_kappa = float(np.percentile(abs_k, 95))
    threshold = max(
        CURVATURE_SIGN_THRESHOLD_FRACTION * p95_abs_kappa,
        1e-8,
    )

    filtered = ki.copy()
    filtered[np.abs(filtered) < threshold] = 0.0
    nonzero = filtered[filtered != 0.0]

    if nonzero.size < 2:
        inversions = 0
    else:
        inversions = int(
            np.sum(np.sign(nonzero[1:]) != np.sign(nonzero[:-1]))
        )

    p99_abs_kappa = float(np.percentile(abs_k, 99))
    total_variation = float(np.sum(np.abs(np.diff(ki))))
    return inversions, threshold, p99_abs_kappa, total_variation


def diagnosticar_regularidade_curvatura(au: np.ndarray, al: np.ndarray):
    """Calcula as métricas de regularidade das superfícies CST."""
    x = np.linspace(0.0, 1.0, CURVATURE_N_POINTS)
    yu = cst_surface(x, au, delta_te=0.0)
    yl = cst_surface(x, al, delta_te=0.0)

    upper = contar_inversoes_relevantes_curvatura(x, yu)
    lower = contar_inversoes_relevantes_curvatura(x, yl)

    return {
        "upper_curvature_inversions": int(upper[0]),
        "lower_curvature_inversions": int(lower[0]),
        "upper_curvature_threshold": float(upper[1]),
        "lower_curvature_threshold": float(lower[1]),
        "upper_kappa_abs_p99": float(upper[2]),
        "lower_kappa_abs_p99": float(lower[2]),
        "upper_kappa_total_variation": float(upper[3]),
        "lower_kappa_total_variation": float(lower[3]),
    }


def validar_geometria(

    au: np.ndarray,

    al: np.ndarray,

    dominio: DominioGeometrico,

    t_min: float,

    t_max: float,

):

    """

    Valida a geometria e SEMPRE retorna um dicionário diagnóstico completo.

    Isso evita NaN no relatório final apenas porque a função retornou cedo.

    """

    cst_vector = np.concatenate([au, al])

    info = {

        "motivo": "não avaliado",

        "le_gap": np.nan,

        "te_gap": np.nan,

        "min_thickness": np.nan,

        "max_thickness": np.nan,

        "max_abs_camber": np.nan,

        "max_abs_camber_pct": np.nan,

        "x_max_abs_camber": np.nan,

        "max_abs_camber_limit": MAX_ABS_CAMBER,
        "upper_curvature_inversions": np.nan,
        "lower_curvature_inversions": np.nan,
        "upper_curvature_inversions_limit": MAX_CURVATURE_INVERSIONS_UPPER,
        "lower_curvature_inversions_limit": MAX_CURVATURE_INVERSIONS_LOWER,
        "upper_curvature_threshold": np.nan,
        "lower_curvature_threshold": np.nan,
        "upper_kappa_abs_p99": np.nan,
        "lower_kappa_abs_p99": np.nan,
        "upper_kappa_total_variation": np.nan,
        "lower_kappa_total_variation": np.nan,

        "distance_to_train": np.nan,

        "distance_limit": dominio.distance_limit,

        "cst_bound_violation": 0.0,

    }

    if not np.all(np.isfinite(cst_vector)):

        info["motivo"] = "CST não finito"

        return False, info

    mins = dominio.min_cst[CST_FEATURES].to_numpy(dtype=float)

    maxs = dominio.max_cst[CST_FEATURES].to_numpy(dtype=float)

    lower_violation = np.maximum(mins - cst_vector, 0.0)

    upper_violation = np.maximum(cst_vector - maxs, 0.0)

    info["cst_bound_violation"] = float(

        np.linalg.norm(lower_violation + upper_violation)

    )

    if info["cst_bound_violation"] > GEOM_TOL:

        info["motivo"] = "fora dos limites individuais CST"

        return False, info

    x, yu, yl = reconstruir_geometria(au, al)

    if not (np.all(np.isfinite(yu)) and np.all(np.isfinite(yl))):

        info["motivo"] = "geometria não finita"

        return False, info

    info["le_gap"] = abs(float(yu[0] - yl[0]))

    info["te_gap"] = abs(float(yu[-1] - yl[-1]))

    thickness = yu - yl

    camber = 0.5 * (yu + yl)

    info["min_thickness"] = float(np.min(thickness))

    info["max_thickness"] = float(np.max(thickness))

    i_camber = int(np.argmax(np.abs(camber)))

    info["max_abs_camber"] = float(np.abs(camber[i_camber]))

    info["max_abs_camber_pct"] = float(100.0 * np.abs(camber[i_camber]))

    info["x_max_abs_camber"] = float(x[i_camber])

    curvature_info = diagnosticar_regularidade_curvatura(au, al)
    info.update(curvature_info)

    # Calculamos a distância mesmo se outra restrição falhar,

    # para o relatório sempre mostrar o estado geométrico completo.

    info["distance_to_train"] = dominio.distancia_ao_treino(cst_vector)

    if info["le_gap"] > GEOM_TOL:

        info["motivo"] = "bordo de ataque aberto"

        return False, info

    if info["te_gap"] > GEOM_TOL:

        info["motivo"] = "bordo de fuga aberto"

        return False, info

    if info["min_thickness"] < -GEOM_TOL:

        info["motivo"] = "cruzamento entre superfícies"

        return False, info

    if info["max_thickness"] < t_min - GEOM_TOL:

        info["motivo"] = "perfil mais fino que domínio de referência"

        return False, info

    if info["max_thickness"] > t_max + GEOM_TOL:

        info["motivo"] = "perfil mais espesso que domínio de referência"

        return False, info

    if info["max_abs_camber"] > MAX_ABS_CAMBER + GEOM_TOL:

        info["motivo"] = (
            "camber absoluto máximo excessivo "
            f"({100.0 * info['max_abs_camber']:.4f}% c > "
            f"{100.0 * MAX_ABS_CAMBER:.4f}% c)"
        )

        return False, info

    if (
        info["upper_curvature_inversions"]
        > MAX_CURVATURE_INVERSIONS_UPPER
    ):
        info["motivo"] = (
            "excesso de inversões relevantes de curvatura no extradorso "
            f"({info['upper_curvature_inversions']} > "
            f"{MAX_CURVATURE_INVERSIONS_UPPER})"
        )
        return False, info

    if (
        info["lower_curvature_inversions"]
        > MAX_CURVATURE_INVERSIONS_LOWER
    ):
        info["motivo"] = (
            "excesso de inversões relevantes de curvatura no intradorso "
            f"({info['lower_curvature_inversions']} > "
            f"{MAX_CURVATURE_INVERSIONS_LOWER})"
        )
        return False, info

    if info["distance_to_train"] > dominio.distance_limit:

        info["motivo"] = "distância geométrica excessiva ao domínio de treino"

        return False, info

    info["motivo"] = "valido"

    return True, info





def gerar_populacao_inicial_factivel(
    dominio: DominioGeometrico,
    df_ref: pd.DataFrame,
    t_min: float,
    t_max: float,
    optimize_alpha: bool = OPTIMIZE_ALPHA,
    seed: int = SEED,
):
    """
    População inicial V2:
      - popsize=20 -> ~280 indivíduos para 14 variáveis;
      - ~70% perto dos Top-N perfis de maior CL;
      - ~30% exploração global;
      - número de tentativas ESTRITAMENTE limitado;
      - progresso impresso durante a geração;
      - se a parcela global tiver baixa aceitação, completa rapidamente
        com perturbações em regiões de alto CL.

    A validação geométrica continua ativa, agora incluindo camber absoluto
    máximo <= 10.3% c. O limite de distância é P80 * 1.80, mantendo a
    região do s1223 acessível sem repetir a expansão excessiva anterior.
    """
    rng = np.random.default_rng(seed)
    n_dim = 15 if optimize_alpha else 14
    target_size = max(5, POP_SIZE * n_dim)
    target_high = int(round(target_size * INIT_HIGH_CL_FRACTION))
    target_global = target_size - target_high

    mins = dominio.min_cst[CST_FEATURES].to_numpy(dtype=float)
    maxs = dominio.max_cst[CST_FEATURES].to_numpy(dtype=float)

    # --------------------------------------------------------------
    # Centros promissores: maior CL na condição oficial.
    # --------------------------------------------------------------
    ref = df_ref.copy()
    mask = (
        np.isclose(ref["Re"].to_numpy(float), RE_FIXED, rtol=0.0, atol=1e-6)
        & np.isclose(
            ref["alpha"].to_numpy(float),
            ALPHA_FIXED,
            rtol=0.0,
            atol=1e-8,
        )
    )
    ref = ref.loc[mask].copy()

    if ref.empty:
        raise RuntimeError(
            "Sem dados de referência em Re=250k e alpha=6° "
            "para selecionar centros de alto CL."
        )

    if "perfil" in ref.columns:
        ref = (
            ref.sort_values("CL", ascending=False)
            .drop_duplicates("perfil", keep="first")
        )
    else:
        ref = ref.sort_values("CL", ascending=False)

    high = ref.head(HIGH_CL_TOP_N_PROFILES).copy()
    centers = high[CST_FEATURES].to_numpy(dtype=float)
    z_centers = dominio.scaler.transform(centers)

    names = (
        high["perfil"].astype(str).tolist()
        if "perfil" in high.columns
        else [f"centro_{i+1}" for i in range(len(high))]
    )
    cls = high["CL"].to_numpy(dtype=float)

    print("\nCentros de alto CL:")
    for i, (name, cl) in enumerate(zip(names, cls), 1):
        print(f"  {i:02d}. {name:<16s} CL_base={cl:.6f}")

    print("\nGERANDO POPULAÇÃO INICIAL V2")
    print(f"Alvo total: {target_size}")
    print(f"Alvo alto-CL: {target_high}")
    print(f"Alvo global: {target_global}")

    population = []
    last_report = 0

    def adicionar(cst):
        nonlocal last_report
        cst = np.asarray(cst, dtype=float)
        valid, _ = validar_geometria(
            cst[:7], cst[7:], dominio, t_min, t_max
        )
        if not valid:
            return False

        member = (
            np.concatenate([cst, [rng.uniform(ALPHA_MIN, ALPHA_MAX)]])
            if optimize_alpha
            else cst.copy()
        )
        population.append(member)

        # Feedback visual: nunca mais parece "travado".
        milestone = (len(population) // 25) * 25
        if milestone >= 25 and milestone > last_report:
            last_report = milestone
            print(
                f"  População: {len(population)}/{target_size} "
                f"({100*len(population)/target_size:.1f}%)"
            )
        return True

    # --------------------------------------------------------------
    # 1) Tenta inserir centros reais válidos.
    # --------------------------------------------------------------
    for idx in rng.permutation(len(centers)):
        if len(population) >= target_high:
            break
        adicionar(np.clip(centers[idx], mins, maxs))

    # --------------------------------------------------------------
    # 2) Completa os 70% perto de regiões de alto CL.
    # Limite rígido de tentativas para não travar.
    # --------------------------------------------------------------
    high_attempt_limit = max(5000, target_high * 40)
    high_attempts = 0

    while len(population) < target_high and high_attempts < high_attempt_limit:
        high_attempts += 1
        idx = rng.integers(0, len(z_centers))

        sigma = (
            HIGH_CL_WIDE_STD
            if rng.random() < HIGH_CL_WIDE_FRACTION
            else HIGH_CL_LOCAL_STD
        )

        z = z_centers[idx] + rng.normal(
            0.0, sigma, size=z_centers.shape[1]
        )
        cst = dominio.scaler.inverse_transform(z.reshape(1, -1))[0]
        cst = np.clip(cst, mins, maxs)
        adicionar(cst)

    high_count = len(population)

    print(
        f"Etapa alto-CL concluída: {high_count}/{target_high} aceitos "
        f"em {high_attempts} tentativas."
    )

    # --------------------------------------------------------------
    # 3) Até 30% global, mas com orçamento pequeno e previsível.
    # --------------------------------------------------------------
    global_start = len(population)
    global_attempt_limit = max(3000, target_global * 30)
    global_attempts = 0

    while (
        len(population) < target_size
        and (len(population) - global_start) < target_global
        and global_attempts < global_attempt_limit
    ):
        global_attempts += 1
        cst = rng.uniform(mins, maxs)
        adicionar(cst)

    global_count = len(population) - global_start

    print(
        f"Etapa global concluída: {global_count}/{target_global} aceitos "
        f"em {global_attempts} tentativas."
    )

    # --------------------------------------------------------------
    # 4) Fallback rápido.
    # Se o global não preencher o alvo, NÃO fica insistindo.
    # Completa perto das regiões promissoras.
    # --------------------------------------------------------------
    fallback_attempt_limit = max(5000, target_size * 40)
    fallback_attempts = 0

    while len(population) < target_size and fallback_attempts < fallback_attempt_limit:
        fallback_attempts += 1
        idx = rng.integers(0, len(z_centers))

        # Mais amplo que a perturbação local normal.
        sigma = rng.uniform(HIGH_CL_LOCAL_STD, HIGH_CL_WIDE_STD)
        z = z_centers[idx] + rng.normal(
            0.0, sigma, size=z_centers.shape[1]
        )
        cst = dominio.scaler.inverse_transform(z.reshape(1, -1))[0]
        cst = np.clip(cst, mins, maxs)
        adicionar(cst)

    if len(population) < 5:
        raise RuntimeError(
            "Não foi possível gerar população inicial factível."
        )

    # O SciPy aceita população explícita; não precisamos insistir
    # indefinidamente se faltarem poucos membros.
    if len(population) < target_size:
        warnings.warn(
            f"População inicial ficou em {len(population)}/{target_size}. "
            "A busca seguirá com a população factível disponível."
        )

    population = np.asarray(population, dtype=float)

    print("\nPOPULAÇÃO INICIAL PRONTA")
    print(f"  Total: {len(population)}")
    print(f"  Alto-CL: {high_count}")
    print(f"  Global: {global_count}")
    print(f"  Fallback alto-CL: {len(population) - high_count - global_count}")
    print(
        f"  Tentativas: alto-CL={high_attempts}, "
        f"global={global_attempts}, fallback={fallback_attempts}"
    )
    print("  Iniciando Differential Evolution...\n")

    return population


# ======================================================================
# PROBLEMA DE OTIMIZAÇÃO
# ======================================================================

class ProblemaOtimizacao:

    """

    O Differential Evolution otimiza SOMENTE os 14 coeficientes CST.

    Para cada geometria válida:

      1) o XGBoost avalia toda a grade de alpha de forma vetorizada;

      2) aplica os filtros locais de CD e CM em cada alpha;

      3) seleciona o maior CL surrogate admissível;

      4) usa -max(CL) como função objetivo;

      5) guarda as melhores geometrias para validação posterior no XFOIL.

    """

    def __init__(

        self,

        model_cl,

        model_cd,

        model_cm,

        dominio: DominioGeometrico,

        t_min: float,

        t_max: float,

        cd_min_global: float,

        df_ref: pd.DataFrame,

    ):

        self.model_cl = model_cl

        self.model_cd = model_cd

        self.model_cm = model_cm

        self.dominio = dominio

        self.t_min = t_min

        self.t_max = t_max

        self.cd_min_global = cd_min_global

        self.df_ref = df_ref

        self.n_calls = 0

        self.n_valid = 0

        # Cada execução DE mantém seu próprio reservatório de candidatos.
        # Isso impede que uma única seed/platô ocupe todo o pool global.
        self.seed_candidate_pools = {}

        # Seed da execução DE atualmente ativa; salvo com cada candidato.
        self.current_seed = None

        # Os limites locais dependem somente de Re e alpha. Como Re é fixo

        # e a grade de alpha também, calculamos UMA VEZ no início.

        self.alpha_grid = np.array([ALPHA_FIXED], dtype=float)

        self.cd_floor_grid = np.empty_like(self.alpha_grid, dtype=float)

        self.cm_min_grid = np.empty_like(self.alpha_grid, dtype=float)

        self.cm_max_grid = np.empty_like(self.alpha_grid, dtype=float)

        for i, alpha in enumerate(self.alpha_grid):

            self.cd_floor_grid[i] = calcular_cd_minimo_local(

                self.df_ref,

                re_value=RE_FIXED,

                alpha=float(alpha),

                cd_global_min=self.cd_min_global,

            )

            cm_min, cm_max = calcular_limites_cm_local(

                self.df_ref,

                re_value=RE_FIXED,

                alpha=float(alpha),

            )

            self.cm_min_grid[i] = np.nan if cm_min is None else float(cm_min)

            self.cm_max_grid[i] = np.nan if cm_max is None else float(cm_max)

    def decode(self, vector):

        """Decodifica somente os 14 CST. Alpha não pertence mais ao vetor."""

        vector = np.asarray(vector, dtype=float)

        if vector.size != 14:

            raise ValueError(

                f"O vetor de otimização deve ter 14 variáveis CST; "

                f"recebido: {vector.size}."

            )

        au = vector[:7]

        al = vector[7:14]

        re_value = float(RE_FIXED)

        return au, al, re_value

    def _montar_dataframe_polar(self, model, au, al, re_value):

        """Monta a condição operacional única do surrogate: alpha=6°, Re=250.000."""

        base = {

            **{f"Au{i}": float(au[i]) for i in range(7)},

            **{f"Al{i}": float(al[i]) for i in range(7)},

            "DeltaTE_upper": 0.0,

            "DeltaTE_lower": 0.0,

            "Re": float(re_value),

        }

        feature_names = resolver_feature_names(model)

        rows = []

        for alpha in self.alpha_grid:

            values = dict(base)

            values["alpha"] = float(alpha)

            faltantes = [f for f in feature_names if f not in values]

            if faltantes:

                raise KeyError(

                    "O modelo solicita features que o otimizador não reconheceu: "

                    + ", ".join(faltantes)

                )

            rows.append([values[f] for f in feature_names])

        return pd.DataFrame(rows, columns=feature_names)

    def predict_polar(self, au, al, re_value):

        """

        Calcula toda a polar surrogate vetorizada e retorna:

        - DataFrame completo da polar prevista;

        - melhor ponto admissível segundo CL máximo.

        """

        X_cl = self._montar_dataframe_polar(

            self.model_cl, au, al, re_value

        )

        X_cd = self._montar_dataframe_polar(

            self.model_cd, au, al, re_value

        )

        X_cm = self._montar_dataframe_polar(

            self.model_cm, au, al, re_value

        )

        cl = np.asarray(self.model_cl.predict(X_cl), dtype=float).ravel()

        cd = np.asarray(self.model_cd.predict(X_cd), dtype=float).ravel()

        cm = np.asarray(self.model_cm.predict(X_cm), dtype=float).ravel()

        polar = pd.DataFrame({

            "alpha": self.alpha_grid,

            "CL_pred": cl,

            "CD_pred": cd,

            "CM_pred": cm,

            "CD_local_min": self.cd_floor_grid,

            "CM_local_min": self.cm_min_grid,

            "CM_local_max": self.cm_max_grid,

        })

        finite = (

            np.isfinite(cl)

            & np.isfinite(cd)

            & np.isfinite(cm)

            & (cd > 0.0)

        )

        if USE_CD_FLOOR_CONSTRAINT:
            cd_ok = cd >= self.cd_floor_grid
        else:
            cd_ok = np.ones(len(self.alpha_grid), dtype=bool)

        cm_ok = np.ones(len(self.alpha_grid), dtype=bool)
        if USE_CM_CONSTRAINT:
            has_cm_min = np.isfinite(self.cm_min_grid)
            has_cm_max = np.isfinite(self.cm_max_grid)

            cm_ok[has_cm_min] &= (
                cm[has_cm_min] >= self.cm_min_grid[has_cm_min]
            )
            cm_ok[has_cm_max] &= (
                cm[has_cm_max] <= self.cm_max_grid[has_cm_max]
            )

        admissible = finite & cd_ok & cm_ok

        polar["admissible"] = admissible

        polar["CL_CD_pred"] = np.where(

            finite,

            cl / np.maximum(cd, 1e-12),

            np.nan,

        )

        valid_polar = polar[polar["admissible"]].copy()

        if valid_polar.empty:

            return polar, None

        best_idx = valid_polar["CL_pred"].idxmax()

        best = valid_polar.loc[best_idx].to_dict()

        return polar, best

    def registrar_candidato(
        self,
        vector,
        best_point,
        distance_penalty: float = 0.0,
    ):
        """
        Guarda candidatos em um POOL INDEPENDENTE POR SEED.

        Cada seed retém seus próprios melhores candidatos segundo a função
        objetivo penalizada. Só depois das execuções os pools são combinados
        e submetidos ao farthest-point sampling no espaço CST padronizado.

        Isso evita que centenas de pequenas variações do mesmo platô de uma
        única execução eliminem candidatos encontrados pelas demais seeds.
        """
        if best_point is None:
            return

        eff = float(best_point["CL_pred"])
        if not np.isfinite(eff):
            return

        distance_penalty = float(distance_penalty)
        score = eff - distance_penalty
        vec = np.asarray(vector, dtype=float).copy()

        seed_key = (
            int(self.current_seed)
            if self.current_seed is not None
            else -1
        )
        pool = self.seed_candidate_pools.setdefault(seed_key, [])

        # Elimina somente duplicatas numéricas dentro da própria seed.
        for item in pool:
            if np.linalg.norm(item["vector"] - vec) <= 1e-10:
                if score > item["optimization_score"]:
                    item.update({
                        "vector": vec,
                        "alpha_pred_opt": float(best_point["alpha"]),
                        "CL_pred": float(best_point["CL_pred"]),
                        "CD_pred": float(best_point["CD_pred"]),
                        "CM_pred": float(best_point["CM_pred"]),
                        "CL_CD_pred": float(best_point["CL_CD_pred"]),
                        "distance_soft_penalty": distance_penalty,
                        "optimization_score": score,
                        "origin_seed": seed_key,
                    })
                return

        pool.append({
            "vector": vec,
            "alpha_pred_opt": float(best_point["alpha"]),
            "CL_pred": float(best_point["CL_pred"]),
            "CD_pred": float(best_point["CD_pred"]),
            "CM_pred": float(best_point["CM_pred"]),
            "CL_CD_pred": float(best_point["CL_CD_pred"]),
            "distance_soft_penalty": distance_penalty,
            "optimization_score": score,
            "origin_seed": seed_key,
        })

        pool.sort(
            key=lambda item: item["optimization_score"],
            reverse=True,
        )
        del pool[CANDIDATE_POOL_SIZE_PER_SEED:]

    def todos_candidatos(self):
        """Combina os pools por seed sem truncamento global."""
        merged = []
        for seed in sorted(self.seed_candidate_pools):
            merged.extend(self.seed_candidate_pools[seed])

        merged.sort(
            key=lambda item: item["optimization_score"],
            reverse=True,
        )
        return merged


    def _penalidade_distancia_suave(self, distance: float) -> float:
        """
        Penalização contínua dentro do domínio permitido.

        ratio <= DISTANCE_SOFT_START -> 0
        ratio = 1.0                    -> DISTANCE_SOFT_WEIGHT

        O limite duro continua sendo aplicado em validar_geometria().
        """
        if not np.isfinite(distance):
            return 0.0

        limit = max(float(self.dominio.distance_limit), 1e-12)
        ratio = float(distance) / limit

        if ratio <= DISTANCE_SOFT_START:
            return 0.0

        normalized = (
            (ratio - DISTANCE_SOFT_START)
            / max(1.0 - DISTANCE_SOFT_START, 1e-12)
        )
        normalized = float(np.clip(normalized, 0.0, 1.0))

        return float(
            DISTANCE_SOFT_WEIGHT
            * normalized ** DISTANCE_SOFT_POWER
        )

    def _penalidade_geometrica(self, info):

        penalty = PENALTY_BASE

        if np.isfinite(info.get("cst_bound_violation", np.nan)):

            penalty += PENALTY_SCALE * info["cst_bound_violation"]

        if np.isfinite(info.get("le_gap", np.nan)):

            penalty += PENALTY_SCALE * 100.0 * info["le_gap"]

        if np.isfinite(info.get("te_gap", np.nan)):

            penalty += PENALTY_SCALE * 100.0 * info["te_gap"]

        min_t = info.get("min_thickness", np.nan)

        if np.isfinite(min_t) and min_t < 0.0:

            penalty += PENALTY_SCALE * 100.0 * abs(min_t)

        max_t = info.get("max_thickness", np.nan)

        if np.isfinite(max_t):

            if max_t < self.t_min:

                penalty += PENALTY_SCALE * 100.0 * (self.t_min - max_t)

            elif max_t > self.t_max:

                penalty += PENALTY_SCALE * 100.0 * (max_t - self.t_max)

        distance = info.get("distance_to_train", np.nan)

        if np.isfinite(distance) and distance > self.dominio.distance_limit:

            penalty += PENALTY_SCALE * (

                distance - self.dominio.distance_limit

            )

        return float(penalty)

    def objective(self, vector):

        self.n_calls += 1

        au, al, re_value = self.decode(vector)

        if not (RE_MIN <= re_value <= RE_MAX):

            return PENALTY_BASE + PENALTY_SCALE * abs(

                re_value - np.clip(re_value, RE_MIN, RE_MAX)

            )

        valid, info = validar_geometria(

            au,

            al,

            self.dominio,

            self.t_min,

            self.t_max,

        )

        if not valid:

            return self._penalidade_geometrica(info)

        _, best = self.predict_polar(au, al, re_value)

        if best is None:

            return float(PENALTY_BASE)

        self.n_valid += 1

        eff = float(best["CL_pred"])
        distance = float(info["distance_to_train"])
        distance_penalty = self._penalidade_distancia_suave(distance)

        # Para minimização pelo scipy:
        #   objective = -CL + penalização
        # equivalente a maximizar CL mantendo distância do limite.
        self.registrar_candidato(
            vector,
            best,
            distance_penalty=distance_penalty,
        )

        return -eff + distance_penalty



# ======================================================================

# VALIDAÇÃO FINAL DOS MELHORES CANDIDATOS COM XFOIL — POLAR COMPLETA

# ======================================================================

def localizar_xfoil() -> Path:

    p = Path(XFOIL_EXE)

    if not p.exists():

        raise FileNotFoundError(f"XFOIL não encontrado em: {p}")

    return p



def ler_polar_xfoil(path: Path):

    """Lê toda a polar XFOIL e retorna DataFrame + ponto de CL máximo."""

    if not path.exists():

        return None

    rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:

                continue

            try:

                vals = [float(v) for v in parts[:5]]

            except ValueError:

                continue

            rows.append(vals)

    if not rows:

        return None

    df = pd.DataFrame(

        rows,

        columns=["alpha", "CL", "CD", "CDp", "CM"],

    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(

        subset=["alpha", "CL", "CD", "CM"]

    )

    df = df[df["CD"] > 0.0].copy()

    if df.empty:

        return None

    # Caso XFOIL grave a mesma linha mais de uma vez, preserva a última.

    df = (

        df.sort_values("alpha")

        .drop_duplicates(subset=["alpha"], keep="last")

        .reset_index(drop=True)

    )

    df["CL_CD"] = df["CL"] / df["CD"]

    target = df[np.isclose(
        df["alpha"].to_numpy(dtype=float),
        float(ALPHA_FIXED),
        atol=1e-8,
        rtol=0.0,
    )].copy()

    # O experimento é controlado em alpha=6°. Se o XFOIL não convergir
    # exatamente nesse ponto, o candidato não é válido para o ranking oficial.
    if target.empty:
        return None

    best = target.iloc[-1]

    return {

        "polar": df,

        "alpha_xfoil_opt": float(best["alpha"]),

        "CL_xfoil_opt": float(best["CL"]),

        "CD_xfoil_opt": float(best["CD"]),

        "CM_xfoil_opt": float(best["CM"]),

        "CL_CD_xfoil_opt": float(best["CL_CD"]),

        "n_xfoil_points": int(len(df)),

    }



def executar_xfoil_polar(

    xfoil_exe: Path,

    dat_path: Path,

    re_value: float,

    candidate_id: int,

):

    work_dir = XFOIL_VALIDATION_DIR / f"candidato_{candidate_id:03d}"

    work_dir.mkdir(parents=True, exist_ok=True)

    local_dat = work_dir / "aerofolio.dat"

    polar_path = work_dir / "polar.txt"

    shutil.copy2(dat_path, local_dat)

    if polar_path.exists():

        polar_path.unlink()

    commands = "\n".join([

        f"LOAD {local_dat.name}",

        "",

        "PANE",

        "OPER",

        f"VISC {re_value:.0f}",

        f"MACH {MACH:.6f}",

        f"ITER {XFOIL_ITER}",

        "PACC",

        polar_path.name,

        "",

        (

            f"ASEQ {XFOIL_ALPHA_MIN:.6f} "

            f"{XFOIL_ALPHA_MAX:.6f} "

            f"{XFOIL_ALPHA_STEP:.6f}"

        ),

        "PACC",

        "",

        "QUIT",

        "",

    ])

    (work_dir / "comandos_xfoil.txt").write_text(

        commands,

        encoding="utf-8",

    )

    try:

        proc = subprocess.run(

            [str(xfoil_exe)],

            input=commands,

            text=True,

            capture_output=True,

            timeout=XFOIL_TIMEOUT,

            cwd=str(work_dir),

        )

    except subprocess.TimeoutExpired:

        (work_dir / "erro_xfoil.txt").write_text(

            f"Timeout após {XFOIL_TIMEOUT} s.",

            encoding="utf-8",

        )

        return None

    (work_dir / "stdout.txt").write_text(

        proc.stdout or "",

        encoding="utf-8",

        errors="ignore",

    )

    (work_dir / "stderr.txt").write_text(

        proc.stderr or "",

        encoding="utf-8",

        errors="ignore",

    )

    if proc.returncode != 0:

        (work_dir / "erro_xfoil.txt").write_text(

            f"XFOIL retornou código {proc.returncode}.\n",

            encoding="utf-8",

        )

        return None

    result = ler_polar_xfoil(polar_path)

    if result is None:

        (work_dir / "erro_xfoil.txt").write_text(

            "XFOIL não gerou pontos de polar válidos. Consulte stdout.txt.",

            encoding="utf-8",

        )

        return None

    result["polar"].to_csv(

        work_dir / "polar_xfoil_processada.csv",

        index=False,

    )

    return result



def distancia_padronizada_entre_geometrias(

    dominio: DominioGeometrico,

    vec_a: np.ndarray,

    vec_b: np.ndarray,

) -> float:

    a = dominio.scaler.transform(

        np.asarray(vec_a, dtype=float).reshape(1, -1)

    )[0]

    b = dominio.scaler.transform(

        np.asarray(vec_b, dtype=float).reshape(1, -1)

    )[0]

    return float(np.linalg.norm(a - b))



def selecionar_candidatos_distintos(
    problema,
    dominio,
    t_min,
    t_max,
):
    """
    Seleciona candidatos com FARTHEST-POINT SAMPLING (FPS).

    Etapas:
    1) combina os pools independentes das seeds;
    2) remove geometrias inválidas;
    3) inicia pelo melhor score surrogate penalizado;
    4) escolhe iterativamente o ponto cuja MENOR distância a qualquer ponto
       já selecionado é a maior possível;
    5) mede todas as distâncias no espaço CST padronizado.

    Os pools por seed funcionam como filtro de qualidade. O FPS é responsável
    por maximizar a cobertura geométrica entre os candidatos enviados ao XFOIL.
    """
    all_candidates = problema.todos_candidatos()
    if not all_candidates:
        return []

    valid_items = []

    for item in all_candidates:
        vec = np.asarray(item["vector"], dtype=float)
        au, al, _ = problema.decode(vec)

        valid, info = validar_geometria(
            au,
            al,
            dominio,
            t_min,
            t_max,
        )
        if not valid:
            continue

        scaled = dominio.scaler.transform(
            vec.reshape(1, -1)
        )[0]

        new_item = dict(item)
        new_item["geom_info"] = info
        new_item["_scaled_vector"] = scaled
        valid_items.append(new_item)

    if not valid_items:
        return []

    # Melhor candidato segundo o surrogate penalizado sempre entra primeiro.
    valid_items.sort(
        key=lambda item: item["optimization_score"],
        reverse=True,
    )

    first = valid_items.pop(0)
    first["fps_min_distance_to_selected"] = np.nan
    selected = [first]

    while valid_items and len(selected) < TOP_SURROGATE_TO_VALIDATE:
        selected_matrix = np.vstack(
            [item["_scaled_vector"] for item in selected]
        )
        candidate_matrix = np.vstack(
            [item["_scaled_vector"] for item in valid_items]
        )

        # Distância de cada candidato a todos os já selecionados.
        diff = (
            candidate_matrix[:, None, :]
            - selected_matrix[None, :, :]
        )
        distances = np.linalg.norm(diff, axis=2)
        min_distances = distances.min(axis=1)

        # FPS puro: escolhe a maior distância mínima.
        best_pos = int(np.argmax(min_distances))
        best_distance = float(min_distances[best_pos])

        # Se só restarem duplicatas numéricas, não há nova diversidade.
        if best_distance <= FARTHEST_POINT_MIN_DISTANCE_STD:
            break

        chosen = valid_items.pop(best_pos)
        chosen["fps_min_distance_to_selected"] = best_distance
        selected.append(chosen)

    # Remove campo auxiliar antes de salvar/retornar.
    for item in selected:
        item.pop("_scaled_vector", None)

    return selected


def _surrogate_no_alpha(

    problema: ProblemaOtimizacao,

    au,

    al,

    re_value,

    alpha_target,

):

    """Obtém a previsão surrogate no alpha da polar XFOIL mais próximo."""

    polar, _ = problema.predict_polar(au, al, re_value)

    idx = (polar["alpha"] - float(alpha_target)).abs().idxmin()

    row = polar.loc[idx]

    return {

        "alpha_pred_at_xfoil_opt": float(row["alpha"]),

        "CL_pred_at_xfoil_opt": float(row["CL_pred"]),

        "CD_pred_at_xfoil_opt": float(row["CD_pred"]),

        "CM_pred_at_xfoil_opt": float(row["CM_pred"]),

        "CL_CD_pred_at_xfoil_opt": float(row["CL_CD_pred"]),

    }



def validar_melhores_candidatos_xfoil(

    problema,

    dominio,

    t_min,

    t_max,

):

    xfoil_exe = localizar_xfoil()

    candidates = selecionar_candidatos_distintos(

        problema,

        dominio,

        t_min,

        t_max,

    )

    if not candidates:

        raise RuntimeError(

            "Nenhum candidato válido disponível para validação no XFOIL."

        )

    print("\n" + "=" * 88)

    print("VALIDAÇÃO TOP-N NO XFOIL — RANKING OFICIAL EM ALPHA=6°")

    print("=" * 88)

    print(f"Candidatos selecionados: {len(candidates)}")

    print(
        f"Polar XFOIL diagnóstica: {XFOIL_ALPHA_MIN:.2f}° -> {XFOIL_ALPHA_MAX:.2f}° "
        f"com passo {XFOIL_ALPHA_STEP:.2f}° | ranking em alpha={ALPHA_FIXED:.2f}°"
    )

    rows = []

    for idx, item in enumerate(candidates, start=1):

        au, al, re_value = problema.decode(item["vector"])

        dat_path = XFOIL_VALIDATION_DIR / f"candidato_{idx:03d}.dat"

        salvar_dat(dat_path, au, al)

        xf = executar_xfoil_polar(

            xfoil_exe=xfoil_exe,

            dat_path=dat_path,

            re_value=re_value,

            candidate_id=idx,

        )

        row = {

            "candidate_id": idx,
            "origin_seed": item.get("origin_seed"),

            **{f"Au{i}": float(au[i]) for i in range(7)},

            **{f"Al{i}": float(al[i]) for i in range(7)},

            "Re": float(re_value),

            "alpha_pred_opt": float(item["alpha_pred_opt"]),

            "CL_pred_opt": float(item["CL_pred"]),

            "CD_pred_opt": float(item["CD_pred"]),

            "CM_pred_opt": float(item["CM_pred"]),

            "CL_CD_pred_opt": float(item["CL_CD_pred"]),

            "distance_soft_penalty": float(item.get("distance_soft_penalty", 0.0)),

            "optimization_score": float(item.get("optimization_score", item["CL_CD_pred"])),

            "fps_min_distance_to_selected": float(
                item.get("fps_min_distance_to_selected", np.nan)
            ),

            "distance_to_train": float(

                item["geom_info"]["distance_to_train"]

            ),

            "max_thickness": float(

                item["geom_info"]["max_thickness"]

            ),

            "xfoil_converged": xf is not None,

        }

        if xf is not None:

            row.update({

                "alpha_xfoil_opt": xf["alpha_xfoil_opt"],

                "CL_xfoil_opt": xf["CL_xfoil_opt"],

                "CD_xfoil_opt": xf["CD_xfoil_opt"],

                "CM_xfoil_opt": xf["CM_xfoil_opt"],

                "CL_CD_xfoil_opt": xf["CL_CD_xfoil_opt"],

                "n_xfoil_points": xf["n_xfoil_points"],

                "xfoil_polar_sufficient": (

                    xf["n_xfoil_points"] >= XFOIL_MIN_CONVERGED_POINTS

                ),

            })

            pred_same_alpha = _surrogate_no_alpha(

                problema,

                au,

                al,

                re_value,

                xf["alpha_xfoil_opt"],

            )

            row.update(pred_same_alpha)

            row["delta_alpha_opt_deg"] = abs(

                row["alpha_pred_opt"] - row["alpha_xfoil_opt"]

            )

            row["erro_CL_CD_max_pct"] = (

                100.0

                * abs(row["CL_CD_pred_opt"] - row["CL_CD_xfoil_opt"])

                / max(abs(row["CL_CD_xfoil_opt"]), 1e-12)

            )

            row["erro_CL_no_alpha_xfoil_pct"] = (

                100.0

                * abs(row["CL_pred_at_xfoil_opt"] - row["CL_xfoil_opt"])

                / max(abs(row["CL_xfoil_opt"]), 1e-12)

            )

            row["erro_CD_no_alpha_xfoil_pct"] = (

                100.0

                * abs(row["CD_pred_at_xfoil_opt"] - row["CD_xfoil_opt"])

                / max(abs(row["CD_xfoil_opt"]), 1e-12)

            )

            print(

                f"[{idx:02d}/{len(candidates)}] "

                f"ML CL={row['CL_pred_opt']:.5f} @ {row['alpha_pred_opt']:.2f}° | "

                f"XFOIL CL={row['CL_xfoil_opt']:.5f} @ {row['alpha_xfoil_opt']:.2f}° | "

                f"pontos={row['n_xfoil_points']}"

            )

        else:

            row["xfoil_polar_sufficient"] = False

            print(

                f"[{idx:02d}/{len(candidates)}] "

                "XFOIL não gerou polar válida."

            )

        rows.append(row)

    df = pd.DataFrame(rows)

    output_csv = OUTPUT_DIR / "validacao_candidatos_xfoil_polar.csv"

    df.to_csv(output_csv, index=False)

    converged = df[

        (df["xfoil_converged"] == True)

        & (df["xfoil_polar_sufficient"] == True)

    ].copy()

    if "CL_xfoil_opt" in converged.columns:

        converged = converged[

            np.isfinite(converged["CL_xfoil_opt"])

        ]

    if converged.empty:

        raise RuntimeError(

            "Nenhum candidato possui polar XFOIL suficiente para ranking final."

        )

    converged = converged.sort_values(

        "CL_xfoil_opt",

        ascending=False,

    ).reset_index(drop=True)

    converged.insert(0, "rank_xfoil", np.arange(1, len(converged) + 1))

    converged.to_csv(

        OUTPUT_DIR / "ranking_candidatos_xfoil.csv",

        index=False,

    )

    best_row = converged.iloc[0].copy()

    candidate_id = int(best_row["candidate_id"])

    best_candidate = candidates[candidate_id - 1]

    return df, best_row, best_candidate



# ======================================================================

# EXPORTAÇÃO DO RESULTADO

# ======================================================================

def salvar_dat(path: Path, au: np.ndarray, al: np.ndarray):

    """Formato .dat: extradorso BF->BA, depois intradorso BA->BF."""

    x, yu, yl = reconstruir_geometria(au, al, n_points=401)

    xu = x[::-1]

    yuu = yu[::-1]

    xl = x[1:]

    yll = yl[1:]

    with path.open("w", encoding="utf-8") as f:

        f.write("CST_OPTIMIZED\n")

        for xx, yy in zip(xu, yuu):

            f.write(f"{xx:.8f} {yy:.8f}\n")

        for xx, yy in zip(xl, yll):

            f.write(f"{xx:.8f} {yy:.8f}\n")



# ======================================================================

# MAIN

# ======================================================================

def main():

    print("=" * 88)

    print("OTIMIZAÇÃO HÍBRIDA CL | ALPHA=6° | RE=250K | DE FOCADO EM ALTO CL V2 + XFOIL")

    print("=" * 88)

    if not (RE_MIN <= RE_FIXED <= RE_MAX):

        raise ValueError(

            f"RE_FIXED={RE_FIXED} está fora do intervalo "

            f"[{RE_MIN}, {RE_MAX}]."

        )

    csv_ref = localizar_csv_referencia()

    print(f"\nBase geométrica de referência:\n{csv_ref}")

    df_ref = pd.read_csv(csv_ref)

    model_cl, model_cd, model_cm = carregar_modelos()

    print("\nModelos carregados com sucesso.")

    dominio = DominioGeometrico(df_ref)

    t_min, t_max, _ = calcular_limites_espessura(df_ref)

    cd_min_global = calcular_cd_minimo_global(df_ref)

    print("\n" + "-" * 88)

    print("LIMITES EXTRAÍDOS DA BASE")

    print("-" * 88)

    print(f"Perfis geométricos únicos: {len(dominio.geom)}")

    print(f"Espessura máxima observada mínima: {100*t_min:.3f}% c")

    print(f"Espessura máxima observada máxima: {100*t_max:.3f}% c")

    print(f"CD mínimo positivo global observado: {cd_min_global:.8f}")

    print(
        f"Distância CST base "
        f"(P{DISTANCE_PERCENTILE:.0f} do vizinho mais próximo): "
        f"{dominio.distance_limit_base:.6f}"
    )
    print(
        f"Limite duro AMPLIADO: {dominio.distance_limit:.6f} "
        f"({DISTANCE_LIMIT_MULTIPLIER:.2f} x P{DISTANCE_PERCENTILE:.0f})"
    )
    print(
        f"Expansão dos bounds CST: ±{100*CST_BOUND_EXPANSION:.0f}% "
        f"da amplitude observada no treino."
    )

    print(
        f"Penalização suave de distância: inicia em "
        f"{100*DISTANCE_SOFT_START:.0f}% do limite; "
        f"peso máximo={DISTANCE_SOFT_WEIGHT:.2f}; "
        f"potência={DISTANCE_SOFT_POWER:.1f}"
    )
    print(
        f"Pool por seed: {CANDIDATE_POOL_SIZE_PER_SEED}; "
        f"distância mínima Top-N padronizada: "
        f"FPS min={FARTHEST_POINT_MIN_DISTANCE_STD:.1e}"
    )

    print(f"Condição surrogate fixa: alpha={ALPHA_FIXED:.2f}°, Re={RE_FIXED:.0f}")

    print(

        f"Validação XFOIL Top-{TOP_SURROGATE_TO_VALIDATE}: "

        f"{XFOIL_ALPHA_MIN:.2f}° -> {XFOIL_ALPHA_MAX:.2f}° "

        f"com passo {XFOIL_ALPHA_STEP:.2f}°"

    )

    print("\nLimites CST:")

    for c in CST_FEATURES:

        print(

            f"  {c:>4s}: "

            f"[{dominio.min_cst[c]: .8f}, {dominio.max_cst[c]: .8f}]"

        )

    # SOMENTE 14 bounds CST. Alpha não é dimensão do DE.

    bounds = dominio.bounds_scipy()

    print("\nVariáveis do DE: 14 coeficientes CST.")

    print(f"Reynolds fixo: {RE_FIXED:.0f}")

    print(f"Alpha fixo durante a otimização: {ALPHA_FIXED:.2f}°")

    problema = ProblemaOtimizacao(

        model_cl=model_cl,

        model_cd=model_cd,

        model_cm=model_cm,

        dominio=dominio,

        t_min=t_min,

        t_max=t_max,

        cd_min_global=cd_min_global,

        df_ref=df_ref,

    )

    print("\n" + "=" * 88)
    print("INICIANDO BUSCA ÚNICA E AMPLA — DIFFERENTIAL EVOLUTION")
    print("=" * 88)
    print(f"Seed: {SEED}")
    print(f"Popsize: {POP_SIZE}")
    print(
        f"Inicialização: ~{100*INIT_HIGH_CL_FRACTION:.0f}% Top-"
        f"{HIGH_CL_TOP_N_PROFILES} alto-CL + "
        f"~{100*INIT_GLOBAL_FRACTION:.0f}% global"
    )
    print(f"Iterações máximas: {MAX_ITER}")
    print("Estratégia: rand1bin")
    print(
        f"CM como restrição: {USE_CM_CONSTRAINT} | "
        f"CD mínimo como restrição: {USE_CD_FLOOR_CONSTRAINT}"
    )

    problema.current_seed = int(SEED)

    init_population = gerar_populacao_inicial_factivel(
        dominio=dominio,
        df_ref=df_ref,
        t_min=t_min,
        t_max=t_max,
        optimize_alpha=False,
        seed=SEED,
    )

    if init_population.shape[1] != 14:
        raise RuntimeError(
            f"População inicial deveria possuir 14 colunas; "
            f"recebido {init_population.shape[1]}."
        )

    calls_before = problema.n_calls
    valid_before = problema.n_valid

    result = differential_evolution(
        problema.objective,
        bounds=bounds,
        strategy="rand1bin",
        maxiter=MAX_ITER,
        popsize=POP_SIZE,
        tol=TOL,
        mutation=(0.4, 1.2),
        recombination=0.8,
        seed=SEED,
        init=init_population,
        polish=POLISH,
        workers=WORKERS,
        updating="immediate" if WORKERS == 1 else "deferred",
        disp=True,
    )

    au_de, al_de, re_value = problema.decode(result.x)

    valid_de, geom_info_de = validar_geometria(
        au_de,
        al_de,
        dominio,
        t_min,
        t_max,
    )

    if not valid_de:
        raise RuntimeError(
            f"Melhor solução DE inválida: {geom_info_de.get('motivo')}"
        )

    polar_de, best_de = problema.predict_polar(
        au_de,
        al_de,
        re_value,
    )

    if best_de is None:
        raise RuntimeError(
            "A melhor solução DE não possui predição surrogate válida."
        )

    dist_pen = problema._penalidade_distancia_suave(
        float(geom_info_de["distance_to_train"])
    )

    problema.registrar_candidato(
        result.x,
        best_de,
        distance_penalty=dist_pen,
    )

    polar_de.to_csv(
        OUTPUT_DIR / "polar_surrogate_melhor_de.csv",
        index=False,
    )

    resumo_de = pd.DataFrame([{
        "seed": int(SEED),
        "scipy_success": bool(result.success),
        "message": str(result.message),
        "geometry_valid": bool(valid_de),
        "distance_to_train": float(
            geom_info_de.get("distance_to_train", np.nan)
        ),
        "distance_limit_base": float(dominio.distance_limit_base),
        "distance_limit_expanded": float(dominio.distance_limit),
        "alpha_pred_opt": float(best_de["alpha"]),
        "CL_pred_obj": float(best_de["CL_pred"]),
        "CD_pred": float(best_de["CD_pred"]),
        "CM_pred": float(best_de["CM_pred"]),
        "objective_calls": int(problema.n_calls - calls_before),
        "valid_calls": int(problema.n_valid - valid_before),
        "candidate_pool_size": int(
            len(problema.seed_candidate_pools.get(int(SEED), []))
        ),
    }])

    resumo_de.to_csv(
        OUTPUT_DIR / "resumo_de_exploratorio.csv",
        index=False,
    )

    best_seed = int(SEED)

    print("\n" + "=" * 88)
    print("DIAGNÓSTICO DA MELHOR SOLUÇÃO SEGUNDO O SURROGATE")
    print("=" * 88)
    print(f"Seed: {SEED}")
    print(f"Sucesso scipy: {result.success}")
    print(f"Mensagem: {result.message}")
    print(f"Geometria válida: {valid_de}")
    print(f"Motivo: {geom_info_de.get('motivo')}")
    print(
        f"Distância ao treino: "
        f"{geom_info_de.get('distance_to_train', np.nan):.6f} / "
        f"limite ampliado {dominio.distance_limit:.6f} "
        f"(base={dominio.distance_limit_base:.6f})"
    )
    print(f"Avaliações totais: {problema.n_calls - calls_before}")
    print(f"Avaliações válidas: {problema.n_valid - valid_before}")
    print(
        f"Candidatos retidos: "
        f"{len(problema.seed_candidate_pools.get(int(SEED), []))}"
    )
    print(f"Alpha ML: {best_de['alpha']:.6f}°")
    print(f"CL ML: {best_de['CL_pred']:.8f}")
    print(f"CD ML: {best_de['CD_pred']:.8f}")
    print(f"CM ML: {best_de['CM_pred']:.8f}")

    # Validação física Top-N por polar completa no XFOIL.

    df_xfoil, best_xfoil_row, best_xfoil_candidate = (

        validar_melhores_candidatos_xfoil(

            problema=problema,

            dominio=dominio,

            t_min=t_min,

            t_max=t_max,

        )

    )

    best_vector = np.asarray(best_xfoil_candidate["vector"], dtype=float)

    au, al, re_value = problema.decode(best_vector)

    valid, geom_info = validar_geometria(

        au,

        al,

        dominio,

        t_min,

        t_max,

    )

    if not valid:

        raise RuntimeError(

            "O candidato vencedor no XFOIL falhou na validação geométrica final."

        )

    polar_ml_final, best_ml_final = problema.predict_polar(

        au,

        al,

        re_value,

    )

    polar_ml_final.to_csv(

        OUTPUT_DIR / "polar_surrogate_aerofolio_otimizado.csv",

        index=False,

    )

    print("\n" + "=" * 88)

    print("ÓTIMO FINAL DE CL CONFIRMADO PELO XFOIL EM ALPHA=6°")

    print("=" * 88)

    print(f"Candidato Top-N: {int(best_xfoil_row['candidate_id'])}")
    print(f"Seed de origem: {best_xfoil_row.get('origin_seed')}")

    print(f"Re: {re_value:.0f}")

    print(

        f"Surrogate: CL={float(best_xfoil_row['CL_pred_opt']):.8f} "

        f"em alpha={float(best_xfoil_row['alpha_pred_opt']):.4f}°"

    )

    print(

        f"XFOIL: CL={float(best_xfoil_row['CL_xfoil_opt']):.8f} "

        f"em alpha={float(best_xfoil_row['alpha_xfoil_opt']):.4f}°"

    )

    print(f"XFOIL CL: {float(best_xfoil_row['CL_xfoil_opt']):.8f}")

    print(f"XFOIL CD: {float(best_xfoil_row['CD_xfoil_opt']):.8f}")

    print(f"XFOIL CM: {float(best_xfoil_row['CM_xfoil_opt']):.8f}")

    print(

        f"Distância ao treino: {geom_info['distance_to_train']:.6f} / "

        f"limite {dominio.distance_limit:.6f}"

    )

    print(

        f"Espessura máxima: {100*geom_info['max_thickness']:.4f}% c"

    )

    print("\nCST extradorso:")

    for i, v in enumerate(au):

        print(f"Au{i} = {v:.10f}")

    print("\nCST intradorso:")

    for i, v in enumerate(al):

        print(f"Al{i} = {v:.10f}")

    result_dict = {
        "objective": "CL",

        **{f"Au{i}": float(au[i]) for i in range(7)},

        **{f"Al{i}": float(al[i]) for i in range(7)},

        "DeltaTE_upper": 0.0,

        "DeltaTE_lower": 0.0,

        "Re": float(re_value),

        "alpha_pred_opt": float(best_xfoil_row["alpha_pred_opt"]),

        "CL_pred_opt": float(best_xfoil_row["CL_pred_opt"]),

        "CD_pred_opt": float(best_xfoil_row["CD_pred_opt"]),

        "CM_pred_opt": float(best_xfoil_row["CM_pred_opt"]),

        "CL_CD_pred_opt": float(best_xfoil_row["CL_CD_pred_opt"]),

        "alpha_xfoil_opt": float(best_xfoil_row["alpha_xfoil_opt"]),

        "CL_xfoil_opt": float(best_xfoil_row["CL_xfoil_opt"]),

        "CD_xfoil_opt": float(best_xfoil_row["CD_xfoil_opt"]),

        "CM_xfoil_opt": float(best_xfoil_row["CM_xfoil_opt"]),

        "CL_CD_xfoil_opt": float(best_xfoil_row["CL_CD_xfoil_opt"]),

        "xfoil_candidate_id": int(best_xfoil_row["candidate_id"]),
        "origin_seed": None if pd.isna(best_xfoil_row.get("origin_seed")) else int(best_xfoil_row.get("origin_seed")),
        "best_de_seed": int(best_seed),
        "optimization_seed": int(SEED),

        "xfoil_points": int(best_xfoil_row["n_xfoil_points"]),

        "delta_alpha_opt_deg": float(best_xfoil_row["delta_alpha_opt_deg"]),

        "min_thickness": float(geom_info["min_thickness"]),

        "max_thickness": float(geom_info["max_thickness"]),

        "distance_to_train": float(geom_info["distance_to_train"]),

        "distance_limit": float(dominio.distance_limit),

        "valid_geometry": True,

        "optimizer_success": bool(result.success),
        "optimization_runs": 1,
        "cst_bound_expansion": float(CST_BOUND_EXPANSION),
        "distance_limit_multiplier": float(DISTANCE_LIMIT_MULTIPLIER),
        "max_abs_camber": float(MAX_ABS_CAMBER),
        "max_abs_camber_pct": float(100.0 * MAX_ABS_CAMBER),
        "use_cm_constraint": bool(USE_CM_CONSTRAINT),
        "use_cd_floor_constraint": bool(USE_CD_FLOOR_CONSTRAINT),

        "objective_calls": int(problema.n_calls),

        "valid_calls": int(problema.n_valid),

        "surrogate_alpha_step": float(SURROGATE_ALPHA_STEP),

        "xfoil_alpha_step": float(XFOIL_ALPHA_STEP),

        "top_n_xfoil": int(TOP_SURROGATE_TO_VALIDATE),

    }

    pd.DataFrame([result_dict]).to_csv(

        OUTPUT_DIR / "resultado_otimizacao.csv",

        index=False,

    )

    with (OUTPUT_DIR / "resultado_otimizacao.json").open(

        "w", encoding="utf-8"

    ) as f:

        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    salvar_dat(

        OUTPUT_DIR / "aerofolio_otimizado.dat",

        au,

        al,

    )

    print("\nArquivos salvos em:")

    print(OUTPUT_DIR)

    print(" - resultado_otimizacao.csv")

    print(" - resultado_otimizacao.json")

    print(" - aerofolio_otimizado.dat")

    print(" - polar_surrogate_melhor_de.csv")
    print(" - resumo_de_exploratorio.csv")

    print(" - polar_surrogate_aerofolio_otimizado.csv")

    print(" - validacao_candidatos_xfoil_polar.csv")

    print(" - ranking_candidatos_xfoil.csv")

    print(

        "\nO aerofolio_otimizado.dat corresponde à geometria com maior "

        "CL máximo na polar XFOIL entre os candidatos Top-N do surrogate."

    )



if __name__ == "__main__":

    main()
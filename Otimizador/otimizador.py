r"""
otimizacao_aerofolio.py

Otimização de aerofólio CST usando modelos substitutos XGBoost.

Objetivo:
    maximizar CL/CD

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
8) alpha entre 0 e 12 graus;
9) CD previsto não inferior ao menor CD positivo observado na base;
10) predição de CL/CD somente para candidatos válidos;
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

# Caminho verdadeiro do CSV de referência usado no repositório.
TRAIN_REFERENCE_CSV = ROOT / "Output_dados" / "ml_preparado" / "train.csv"

# Cria a subpasta dentro de Output_dados para os arquivos do otimizador.
OUTPUT_DIR = ROOT / "Output_dados" / "otimizacao"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Condição aerodinâmica
RE_FIXED = 250_000.0
RE_MIN = 200_000.0
RE_MAX = 300_000.0

OPTIMIZE_ALPHA = True
ALPHA_FIXED = 6.0
ALPHA_MIN = 0.0
ALPHA_MAX = 12.0

# CST clássico para aerofólio
N1 = 0.5
N2 = 1.0

# Discretização geométrica
N_GEOM_POINTS = 301

# Distância geométrica:
# percentil das distâncias "perfil de treino -> vizinho de treino mais próximo".
# 95% é conservador sem ser excessivamente restritivo.
DISTANCE_PERCENTILE = 80.0

# Pequena tolerância apenas numérica.
GEOM_TOL = 1e-8

# Otimizador
SEED = 42
POP_SIZE = 20
MAX_ITER = 500
TOL = 1e-7
POLISH = True
WORKERS = 1  
# Inicialização factível do Differential Evolution
# A população nasce próxima de geometrias reais da base de treino.
INIT_PERTURBATION_STD = 0.08
INIT_MAX_ATTEMPTS_PER_MEMBER = 300

# Penalizações graduais para guiar o algoritmo de volta à região factível.
PENALTY_BASE = 1e5
PENALTY_SCALE = 1e4


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
XFOIL_TIMEOUT = 120

TOP_SURROGATE_TO_VALIDATE = 20
CANDIDATE_POOL_SIZE = 100
CANDIDATE_DUPLICATE_TOL = 1e-5

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
    p = Path(TRAIN_REFERENCE_CSV)
    if not p.exists():
        raise FileNotFoundError(
            f"CSV de referência não encontrado no caminho do repositório:\n{p}"
        )
    return p


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
        self.min_cst = geom.min()
        self.max_cst = geom.max()

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
        self.distance_limit = float(
            np.percentile(nearest, DISTANCE_PERCENTILE)
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
    info["min_thickness"] = float(np.min(thickness))
    info["max_thickness"] = float(np.max(thickness))

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

    if info["distance_to_train"] > dominio.distance_limit:
        info["motivo"] = "distância geométrica excessiva ao domínio de treino"
        return False, info

    info["motivo"] = "valido"
    return True, info



def gerar_populacao_inicial_factivel(
    dominio: DominioGeometrico,
    t_min: float,
    t_max: float,
    optimize_alpha: bool = OPTIMIZE_ALPHA,
    seed: int = SEED,
):
    """
    Cria a população inicial do Differential Evolution a partir de
    geometrias REAIS do conjunto de referência, com pequenas perturbações
    no espaço CST padronizado.

    Isso evita iniciar a busca em pontos aleatórios do hipercubo de 14
    dimensões, onde a maior parte dos candidatos pode estar fora do domínio
    aprendido pelo modelo substituto.
    """
    rng = np.random.default_rng(seed)

    n_dim = 15 if optimize_alpha else 14

    # scipy usa população ~ popsize * número de variáveis
    target_size = max(5, POP_SIZE * n_dim)

    raw_geom = dominio.geom[CST_FEATURES].to_numpy(dtype=float)
    z_geom = dominio.scaler.transform(raw_geom)

    population = []

    # Primeiro incluímos perfis reais sem perturbação.
    order = rng.permutation(len(raw_geom))
    for idx in order:
        cst = raw_geom[idx].copy()
        au, al = cst[:7], cst[7:]

        valid, _ = validar_geometria(
            au, al, dominio, t_min, t_max
        )
        if not valid:
            continue

        if optimize_alpha:
            alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX)
            member = np.concatenate([cst, [alpha]])
        else:
            member = cst

        population.append(member)

        if len(population) >= target_size:
            break

    # Completa a população com perturbações locais de perfis reais.
    attempts = 0
    max_attempts = target_size * INIT_MAX_ATTEMPTS_PER_MEMBER

    while len(population) < target_size and attempts < max_attempts:
        attempts += 1

        idx = rng.integers(0, len(z_geom))
        z_candidate = z_geom[idx].copy()

        perturb = rng.normal(
            loc=0.0,
            scale=INIT_PERTURBATION_STD,
            size=z_candidate.shape
        )
        z_candidate = z_candidate + perturb

        cst = dominio.scaler.inverse_transform(
            z_candidate.reshape(1, -1)
        )[0]

        # Mantém rigorosamente os bounds individuais.
        mins = dominio.min_cst[CST_FEATURES].to_numpy(dtype=float)
        maxs = dominio.max_cst[CST_FEATURES].to_numpy(dtype=float)
        cst = np.clip(cst, mins, maxs)

        au, al = cst[:7], cst[7:]

        valid, _ = validar_geometria(
            au, al, dominio, t_min, t_max
        )
        if not valid:
            continue

        if optimize_alpha:
            alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX)
            member = np.concatenate([cst, [alpha]])
        else:
            member = cst

        population.append(member)

    if len(population) < 5:
        raise RuntimeError(
            "Não foi possível gerar uma população inicial factível. "
            "Verifique as restrições geométricas e a base de referência."
        )

    if len(population) < target_size:
        warnings.warn(
            f"População factível gerada com {len(population)} membros "
            f"(alvo era {target_size}). O scipy ainda pode prosseguir."
        )

    population = np.asarray(population, dtype=float)

    print(
        f"\nPopulação inicial factível: "
        f"{len(population)} candidatos válidos."
    )

    return population


# ======================================================================
# PROBLEMA DE OTIMIZAÇÃO
# ======================================================================

class ProblemaOtimizacao:
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
        self.best_candidates = []

    def decode(self, vector):
        vector = np.asarray(vector, dtype=float)

        au = vector[:7]
        al = vector[7:14]

        if OPTIMIZE_ALPHA:
            alpha = float(vector[14])
        else:
            alpha = float(ALPHA_FIXED)

        re_value = float(RE_FIXED)

        return au, al, re_value, alpha

    def predict(self, au, al, re_value, alpha):
        X_cl = montar_dataframe_modelo(
            self.model_cl, au, al, re_value, alpha
        )
        X_cd = montar_dataframe_modelo(
            self.model_cd, au, al, re_value, alpha
        )
        X_cm = montar_dataframe_modelo(
            self.model_cm, au, al, re_value, alpha
        )

        cl = float(np.asarray(self.model_cl.predict(X_cl)).ravel()[0])
        cd = float(np.asarray(self.model_cd.predict(X_cd)).ravel()[0])
        cm = float(np.asarray(self.model_cm.predict(X_cm)).ravel()[0])

        return cl, cd, cm

    def registrar_candidato(self, vector, cl, cd, cm):
        """Guarda os melhores candidatos do surrogate para validação XFOIL."""
        if not (np.isfinite(cl) and np.isfinite(cd) and np.isfinite(cm)):
            return
        if cd <= 0:
            return

        eff = float(cl / cd)
        vec = np.asarray(vector, dtype=float).copy()

        for item in self.best_candidates:
            if np.linalg.norm(item["vector"] - vec) <= CANDIDATE_DUPLICATE_TOL:
                if eff > item["CL_CD_pred"]:
                    item["vector"] = vec
                    item["CL_pred"] = float(cl)
                    item["CD_pred"] = float(cd)
                    item["CM_pred"] = float(cm)
                    item["CL_CD_pred"] = eff
                return

        self.best_candidates.append({
            "vector": vec,
            "CL_pred": float(cl),
            "CD_pred": float(cd),
            "CM_pred": float(cm),
            "CL_CD_pred": eff,
        })

        self.best_candidates.sort(
            key=lambda item: item["CL_CD_pred"],
            reverse=True,
        )
        self.best_candidates = self.best_candidates[:CANDIDATE_POOL_SIZE]

    def objective(self, vector):
        self.n_calls += 1

        au, al, re_value, alpha = self.decode(vector)

        # Reynolds e alpha normalmente já estão nos bounds,
        # mas mantemos a verificação por segurança.
        if not (RE_MIN <= re_value <= RE_MAX):
            return PENALTY_BASE + PENALTY_SCALE * abs(
                re_value - np.clip(re_value, RE_MIN, RE_MAX)
            )

        if not (ALPHA_MIN <= alpha <= ALPHA_MAX):
            return PENALTY_BASE + PENALTY_SCALE * abs(
                alpha - np.clip(alpha, ALPHA_MIN, ALPHA_MAX)
            )

        valid, info = validar_geometria(
            au,
            al,
            self.dominio,
            self.t_min,
            self.t_max,
        )

        if not valid:
            # Penalização graduada: candidatos "menos ruins" recebem
            # penalidade menor que candidatos muito afastados.
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

        cl, cd, cm = self.predict(au, al, re_value, alpha)

        if not (np.isfinite(cl) and np.isfinite(cd) and np.isfinite(cm)):
            return PENALTY_BASE

        cd_min_local = calcular_cd_minimo_local(
            self.df_ref,
            re_value=re_value,
            alpha=alpha,
            cd_global_min=self.cd_min_global,
        )

        if cd < cd_min_local:
            return (
                PENALTY_BASE
                + PENALTY_SCALE
                * (cd_min_local - cd)
                / max(cd_min_local, 1e-12)
            )

        cm_min_local, cm_max_local = calcular_limites_cm_local(
            self.df_ref,
            re_value=re_value,
            alpha=alpha,
        )

        if cm_min_local is not None and cm < cm_min_local:
            escala_cm = max(abs(cm_min_local), 0.05)
            return (
                PENALTY_BASE
                + PENALTY_SCALE
                * (cm_min_local - cm)
                / escala_cm
            )

        if cm_max_local is not None and cm > cm_max_local:
            escala_cm = max(abs(cm_max_local), 0.05)
            return (
                PENALTY_BASE
                + PENALTY_SCALE
                * (cm - cm_max_local)
                / escala_cm
            )

        self.n_valid += 1
        self.registrar_candidato(vector, cl, cd, cm)

        return -(cl / cd)



# ======================================================================
# VALIDAÇÃO FINAL DOS MELHORES CANDIDATOS COM XFOIL
# ======================================================================

def localizar_xfoil() -> Path:
    p = Path(XFOIL_EXE)
    if not p.exists():
        raise FileNotFoundError(f"XFOIL não encontrado em: {p}")
    return p


def ler_polar_xfoil_simples(path: Path):
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

    alpha, cl, cd, _, cm = rows[-1]

    if not (np.isfinite(cl) and np.isfinite(cd) and np.isfinite(cm)):
        return None
    if cd <= 0:
        return None

    return {
        "alpha_xfoil": float(alpha),
        "CL_xfoil": float(cl),
        "CD_xfoil": float(cd),
        "CM_xfoil": float(cm),
        "CL_CD_xfoil": float(cl / cd),
    }


def executar_xfoil_alpha(
    xfoil_exe: Path,
    dat_path: Path,
    alpha: float,
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
        f"ALFA {alpha:.8f}",
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
        return None

    (work_dir / "stdout.txt").write_text(
        proc.stdout or "",
        encoding="utf-8",
        errors="ignore",
    )

    if proc.returncode != 0:
        return None

    return ler_polar_xfoil_simples(polar_path)


def selecionar_candidatos_distintos(
    problema,
    dominio,
    t_min,
    t_max,
):
    selected = []

    for item in problema.best_candidates:
        vec = item["vector"]
        au, al, re_value, alpha = problema.decode(vec)

        valid, info = validar_geometria(
            au,
            al,
            dominio,
            t_min,
            t_max,
        )
        if not valid:
            continue

        duplicate = False
        for prev in selected:
            prev_vec = prev["vector"]
            geom_dist = np.linalg.norm(vec[:14] - prev_vec[:14])

            if OPTIMIZE_ALPHA:
                alpha_dist = abs(float(vec[14] - prev_vec[14]))
            else:
                alpha_dist = 0.0

            if geom_dist < 1e-3 and alpha_dist < 0.10:
                duplicate = True
                break

        if duplicate:
            continue

        new_item = dict(item)
        new_item["geom_info"] = info
        selected.append(new_item)

        if len(selected) >= TOP_SURROGATE_TO_VALIDATE:
            break

    return selected


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

    print("\\n" + "=" * 78)
    print("VALIDAÇÃO DOS MELHORES CANDIDATOS NO XFOIL")
    print("=" * 78)
    print(f"Candidatos selecionados: {len(candidates)}")

    rows = []

    for idx, item in enumerate(candidates, start=1):
        au, al, re_value, alpha = problema.decode(item["vector"])

        dat_path = XFOIL_VALIDATION_DIR / f"candidato_{idx:03d}.dat"
        salvar_dat(dat_path, au, al)

        xf = executar_xfoil_alpha(
            xfoil_exe=xfoil_exe,
            dat_path=dat_path,
            alpha=alpha,
            re_value=re_value,
            candidate_id=idx,
        )

        cm_min_local, cm_max_local = calcular_limites_cm_local(
            problema.df_ref,
            re_value=re_value,
            alpha=alpha,
        )

        row = {
            "candidate_id": idx,
            **{f"Au{i}": float(au[i]) for i in range(7)},
            **{f"Al{i}": float(al[i]) for i in range(7)},
            "Re": float(re_value),
            "alpha_pred": float(alpha),
            "CL_pred": float(item["CL_pred"]),
            "CD_pred": float(item["CD_pred"]),
            "CM_pred": float(item["CM_pred"]),
            "CL_CD_pred": float(item["CL_CD_pred"]),
            "distance_to_train": float(
                item["geom_info"]["distance_to_train"]
            ),
            "max_thickness": float(
                item["geom_info"]["max_thickness"]
            ),
            "CM_local_min": cm_min_local,
            "CM_local_max": cm_max_local,
            "xfoil_converged": xf is not None,
        }

        if xf is not None:
            row.update(xf)
            row["erro_CL_pct"] = (
                100.0 * abs(row["CL_pred"] - row["CL_xfoil"])
                / max(abs(row["CL_xfoil"]), 1e-12)
            )
            row["erro_CD_pct"] = (
                100.0 * abs(row["CD_pred"] - row["CD_xfoil"])
                / max(abs(row["CD_xfoil"]), 1e-12)
            )
            row["erro_CL_CD_pct"] = (
                100.0
                * abs(row["CL_CD_pred"] - row["CL_CD_xfoil"])
                / max(abs(row["CL_CD_xfoil"]), 1e-12)
            )

            print(
                f"[{idx:02d}/{len(candidates)}] "
                f"surrogate={row['CL_CD_pred']:.2f} | "
                f"XFOIL={row['CL_CD_xfoil']:.2f} | "
                f"alpha={alpha:.3f}°"
            )
        else:
            print(
                f"[{idx:02d}/{len(candidates)}] "
                "XFOIL não convergiu."
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv = OUTPUT_DIR / "validacao_candidatos_xfoil.csv"
    df.to_csv(output_csv, index=False)

    converged = df[df["xfoil_converged"] == True].copy()
    if "CL_CD_xfoil" in converged.columns:
        converged = converged[np.isfinite(converged["CL_CD_xfoil"])]

    if converged.empty:
        raise RuntimeError(
            "Nenhum dos candidatos selecionados convergiu no XFOIL."
        )

    best_idx = converged["CL_CD_xfoil"].idxmax()
    best_row = converged.loc[best_idx]

    candidate_id = int(best_row["candidate_id"])
    best_candidate = candidates[candidate_id - 1]

    return df, best_row, best_candidate

# ======================================================================
# EXPORTAÇÃO DO RESULTADO
# ======================================================================

def salvar_dat(path: Path, au: np.ndarray, al: np.ndarray):
    """
    Formato .dat: extradorso BF->BA, depois intradorso BA->BF.
    """
    x, yu, yl = reconstruir_geometria(au, al, n_points=401)

    xu = x[::-1]
    yuu = yu[::-1]

    # evita duplicar o BA
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
    print("=" * 78)
    print("OTIMIZAÇÃO DE AEROFÓLIO CST COM XGBOOST")
    print("=" * 78)

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
    t_min, t_max, thicknesses = calcular_limites_espessura(df_ref)
    cd_min_global = calcular_cd_minimo_global(df_ref)

    print("\n" + "-" * 78)
    print("LIMITES EXTRAÍDOS DA BASE")
    print("-" * 78)
    print(f"Perfis geométricos únicos: {len(dominio.geom)}")
    print(f"Espessura máxima observada mínima: {100*t_min:.3f}% c")
    print(f"Espessura máxima observada máxima: {100*t_max:.3f}% c")
    print(f"CD mínimo positivo global observado: {cd_min_global:.8f}")
    print(
        f"Limite de distância CST ao treino "
        f"(P{DISTANCE_PERCENTILE:.0f} do vizinho mais próximo): "
        f"{dominio.distance_limit:.6f}"
    )
    print(
        f"CD local: P{CD_LOCAL_PERCENTILE:.0f}, "
        f"janela inicial alpha ±{CD_LOCAL_ALPHA_WINDOW:.2f}° "
        f"e Re ±{CD_LOCAL_RE_WINDOW:.0f}"
    )
    print(
        f"CM local: P{CM_LOCAL_LOWER_PERCENTILE:.0f}-"
        f"P{CM_LOCAL_UPPER_PERCENTILE:.0f}, "
        f"janela inicial alpha ±{CM_LOCAL_ALPHA_WINDOW:.2f}° "
        f"e Re ±{CM_LOCAL_RE_WINDOW:.0f}"
    )

    print("\nLimites CST:")
    for c in CST_FEATURES:
        print(
            f"  {c:>4s}: "
            f"[{dominio.min_cst[c]: .8f}, {dominio.max_cst[c]: .8f}]"
        )

    bounds = dominio.bounds_scipy()

    if OPTIMIZE_ALPHA:
        bounds.append((ALPHA_MIN, ALPHA_MAX))
        print(f"\nAlpha será otimizado em [{ALPHA_MIN}, {ALPHA_MAX}] graus.")
    else:
        print(f"\nAlpha fixo: {ALPHA_FIXED} graus.")

    print(f"Reynolds fixo: {RE_FIXED:.0f}")

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

    print("\n" + "=" * 78)
    print("GERANDO POPULAÇÃO INICIAL FACTÍVEL")
    print("=" * 78)

    init_population = gerar_populacao_inicial_factivel(
        dominio=dominio,
        t_min=t_min,
        t_max=t_max,
        optimize_alpha=OPTIMIZE_ALPHA,
        seed=SEED,
    )

    print("\n" + "=" * 78)
    print("INICIANDO DIFFERENTIAL EVOLUTION")
    print("=" * 78)

    result = differential_evolution(
        problema.objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=MAX_ITER,
        popsize=POP_SIZE,
        tol=TOL,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=SEED,
        init=init_population,
        polish=POLISH,
        workers=WORKERS,
        updating="immediate" if WORKERS == 1 else "deferred",
        disp=True,
    )

    au, al, re_value, alpha = problema.decode(result.x)

    valid, geom_info = validar_geometria(
        au,
        al,
        dominio,
        t_min,
        t_max,
    )

    print("\n" + "=" * 78)
    print("DIAGNÓSTICO DA SOLUÇÃO FINAL")
    print("=" * 78)
    print(f"Sucesso scipy: {result.success}")
    print(f"Mensagem: {result.message}")
    print(f"Geometria válida: {valid}")
    print(f"Motivo: {geom_info.get('motivo')}")
    print(f"Gap BA: {geom_info.get('le_gap', np.nan)}")
    print(f"Gap BF: {geom_info.get('te_gap', np.nan)}")
    print(f"Espessura mínima: {geom_info.get('min_thickness', np.nan)}")
    print(f"Espessura máxima: {geom_info.get('max_thickness', np.nan)}")
    print(
        f"Distância ao treino: "
        f"{geom_info.get('distance_to_train', np.nan):.6f} "
        f"/ limite {dominio.distance_limit:.6f}"
    )
    print(f"Avaliações totais: {problema.n_calls}")
    print(f"Avaliações válidas: {problema.n_valid}")

    if not valid:
        print("\n" + "!" * 78)
        print("OTIMIZAÇÃO ENCERRADA SEM SOLUÇÃO GEOMETRICAMENTE VÁLIDA")
        print("!" * 78)
        print(
            "O scipy pode reportar success=True por convergência numérica, "
            "mas esta solução NÃO será aceita, exportada ou avaliada pelos "
            "modelos aerodinâmicos."
        )
        print("Restrição responsável:", geom_info.get("motivo"))
        return

    cl, cd, cm = problema.predict(au, al, re_value, alpha)

    if not (np.isfinite(cl) and np.isfinite(cd) and np.isfinite(cm)):
        raise RuntimeError(
            "A geometria é válida, porém alguma predição do surrogate "
            "não é finita."
        )

    cd_min_final = calcular_cd_minimo_local(
        df_ref,
        re_value=re_value,
        alpha=alpha,
        cd_global_min=cd_min_global,
    )

    if cd < cd_min_final:
        print("\n" + "!" * 78)
        print("SOLUÇÃO REJEITADA POR CD ABAIXO DO LIMITE LOCAL")
        print("!" * 78)
        print(f"CD previsto: {cd:.8f}")
        print(f"CD mínimo local admissível: {cd_min_final:.8f}")
        return

    cm_min_final, cm_max_final = calcular_limites_cm_local(
        df_ref,
        re_value=re_value,
        alpha=alpha,
    )

    if cm_min_final is not None and not (cm_min_final <= cm <= cm_max_final):
        print("\n" + "!" * 78)
        print("SOLUÇÃO REJEITADA POR CM FORA DA FAIXA LOCAL")
        print("!" * 78)
        print(f"CM previsto: {cm:.8f}")
        print(
            f"Faixa local admissível: "
            f"[{cm_min_final:.8f}, {cm_max_final:.8f}]"
        )
        return

    efficiency = cl / cd

    print("\n" + "=" * 78)
    print("RESULTADO VÁLIDO DA OTIMIZAÇÃO")
    print("=" * 78)
    print(f"Re: {re_value:.0f}")
    print(f"alpha: {alpha:.6f} deg")
    print(f"CL: {cl:.8f}")
    print(f"CD: {cd:.8f}")
    print(f"CM: {cm:.8f}")
    print(f"CL/CD: {efficiency:.8f}")
    print(
        f"Distância ao treino: "
        f"{geom_info['distance_to_train']:.6f} "
        f"/ limite {dominio.distance_limit:.6f}"
    )
    print(
        f"Espessura máxima: "
        f"{100*geom_info['max_thickness']:.4f}% c"
    )

    print("\nCST extradorso:")
    for i, v in enumerate(au):
        print(f"Au{i} = {v:.10f}")

    print("\nCST intradorso:")
    for i, v in enumerate(al):
        print(f"Al{i} = {v:.10f}")

    # Validação XFOIL dos melhores candidatos do surrogate
    df_xfoil, best_xfoil_row, best_xfoil_candidate = (
        validar_melhores_candidatos_xfoil(
            problema=problema,
            dominio=dominio,
            t_min=t_min,
            t_max=t_max,
        )
    )

    best_vector = best_xfoil_candidate["vector"]
    au, al, re_value, alpha = problema.decode(best_vector)

    valid, geom_info = validar_geometria(
        au,
        al,
        dominio,
        t_min,
        t_max,
    )

    cl, cd, cm = problema.predict(
        au,
        al,
        re_value,
        alpha,
    )
    efficiency = cl / cd

    print("\n" + "=" * 78)
    print("MELHOR CANDIDATO CONFIRMADO PELO XFOIL")
    print("=" * 78)
    print(f"Candidato: {int(best_xfoil_row['candidate_id'])}")
    print(f"Re: {re_value:.0f}")
    print(f"alpha: {alpha:.6f} deg")
    print(f"Surrogate CL/CD: {efficiency:.8f}")
    print(f"XFOIL CL: {float(best_xfoil_row['CL_xfoil']):.8f}")
    print(f"XFOIL CD: {float(best_xfoil_row['CD_xfoil']):.8f}")
    print(f"XFOIL CM: {float(best_xfoil_row['CM_xfoil']):.8f}")
    print(
        f"XFOIL CL/CD: "
        f"{float(best_xfoil_row['CL_CD_xfoil']):.8f}"
    )

    result_dict = {
        **{f"Au{i}": float(au[i]) for i in range(7)},
        **{f"Al{i}": float(al[i]) for i in range(7)},
        "DeltaTE_upper": 0.0,
        "DeltaTE_lower": 0.0,
        "Re": re_value,
        "alpha": alpha,
        "CL_pred": cl,
        "CD_pred": cd,
        "CM_pred": cm,
        "CL_CD_pred": efficiency,
        "CL_xfoil": float(best_xfoil_row["CL_xfoil"]),
        "CD_xfoil": float(best_xfoil_row["CD_xfoil"]),
        "CM_xfoil": float(best_xfoil_row["CM_xfoil"]),
        "CL_CD_xfoil": float(best_xfoil_row["CL_CD_xfoil"]),
        "xfoil_candidate_id": int(best_xfoil_row["candidate_id"]),
        "CM_local_min": None if cm_min_final is None else float(cm_min_final),
        "CM_local_max": None if cm_max_final is None else float(cm_max_final),
        "min_thickness": geom_info.get("min_thickness"),
        "max_thickness": geom_info.get("max_thickness"),
        "distance_to_train": geom_info.get("distance_to_train"),
        "distance_limit": dominio.distance_limit,
        "valid_geometry": True,
        "optimizer_success": bool(result.success),
        "objective_calls": problema.n_calls,
        "valid_calls": problema.n_valid,
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

    print(
        "\nO aerofolio_otimizado.dat corresponde ao melhor candidato "
        "entre os melhores do surrogate que foram confirmados no XFOIL."
    )
    print(
        "Tabela completa da validação XFOIL:"
    )
    print(OUTPUT_DIR / "validacao_candidatos_xfoil.csv")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def ler_dat(arquivo):
    arquivo = Path(arquivo)

    pontos = []

    with open(
        arquivo,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for linha in f:

            linha = linha.replace(",", " ").strip()

            if not linha:
                continue

            partes = linha.split()

            if len(partes) < 2:
                continue

            try:
                x = float(partes[0])
                y = float(partes[1])
            except ValueError:
                continue

            # Ignora linhas de contagem Lednicer
            # Exemplo: 61 61
            if (
                abs(x - round(x)) < 1e-10
                and abs(y - round(y)) < 1e-10
                and x >= 5
                and y >= 5
            ):
                continue

            if np.isfinite(x) and np.isfinite(y):
                pontos.append([x, y])

    if len(pontos) == 0:
        raise ValueError(
            f"Nenhum ponto encontrado em {arquivo}"
        )

    return np.asarray(pontos, dtype=float)


def normalizar_para_comparacao(data):
    """
    Normalização simples apenas para comparação visual.

    Funciona tanto para:
        x = 0 ... 1

    quanto:
        x = 0 ... 100
    """

    data = data.copy()

    x = data[:, 0]
    y = data[:, 1]

    xmin = np.min(x)
    xmax = np.max(x)

    chord = xmax - xmin

    if chord <= 0:
        raise ValueError("Corda inválida.")

    # Normaliza x
    x_norm = (x - xmin) / chord

    # Escala y pela mesma corda
    y_norm = y / chord

    return np.column_stack([
        x_norm,
        y_norm
    ]), chord


def mostrar_info(nome, bruto, normalizado, chord):

    print("\n" + "=" * 60)

    print(nome)

    print(f"Número de pontos: {len(bruto)}")

    print("\nORIGINAL:")

    print(
        f"x: {bruto[:,0].min():.8f} "
        f"até {bruto[:,0].max():.8f}"
    )

    print(
        f"y: {bruto[:,1].min():.8f} "
        f"até {bruto[:,1].max():.8f}"
    )

    print(
        f"\nCorda detectada: {chord:.8f}"
    )

    print("\nNORMALIZADO:")

    print(
        f"x/c: {normalizado[:,0].min():.8f} "
        f"até {normalizado[:,0].max():.8f}"
    )

    print(
        f"y/c: {normalizado[:,1].min():.8f} "
        f"até {normalizado[:,1].max():.8f}"
    )


# ============================================================
# CAMINHOS
# ============================================================

arquivo_original = Path(
    r"C:\Ciencia de Dados\TCC\Airfoils\clarky.dat"
)

arquivo_cst = Path(
    r"C:\Ciencia de Dados\TCC\Airfoils\cst_reconstruidos\clarky_cst.dat"
)


# ============================================================
# LEITURA
# ============================================================

original_bruto = ler_dat(
    arquivo_original
)

cst_bruto = ler_dat(
    arquivo_cst
)


# ============================================================
# NORMALIZAÇÃO
# ============================================================

original, chord_original = normalizar_para_comparacao(
    original_bruto
)

cst, chord_cst = normalizar_para_comparacao(
    cst_bruto
)


# ============================================================
# INFORMAÇÕES
# ============================================================

mostrar_info(
    "NACA 23015 ORIGINAL",
    original_bruto,
    original,
    chord_original
)

mostrar_info(
    "NACA 23015 CST",
    cst_bruto,
    cst,
    chord_cst
)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)


# Original
ax.plot(
    original[:, 0],
    original[:, 1],
    "-",
    linewidth=2,
    label="Original"
)


# CST
ax.plot(
    cst[:, 0],
    cst[:, 1],
    "o",
    markersize=3,
    label="CST"
)


# ============================================================
# FORMATAÇÃO
# ============================================================

ax.set_title(
    "NACA 23015 — Original × CST"
)

ax.set_xlabel("x/c")
ax.set_ylabel("y/c")

ax.grid(
    True,
    alpha=0.3
)

ax.legend()

ax.set_aspect(
    "equal",
    adjustable="datalim"
)

# IMPORTANTE:
# não colocar xlim manual nesta etapa

ax.relim()
ax.autoscale_view()

plt.tight_layout()

plt.show()
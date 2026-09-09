from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CD_OUTPUT_DIR = ROOT / "Output_dados" / "otimizacao_CD"
CL_OUTPUT_DIR = ROOT / "Output_dados" / "otimizacao_CL"

CD_DAT = CD_OUTPUT_DIR / "aerofolio_otimizado.dat"
CL_DAT = CL_OUTPUT_DIR / "aerofolio_otimizado.dat"


def ler_perfil_dat(path: Path):
    """
    Lê o arquivo .dat exportado pelos otimizadores e devolve
    duas listas de pontos: extradorso e intradorso.

    O formato do arquivo é:
        primeira linha: CST_OPTIMIZED
        seguida: upper surface (BF -> BA) e lower surface (BA -> BF)
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de perfil não encontrado: {path}")

    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines or lines[0] != "CST_OPTIMIZED":
        raise ValueError(f"Formato inesperado em {path}")

    # O gerador de .dat salva o extradorso primeiro (401 pontos) e
    # o intradorso em seguida (400 pontos), porque a rotina de escrita
    # usa x[::-1] para o extradorso e x[1:] para o intradorso.
    coords = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 2:
            continue
        x_val = float(parts[0])
        y_val = float(parts[1])
        coords.append((x_val, y_val))

    if len(coords) < 2:
        raise ValueError(f"Arquivo com poucos pontos: {path}")

    # Número de pontos conhecido a partir de salvar_dat():
    # upper -> 401 points, lower -> 400 points.
    upper = coords[:401]
    lower = coords[401:]

    upper_x = [p[0] for p in upper]
    upper_y = [p[1] for p in upper]
    lower_x = [p[0] for p in lower]
    lower_y = [p[1] for p in lower]

    return upper_x, upper_y, lower_x, lower_y


def plotar_perfil(path: Path, output_plot: Path, title: str):
    """
    Cria uma figura separada com o perfil da geometria CST.
    """
    upper_x, upper_y, lower_x, lower_y = ler_perfil_dat(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(upper_x, upper_y, color="tab:blue", linewidth=2, label="Extradorso")
    ax.plot(lower_x, lower_y, color="tab:red", linewidth=2, label="Intradorso")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(min(min(upper_y), min(lower_y)) - 0.03,
                 max(max(upper_y), max(lower_y)) + 0.03)

    fig.tight_layout()
    fig.savefig(output_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    outputs = []

    if CD_DAT.exists():
        outputs.append(
            {
                "path": CD_DAT,
                "plot": CD_OUTPUT_DIR / "perfil_otimizado_CD.png",
                "title": "Perfil otimizado — CD (Re=250k, alpha=6°)",
            }
        )
    else:
        print(f"Aviso: arquivo não encontrado para CD: {CD_DAT}")

    if CL_DAT.exists():
        outputs.append(
            {
                "path": CL_DAT,
                "plot": CL_OUTPUT_DIR / "perfil_otimizado_CL.png",
                "title": "Perfil otimizado — CL (Re=250k, alpha=6°)",
            }
        )
    else:
        print(f"Aviso: arquivo não encontrado para CL: {CL_DAT}")

    for item in outputs:
        try:
            plotar_perfil(item["path"], item["plot"], item["title"])
            print(f"Figura salva em: {item['plot']}")
        except Exception as exc:
            print(f"Erro ao plotar {item['path']}: {exc}")

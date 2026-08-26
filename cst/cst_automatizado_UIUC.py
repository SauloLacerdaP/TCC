import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import comb
from pathlib import Path

# =========================
# 1) FUNÇÕES MATEMÁTICAS CST
# =========================
def class_function(x: np.ndarray) -> np.ndarray:
    return np.sqrt(x) * (1 - x)

def bernstein_poly(i: int, n: int, x: np.ndarray) -> np.ndarray:
    return comb(n, i) * (x**i) * ((1 - x)**(n - i))

def cst_formula(x: np.ndarray, *coeffs: float) -> np.ndarray:
    n = len(coeffs) - 1
    shape = np.zeros_like(x, dtype=float)
    for i, a in enumerate(coeffs):
        shape += a * bernstein_poly(i, n, x)
    return class_function(x) * shape


# ==========================================
# 2) LEITURA UIUC NO FORMATO SELIG (1 BLOCO)
# ==========================================
def ler_uiuc_selig(dat_path: Path) -> np.ndarray | None:
    """
    Lê .dat do UIUC no formato SELIG (1 bloco):
      header
      x y
      x y
      ...
    Ignora linhas sujas (parênteses, textos, etc) e retorna array Nx2.
    """
    try:
        lines = dat_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        print(f"❌ Erro ao ler {dat_path.name}: {e}")
        return None

    pts = []
    for ln in lines[1:]:  # pula header
        s = ln.strip()
        if not s:
            continue
        s = s.replace(",", " ").replace("(", "").replace(")", "")
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        pts.append([x, y])

    if len(pts) < 20:
        print(f"❌ {dat_path.name}: poucos pontos numéricos válidos.")
        return None

    data = np.array(pts, dtype=float)

    # segurança: limita domínio
    data[:, 0] = np.clip(data[:, 0], 0.0, 1.0)

    # remove duplicatas exatas (TE/LE duplicados)
    seen = set()
    keep = []
    for i, (x, y) in enumerate(data):
        key = (round(x, 12), round(y, 12))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    data = data[np.array(keep, dtype=int)]

    return data


# ============================
# 3) PROCESSAMENTO DO AEROFÓLIO
# ============================
def processar_aerofolio_uiuc_selig(arquivo_dat, ordem: int = 6):
    arquivo_dat = Path(arquivo_dat)
    data = ler_uiuc_selig(arquivo_dat)
    if data is None:
        return None

    x, y = data[:, 0], data[:, 1]

    # separa no LE (x mínimo) -> upper: [0..idx_le] e lower: [idx_le..end]
    idx_le = int(np.argmin(x))
    if idx_le < 5 or idx_le > len(x) - 6:
        print(f"❌ {arquivo_dat.name}: LE em posição estranha (idx={idx_le}).")
        return None

    upper = data[:idx_le + 1]
    lower = data[idx_le:]

    xu, yu = upper[:, 0], upper[:, 1]
    xl, yl = lower[:, 0], lower[:, 1]

    # ordena por x crescente (ajuda muito o curve_fit)
    iu = np.argsort(xu)
    il = np.argsort(xl)
    xu, yu = xu[iu], yu[iu]
    xl, yl = xl[il], yl[il]

    if len(xu) < ordem + 2 or len(xl) < ordem + 2:
        print(f"❌ {arquivo_dat.name}: poucos pontos para ordem {ordem}.")
        return None

    p0 = [0.2] * (ordem + 1)

    try:
        coeffs_u, _ = curve_fit(cst_formula, xu, yu, p0=p0, maxfev=20000)
        coeffs_l, _ = curve_fit(cst_formula, xl, yl, p0=p0, maxfev=20000)
    except Exception as e:
        print(f"❌ Erro no curve_fit em {arquivo_dat.name}: {e}")
        return None

    return coeffs_u, coeffs_l


# ==========================
# 4) EXPORTAÇÃO PARA O XFOIL
# ==========================
def salvar_para_xfoil(nome, coeffs_u, coeffs_l, pontos: int = 120, out_dir="."):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0, 1, pontos)
    yu = cst_formula(x, *coeffs_u)
    yl = cst_formula(x, *coeffs_l)

    # contorno: TE(upper)->LE->TE(lower)
    x_final = np.concatenate([x[::-1], x[1:]])
    y_final = np.concatenate([yu[::-1], yl[1:]])

    arquivo_nome = out_dir / f"{nome}_cst.dat"
    with open(arquivo_nome, "w", encoding="utf-8") as f:
        f.write(f"{nome}_CST_SUAVIZADO\n")
        for xi, yi in zip(x_final, y_final):
            f.write(f" {xi:10.7f} {yi:10.7f}\n")

    print(f"✅ XFOIL: {arquivo_nome.name}")


# ===========================================
# 5) EXECUÇÃO EM LOTE + CSV ÚNICO NO FINAL
# ===========================================
def processar_diretorio_airfoils_uiuc_selig(
    diretorio_airfoils,
    ordem: int = 6,
    pontos_xfoil: int = 120,
    csv_nome: str = "database_aero.csv"
):
    diretorio_airfoils = Path(diretorio_airfoils)
    if not diretorio_airfoils.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {diretorio_airfoils}")

    dat_files = sorted(diretorio_airfoils.glob("*.dat"))
    print(f"🔎 Encontrados {len(dat_files)} arquivos .dat em {diretorio_airfoils}")

    xfoil_out = diretorio_airfoils / "xfoil_out"
    csv_path = diretorio_airfoils / csv_nome

    linhas_csv = []
    failed = []
    ok, fail = 0, 0

    for dat_file in dat_files:
        nome = dat_file.stem
        print(f"\n--- Processando: {dat_file.name} ---")

        res = processar_aerofolio_uiuc_selig(dat_file, ordem=ordem)
        if not res:
            fail += 1
            failed.append(nome)
            continue

        coeffs_u, coeffs_l = res
        salvar_para_xfoil(nome, coeffs_u, coeffs_l, pontos=pontos_xfoil, out_dir=xfoil_out)

        row = {"perfil": nome}
        for i, v in enumerate(coeffs_u):
            row[f"Au{i}"] = v
        for i, v in enumerate(coeffs_l):
            row[f"Al{i}"] = v
        linhas_csv.append(row)
        ok += 1

    if linhas_csv:
        df = pd.DataFrame(linhas_csv)
        au_cols = sorted([c for c in df.columns if c.startswith("Au")], key=lambda s: int(s[2:]))
        al_cols = sorted([c for c in df.columns if c.startswith("Al")], key=lambda s: int(s[2:]))
        df = df[["perfil"] + au_cols + al_cols]
        df.to_csv(csv_path, index=False)
        print(f"\n📄 CSV gerado: {csv_path}")
    else:
        print("\n⚠️ Nenhum perfil foi processado com sucesso. CSV não gerado.")

    print(f"\n✅ Concluído. Sucesso: {ok} | Falhas: {fail}")
    if failed:
        print(f"\n❌ Falharam ({len(failed)}):")
        for a in failed:
            print("  -", a)


if __name__ == "__main__":
    processar_diretorio_airfoils_uiuc_selig(
        r"C:\Ciencia de Dados\TCC\Airfoils",
        ordem=6,
        pontos_xfoil=120,
        csv_nome="database_aero.csv"
    )

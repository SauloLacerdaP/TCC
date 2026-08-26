from pathlib import Path
import subprocess
import csv
import shutil
import re
import time
import os

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

# Executável do XFOIL
XFOIL_EXE = Path(
    r"C:\Repositorios\TCC\xfoil\XFOIL\xfoil.exe"
)

# Pasta com os arquivos .dat
PASTA_PERFIS = Path(
    r"C:\Repositorios\TCC\lednicer_to_selig\Airfoils_Selig/"
)

# Pasta dos resultados
PASTA_RESULTADOS = Path(
    r"C:\Repositorios\TCC\Output_dados"
)

PASTA_ARQUIVOS_XFOIL = XFOIL_EXE.parent.parent / "arquivos_xfoil"

# Condições de simulação
REYNOLDS = [
    200000,
    250000,
    300000
]

MACH = 0.1

ALPHA_INICIAL = 0
ALPHA_FINAL = 12
PASSO_ALPHA = 1

ITERACOES = 500

# Timeout para cada Reynolds
TIMEOUT = 180

# ============================================================
# PARALELIZAÇÃO
# ============================================================

# Ryzen 5 5600X = 6 núcleos / 12 threads
# Começaria com 4.
MAX_WORKERS = 4


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def limpar_nome(nome):

    return re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        nome
    )


def gerar_lista_alpha():

    alphas = []

    alpha = ALPHA_INICIAL

    while alpha <= ALPHA_FINAL + 1e-9:

        alphas.append(
            round(alpha, 6)
        )

        alpha += PASSO_ALPHA

    return alphas


# ============================================================
# CRIAÇÃO DOS COMANDOS XFOIL
# ============================================================

def criar_comandos_xfoil(
    arquivo_dat,
    reynolds,
    arquivo_polar
):

    alphas = gerar_lista_alpha()

    comandos = []

    comandos.append(
        f"LOAD {arquivo_dat}"
    )

    comandos.append("PANE")

    comandos.append("OPER")

    comandos.append(
        f"VISC {reynolds}"
    )

    comandos.append(
        f"MACH {MACH}"
    )

    comandos.append(
        f"ITER {ITERACOES}"
    )

    # Ativa gravação da polar
    comandos.append("PACC")

    comandos.append(
        arquivo_polar
    )

    # dump file vazio
    comandos.append("")

    # Ângulos individualmente
    for alpha in alphas:

        comandos.append(
            f"ALFA {alpha}"
        )

    # Desativa PACC
    comandos.append("PACC")

    # Sai de OPER
    comandos.append("")

    # Sai do XFOIL
    comandos.append("QUIT")

    return (
        "\n".join(comandos)
        +
        "\n"
    )


# ============================================================
# LEITURA DA POLAR
# ============================================================

def ler_polar(
    arquivo_polar,
    perfil,
    reynolds
):

    resultados = []

    if not arquivo_polar.exists():

        return resultados

    with open(
        arquivo_polar,
        "r",
        encoding="latin-1",
        errors="ignore"
    ) as arquivo:

        linhas = arquivo.readlines()

    for linha in linhas:

        partes = linha.split()

        if len(partes) < 7:
            continue

        try:

            alpha = float(
                partes[0]
            )

            cl = float(
                partes[1]
            )

            cd = float(
                partes[2]
            )

            cdp = float(
                partes[3]
            )

            cm = float(
                partes[4]
            )

            top_xtr = float(
                partes[5]
            )

            bot_xtr = float(
                partes[6]
            )

        except ValueError:

            continue

        resultados.append({

            "perfil": perfil,

            "Re": reynolds,

            "Mach": MACH,

            "alpha": alpha,

            "CL": cl,

            "CD": cd,

            "CDp": cdp,

            "CM": cm,

            "Top_Xtr": top_xtr,

            "Bot_Xtr": bot_xtr

        })

    return resultados


# ============================================================
# FUNÇÃO EXECUTADA EM CADA PROCESSO
# ============================================================

def processar_simulacao(
    perfil_path,
    reynolds,
    pasta_base_temp
):

    """
    Cada tarefa é:

        1 perfil
        +
        1 Reynolds

    Cada processo usa uma pasta temporária exclusiva.
    """

    perfil_original = Path(
        perfil_path
    )

    nome_perfil = limpar_nome(
        perfil_original.stem
    )

    # --------------------------------------------------------
    # Pasta exclusiva da tarefa
    # --------------------------------------------------------

    pasta_trabalho = (
        Path(pasta_base_temp)
        /
        f"{nome_perfil}_Re_{reynolds}"
    )

    if pasta_trabalho.exists():

        shutil.rmtree(
            pasta_trabalho,
            ignore_errors=True
        )

    pasta_trabalho.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Copia o .dat
    # --------------------------------------------------------

    perfil_temp = (
        pasta_trabalho
        /
        f"{nome_perfil}.dat"
    )

    shutil.copy2(
        perfil_original,
        perfil_temp
    )

    # --------------------------------------------------------
    # Arquivos de saída
    # --------------------------------------------------------

    arquivo_polar = (
        pasta_trabalho
        /
        f"{nome_perfil}_Re_{reynolds}.txt"
    )

    arquivo_log = (
        pasta_trabalho
        /
        f"{nome_perfil}_Re_{reynolds}_LOG.txt"
    )

    comandos = criar_comandos_xfoil(

        perfil_temp.name,

        reynolds,

        arquivo_polar.name

    )

    inicio = time.time()

    timeout_ocorreu = False
    returncode = None

    try:

        processo = subprocess.run(

            [str(XFOIL_EXE)],

            input=comandos,

            text=True,

            cwd=str(pasta_trabalho),

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            timeout=TIMEOUT,

            errors="ignore"

        )

        returncode = processo.returncode

        stdout = processo.stdout or ""
        stderr = processo.stderr or ""

    except subprocess.TimeoutExpired as e:

        timeout_ocorreu = True

        returncode = -1

        stdout = ""

        stderr = ""

        if e.stdout:

            if isinstance(
                e.stdout,
                bytes
            ):

                stdout = e.stdout.decode(
                    errors="ignore"
                )

            else:

                stdout = e.stdout

        if e.stderr:

            if isinstance(
                e.stderr,
                bytes
            ):

                stderr = e.stderr.decode(
                    errors="ignore"
                )

            else:

                stderr = e.stderr

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    with open(

        arquivo_log,

        "w",

        encoding="utf-8",

        errors="ignore"

    ) as log:

        log.write(
            "========== COMANDOS ENVIADOS ==========\n\n"
        )

        log.write(
            comandos
        )

        log.write(
            "\n\n========== SAÍDA DO XFOIL ==========\n\n"
        )

        log.write(
            stdout
        )

        if stderr:

            log.write(
                "\n\n========== STDERR ==========\n\n"
            )

            log.write(
                stderr
            )

    # --------------------------------------------------------
    # Lê polar
    # --------------------------------------------------------

    resultados = ler_polar(

        arquivo_polar,

        perfil_original.stem,

        reynolds

    )

    # --------------------------------------------------------
    # Copia polar/log para pasta final
    # --------------------------------------------------------

    polar_destino = None
    log_destino = None

    if arquivo_polar.exists():

        polar_destino = (
            PASTA_ARQUIVOS_XFOIL
            /
            arquivo_polar.name
        )

        shutil.copy2(
            arquivo_polar,
            polar_destino
        )

    if arquivo_log.exists():

        log_destino = (
            PASTA_ARQUIVOS_XFOIL
            /
            arquivo_log.name
        )

        shutil.copy2(
            arquivo_log,
            log_destino
        )

    duracao = (
        time.time()
        -
        inicio
    )

    # --------------------------------------------------------
    # Remove pasta temporária exclusiva
    # --------------------------------------------------------

    shutil.rmtree(
        pasta_trabalho,
        ignore_errors=True
    )

    # Retorna somente dados serializáveis
    return {

        "perfil":
            perfil_original.stem,

        "reynolds":
            reynolds,

        "resultados":
            resultados,

        "quantidade":
            len(resultados),

        "timeout":
            timeout_ocorreu,

        "returncode":
            returncode,

        "duracao":
            duracao

    }


# ============================================================
# CONVERGÊNCIA
# ============================================================

def verificar_convergencia(
    resultados
):

    esperados = gerar_lista_alpha()

    convergidos = [

        round(
            resultado["alpha"],
            6
        )

        for resultado in resultados
    ]

    falharam = []

    for alpha in esperados:

        encontrou = any(

            abs(
                alpha - a
            ) < 0.01

            for a in convergidos

        )

        if not encontrou:

            falharam.append(
                alpha
            )

    return (
        convergidos,
        falharam
    )


# ============================================================
# MAIN
# ============================================================

def main():

    inicio_exec = time.time()

    print("\n")
    print("=" * 70)
    print("AUTOMAÇÃO XFOIL PARALELA")
    print("=" * 70)

    print(
        f"\nProcessos simultâneos: "
        f"{MAX_WORKERS}"
    )

    # --------------------------------------------------------
    # Validações
    # --------------------------------------------------------

    if not XFOIL_EXE.exists():

        print(
            "\nERRO: executável não encontrado:"
        )

        print(
            XFOIL_EXE
        )

        return

    if not PASTA_PERFIS.exists():

        print(
            "\nERRO: pasta de perfis não encontrada:"
        )

        print(
            PASTA_PERFIS
        )

        return

    # --------------------------------------------------------
    # Pasta de resultados
    # --------------------------------------------------------

    PASTA_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    PASTA_ARQUIVOS_XFOIL.mkdir(
        parents=True,
        exist_ok=True
    )

    pasta_base_temp = (
        XFOIL_EXE.parent.parent
        /
        "_temp_xfoil"
    )

    if pasta_base_temp.exists():

        shutil.rmtree(
            pasta_base_temp,
            ignore_errors=True
        )

    pasta_base_temp.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Perfis
    # --------------------------------------------------------

    perfis = sorted(
        PASTA_PERFIS.glob(
            "*.dat"
        )
    )

    if not perfis:

        print(
            "\nNenhum arquivo .dat encontrado."
        )

        return

    print(
        f"\nPerfis encontrados: "
        f"{len(perfis)}"
    )

    print(
        f"Reynolds: "
        f"{REYNOLDS}"
    )

    print(
        f"Mach: "
        f"{MACH}"
    )

    print(
        f"Alpha: "
        f"{ALPHA_INICIAL}° "
        f"até "
        f"{ALPHA_FINAL}° "
        f"passo "
        f"{PASSO_ALPHA}°"
    )

    # ========================================================
    # CRIA LISTA DE TAREFAS
    # ========================================================

    tarefas = []

    for perfil in perfis:

        for reynolds in REYNOLDS:

            tarefas.append(
                (
                    str(perfil),
                    reynolds,
                    str(pasta_base_temp)
                )
            )

    total_simulacoes = len(
        tarefas
    )

    print(
        f"\nTotal de tarefas XFOIL: "
        f"{total_simulacoes}"
    )

    banco_dados = []

    concluidas = 0

    # ========================================================
    # EXECUÇÃO PARALELA
    # ========================================================

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futuros = {

            executor.submit(
                processar_simulacao,
                perfil,
                reynolds,
                pasta_temp
            ):
            (
                perfil,
                reynolds
            )

            for (
                perfil,
                reynolds,
                pasta_temp
            ) in tarefas
        }

        for futuro in as_completed(
            futuros
        ):

            concluidas += 1

            perfil_path, reynolds = (
                futuros[futuro]
            )

            nome = Path(
                perfil_path
            ).stem

            try:

                resultado = futuro.result()

            except Exception as e:

                print(
                    f"\n[{concluidas}/{total_simulacoes}] "
                    f"ERRO: {nome} "
                    f"Re={reynolds:,}"
                )

                print(
                    f"   {e}"
                )

                continue

            resultados = resultado[
                "resultados"
            ]

            (
                convergidos,
                falharam
            ) = verificar_convergencia(
                resultados
            )

            print(
                f"\n[{concluidas}/{total_simulacoes}] "
                f"{resultado['perfil']} "
                f"| Re={resultado['reynolds']:,} "
                f"| {resultado['quantidade']}/"
                f"{len(gerar_lista_alpha())} "
                f"| {resultado['duracao']:.1f}s"
            )

            if resultado[
                "timeout"
            ]:

                print(
                    "   ⚠ TIMEOUT"
                )

            if resultados:

                banco_dados.extend(
                    resultados
                )

                if falharam:

                    print(
                        "   Falharam:",
                        ", ".join(
                            str(a)
                            for a in falharam
                        )
                    )

            else:

                print(
                    "   ⚠ Nenhum ponto salvo."
                )

    # ========================================================
    # ORDENA O BANCO
    # ========================================================

    banco_dados.sort(
        key=lambda r: (
            r["perfil"],
            r["Re"],
            r["alpha"]
        )
    )

    # ========================================================
    # CSV FINAL
    # ========================================================

    csv_final = (
        PASTA_RESULTADOS
        /
        "AA_banco_dados_xfoil.csv"
    )

    campos = [

        "perfil",

        "Re",

        "Mach",

        "alpha",

        "CL",

        "CD",

        "CDp",

        "CM",

        "Top_Xtr",

        "Bot_Xtr"

    ]

    with open(

        csv_final,

        "w",

        newline="",

        encoding="utf-8"

    ) as arquivo:

        writer = csv.DictWriter(

            arquivo,

            fieldnames=campos

        )

        writer.writeheader()

        writer.writerows(
            banco_dados
        )

    # ========================================================
    # FINAL
    # ========================================================

    duracao = (
        time.time()
        -
        inicio_exec
    )

    horas = int(
        duracao // 3600
    )

    minutos = int(
        (duracao % 3600) // 60
    )

    segundos = (
        duracao % 60
    )

    print("\n")
    print("=" * 70)
    print("SIMULAÇÕES FINALIZADAS")
    print("=" * 70)

    print(
        f"\nPontos aerodinâmicos: "
        f"{len(banco_dados)}"
    )

    print(
        f"\nCSV final:\n"
        f"{csv_final}"
    )

    print(
        f"\nTempo total: "
        f"{horas}h "
        f"{minutos}m "
        f"{segundos:.2f}s"
    )

    shutil.rmtree(
        pasta_base_temp,
        ignore_errors=True
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    # Necessário no Windows com multiprocessing
    main()
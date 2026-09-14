#!/usr/bin/env python3
"""
Instala e dispara ESTA maquina a partir do HD externo. Windows ou Linux.

    Windows:  python E:\\Disaster-Damage\\scripts\\instalar.py
              (ou clique duas vezes em INSTALAR.bat, na raiz do HD)

    Linux:    python3 /media/USUARIO/XBD-LAB/Disaster-Damage/scripts/instalar.py

Nao precisa de privilegio de administrador. Tudo acontece dentro da pasta do
usuario: o venv, o torch, os dados e os checkpoints.

Existe em Python, e nao em .sh mais .ps1, para haver UMA implementacao. Duas
versoes do mesmo instalador divergem na primeira correcao aplicada so de um
lado — e o erro aparece no laboratorio, na metade das maquinas.

Pode ser interrompido e rodado de novo: cada etapa detecta o que ja esta pronto.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

WINDOWS = os.name == "nt"
CENAS_ESPERADAS = 11034
GB_NECESSARIOS = 45

AQUI = Path(__file__).resolve().parent
HD = AQUI.parent.parent                       # .../HD/Disaster-Damage/scripts -> .../HD
DESTINO = Path(os.environ.get("DESTINO") or (Path.home() / "Disaster-Damage"))

EXCLUIR = {".git", ".venv", "venv", "geotiffs", "cache", "__pycache__",
           ".ipynb_checkpoints", "checkpoints", "results", "data"}


# --------------------------------------------------------------------- utilidades


def falhar(msg: str) -> "NoReturn":                     # type: ignore[valid-type]
    print(f"\n!! {msg}\n")
    if WINDOWS:
        input("Pressione Enter para fechar...")
    raise SystemExit(1)


def etapa(n: int, texto: str) -> None:
    print(f"\n[{n}/6] {texto}", flush=True)


def py_do_venv(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if WINDOWS else "bin/python")


def rodar(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def gb_livres(caminho: Path) -> float:
    return shutil.disk_usage(caminho).free / 1e9


def ip_local() -> str:
    """IP da interface que alcanca a rede. `hostname -I` nao existe no Windows,
    e gethostbyname devolve 127.0.1.1 em muitas distribuicoes."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))     # nao envia pacote; so escolhe a rota
        return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


def versao_cuda() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        saida = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for parte in saida.split("CUDA Version:")[1:]:
        return parte.strip().split()[0]
    return None


def roda_do_torch(cuda: str | None) -> str:
    """Escolhe o indice do torch pela versao do DRIVER.

    Instalar cu124 onde o driver so suporta 12.1 resulta num
    `torch.cuda.is_available() == False` silencioso: o treino cai para CPU, roda
    dezenas de vezes mais devagar, e isso so aparece quando a primeira epoca
    fecha — horas depois.
    """
    if not cuda:
        return "cpu"
    try:
        maior, menor = (int(x) for x in cuda.split(".")[:2])
    except ValueError:
        return "cu121"
    if maior >= 13 or (maior == 12 and menor >= 4):
        return "cu124"
    if maior == 12:
        return "cu121"
    if maior == 11 and menor >= 8:
        return "cu118"
    return "cu121"


def copiar_codigo(origem: Path, destino: Path) -> int:
    n = 0
    for raiz, dirs, arquivos in os.walk(origem):
        dirs[:] = [d for d in dirs if d not in EXCLUIR]
        rel = Path(raiz).relative_to(origem)
        (destino / rel).mkdir(parents=True, exist_ok=True)
        for a in arquivos:
            shutil.copy2(Path(raiz) / a, destino / rel / a)
            n += 1
    return n


def extrair(tar: Path, destino: Path) -> None:
    """Usa o tar do sistema quando existir — e varias vezes mais rapido que o
    tarfile do Python em 22 mil membros. O Windows 10+ traz bsdtar em System32."""
    exe = shutil.which("tar")
    if exe:
        try:
            rodar([exe, "-xf", str(tar), "-C", str(destino)])
            return
        except subprocess.CalledProcessError:
            print("      tar do sistema falhou; usando o do Python (mais lento)")
    with tarfile.open(tar, "r") as tf:
        try:
            tf.extractall(destino, filter="data")        # Python >= 3.12
        except TypeError:
            tf.extractall(destino)


# ------------------------------------------------------------------------ etapas


def main() -> int:
    print("=" * 70)
    print(f"  {socket.gethostname()}  —  instalacao a partir do HD")
    print(f"  HD:      {HD}")
    print(f"  destino: {DESTINO}")
    print(f"  sistema: {'Windows' if WINDOWS else 'Linux'}  python {sys.version.split()[0]}")
    print("=" * 70)

    repo_hd, tar = HD / "Disaster-Damage", HD / "xbd-data.tar"
    if not repo_hd.is_dir():
        falhar(f"nao achei {repo_hd}. O HD montou em outra letra/caminho?")
    if not tar.is_file():
        falhar(f"nao achei {tar}. O HD nao foi preparado com scripts/preparar_hd.sh")

    # ---- credenciais ----
    env = {}
    arq_env = HD / "lab.env"
    if arq_env.is_file():
        for linha in arq_env.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if linha and not linha.startswith("#") and "=" in linha:
                k, v = linha.split("=", 1)
                env[k.strip()] = v.strip()
    url = os.environ.get("LAB_URL") or env.get("LAB_URL", "")
    token = os.environ.get("LAB_TOKEN") or env.get("LAB_TOKEN", "")
    if not url or not token or token == "COLE_O_TOKEN_AQUI":
        falhar(f"LAB_URL/LAB_TOKEN ausentes. Edite {arq_env} e cole o token.")

    # ---- 1. espaco ----
    etapa(1, "conferindo espaco em disco...")
    DESTINO.mkdir(parents=True, exist_ok=True)
    livre = gb_livres(DESTINO)
    print(f"      {livre:.0f} GB livres em {DESTINO.anchor or DESTINO}")
    if livre < GB_NECESSARIOS:
        falhar(f"precisa de ~{GB_NECESSARIOS} GB (33 de dados + venv + checkpoints); ha {livre:.0f} GB")

    # ---- 2. codigo ----
    etapa(2, "copiando o codigo...")
    n = copiar_codigo(repo_hd, DESTINO)
    print(f"      {n} arquivos")

    # ---- 3. dados ----
    etapa(3, "extraindo os dados (33 GB, 5-15 min)...")
    masks = DESTINO / "data" / "masks"
    ja = sum(1 for _ in masks.rglob("*.png")) if masks.is_dir() else 0
    if ja >= CENAS_ESPERADAS and (DESTINO / "data" / "index.parquet").is_file():
        print(f"      ja extraidos ({ja} cenas); pulando")
    else:
        t0 = time.time()
        extrair(tar, DESTINO)
        n = sum(1 for _ in masks.rglob("*.png"))
        if n < CENAS_ESPERADAS:
            falhar(f"extraiu {n} de {CENAS_ESPERADAS} cenas — HD com defeito ou tar incompleto")
        print(f"      {n} cenas em {(time.time()-t0)/60:.1f} min")

    # ---- 4. ambiente ----
    cuda = versao_cuda()
    roda = roda_do_torch(cuda)
    etapa(4, f"montando o ambiente (driver CUDA {cuda or 'nenhum'} -> torch {roda}; ~5 min)...")
    if roda == "cpu":
        print("      !! nvidia-smi nao respondeu. O treino cairia para CPU e levaria")
        print("         dias em vez de horas. Verifique o driver antes de seguir.")
        if input("      Continuar assim mesmo? [s/N] ").strip().lower() != "s":
            return 1

    venv = DESTINO / ".venv"
    pyv = py_do_venv(venv)
    if not pyv.is_file():
        rodar([sys.executable, "-m", "venv", str(venv)])
    try:
        rodar([str(pyv), "-m", "pip", "install", "-q", "--upgrade", "pip", "wheel"])
        indice = f"https://download.pytorch.org/whl/{roda}"
        rodar([str(pyv), "-m", "pip", "install", "-q", "torch", "torchvision",
               "--index-url", indice])
        rodar([str(pyv), "-m", "pip", "install", "-q", "-r",
               str(DESTINO / "requirements-treino.txt")])
    except subprocess.CalledProcessError as exc:
        falhar(f"instalacao falhou ({exc}). Sem internet nesta maquina?")

    conf = subprocess.run(
        [str(pyv), "-c",
         "import torch;print(torch.__version__, torch.cuda.is_available(),"
         "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
        capture_output=True, text=True)
    print(f"      torch: {conf.stdout.strip() or conf.stderr.strip()[:200]}")
    if "True" not in conf.stdout:
        print("      !! torch NAO enxerga a GPU. Confira o driver.")

    # ---- 5. conexao ----
    etapa(5, "testando a conexao com o painel...")
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/api/estado",
                                     headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=20) as r:
            import json
            d = json.load(r)
        pend = sum(1 for e in d.get("experimentos", []) if e["status"] == "pendente")
        print(f"      ok — {len(d.get('experimentos', []))} experimentos ({pend} pendentes)")
    except Exception as exc:
        print(f"      !! sem contato com o painel: {exc}")
        print("         O treino ainda funciona: as metricas ficam no spool local")
        print("         e sobem quando a rede voltar.")
        if input("      Continuar assim mesmo? [s/N] ").strip().lower() != "s":
            return 1

    # ---- 6. agente ----
    etapa(6, "disparando o agente...")
    saida = DESTINO / "agente.out"
    ambiente = {**os.environ, "LAB_URL": url, "LAB_TOKEN": token,
                "LAB_WORKER": socket.gethostname(), "PYTHONUNBUFFERED": "1"}
    flags = {}
    if WINDOWS:
        # DETACHED_PROCESS: sobrevive ao fechar a janela do terminal.
        flags["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        flags["start_new_session"] = True

    with open(saida, "ab") as fh:
        subprocess.Popen([str(pyv), str(DESTINO / "scripts" / "rodar_agente.py")],
                         cwd=str(DESTINO), env=ambiente, stdout=fh, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, **flags)
    time.sleep(10)
    vivo = saida.is_file() and saida.stat().st_size > 0

    # ---- resumo ----
    print()
    print("=" * 70)
    print("  PRONTO" if vivo else "  AGENTE NAO CONFIRMADO — veja agente.out")
    print()
    print(f"  hostname : {socket.gethostname()}")
    print(f"  IP       : {ip_local()}        <-- ANOTE")
    print(f"  usuario  : {os.environ.get('USERNAME') or os.environ.get('USER')}")
    print(f"  GPU      : {conf.stdout.strip().split(' ', 2)[-1] if 'True' in conf.stdout else 'NENHUMA'}")
    print()
    print(f"  acompanhar: {saida}")
    print(f"  painel:     {url}")
    print()
    print("  Pode desplugar o HD e ir para a proxima maquina.")
    print("=" * 70)
    if WINDOWS:
        input("\nPressione Enter para fechar...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

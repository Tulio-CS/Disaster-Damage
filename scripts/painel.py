#!/usr/bin/env python3
"""
Administra a fila do painel pela linha de comando. Windows ou Linux.

    python scripts/painel.py estado              # tabela dos slots
    python scripts/painel.py erros               # o que falhou, e por que
    python scripts/painel.py maquinas            # quem esta vivo
    python scripts/painel.py resetar --erro      # devolve os com erro para a fila
    python scripts/painel.py resetar ref abl-crop256
    python scripts/painel.py resetar --tudo      # zera a fila inteira

Le LAB_URL/LAB_TOKEN do ambiente ou do lab.env (raiz do HD, ou ao lado do repo).

`resetar` mexe SO na fila: o slot volta a 'pendente' e pode ser reivindicado de
novo. O historico do run que falhou fica preservado, e o checkpoint local da
maquina tambem — se a mesma maquina pegar o slot de volta, ela retoma da epoca
em que parou em vez de recomecar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

WINDOWS = os.name == "nt"


def credenciais() -> tuple[str, str]:
    url, token = os.environ.get("LAB_URL", ""), os.environ.get("LAB_TOKEN", "")
    if url and token:
        return url.rstrip("/"), token
    aqui = Path(__file__).resolve().parent
    for cand in (aqui.parent.parent / "lab.env", aqui.parent / "lab.env"):
        if cand.is_file():
            for linha in cand.read_text(encoding="utf-8").splitlines():
                linha = linha.strip()
                if linha and not linha.startswith("#") and "=" in linha:
                    k, v = linha.split("=", 1)
                    if k.strip() == "LAB_URL" and not url:
                        url = v.strip().rstrip("/")
                    elif k.strip() == "LAB_TOKEN" and not token:
                        token = v.strip()
            break
    if not url or not token:
        sys.exit("! LAB_URL/LAB_TOKEN ausentes. Defina no ambiente ou no lab.env.")
    return url, token


URL, TOKEN = "", ""


def api(caminho: str, metodo: str = "GET", corpo: dict | None = None):
    req = urllib.request.Request(
        f"{URL}{caminho}", method=metodo,
        data=json.dumps(corpo).encode() if corpo is not None else None,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        bruto = r.read()
        return json.loads(bruto) if bruto else {}


def estado() -> dict:
    return api("/api/estado")


def _run_de(st: dict) -> dict[str, dict]:
    """Ultimo run de cada experimento."""
    por_nome: dict[str, dict] = {}
    for r in st["runs"]:
        a = por_nome.get(r["nome"])
        if not a or (r.get("iniciado_em") or 0) > (a.get("iniciado_em") or 0):
            por_nome[r["nome"]] = r
    return por_nome


# ------------------------------------------------------------------- comandos


def cmd_estado(_args) -> int:
    st = estado()
    runs = _run_de(st)
    larg = max((len(e["nome"]) for e in st["experimentos"]), default=10) + 1

    print(f"{'slot':<5}{'experimento':<{larg}}{'grupo':<12}{'status':<12}"
          f"{'ep':<8}{'score':<9}{'maquina'}")
    print("-" * (46 + larg + 14))
    contagem: dict[str, int] = {}
    for e in sorted(st["experimentos"], key=lambda x: (x["prioridade"], x["slot"] or 0)):
        r = runs.get(e["nome"]) or {}
        # A fila manda. O run que falhou continua com status 'erro' em `runs`,
        # mas isso e historico: se o slot voltou para a fila, ele esta pendente.
        sit = e["status"]
        if sit == "pendente" and r.get("status") in ("erro", "perdido", "interrompido"):
            sit = "pendente*"      # * = tentativa anterior falhou
        contagem[sit] = contagem.get(sit, 0) + 1
        tot = (r.get("epochs_total") or e["config"].get("epochs") or "?")
        ep = f"{r.get('epocas', 0)}/{tot}" if r else "—"
        sc = r.get("melhor", {}).get("score_xview2")
        print(f"{e['slot'] or '-':<5}{e['nome']:<{larg}}{e['grupo']:<12}{sit:<12}"
              f"{ep:<8}{(f'{sc:.4f}' if sc is not None else '—'):<9}"
              f"{r.get('hostname') or ''}")

    print()
    print("  ".join(f"{n} {k}" for k, n in sorted(contagem.items())))
    if any(k.endswith("*") for k in contagem):
        print("\n* slot de volta na fila; a tentativa anterior falhou")
    return 0


def cmd_erros(_args) -> int:
    st = estado()
    ruins = [r for r in st["runs"] if r["status"] in ("erro", "perdido")]
    if not ruins:
        print("Nenhum run com erro.")
        return 0

    for r in sorted(ruins, key=lambda x: x["nome"]):
        print("=" * 72)
        print(f"{r['nome']}   [{r['status']}]   {r.get('hostname') or '?'}   "
              f"epocas: {r.get('epocas', 0)}")
        erro = (r.get("erro") or "").strip()
        if erro:
            linhas = [l for l in erro.splitlines() if l.strip()]
            for l in linhas[-14:]:
                print(f"  {l[:160]}")
        else:
            print("  (sem mensagem — provavel morte abrupta: OOM, driver, ou energia)")
    print("=" * 72)
    print(f"\n{len(ruins)} run(s) com problema.")
    print("Para devolver todos a fila:  python scripts/painel.py resetar --erro")
    return 0


def cmd_maquinas(_args) -> int:
    st = estado()
    if not st["workers"]:
        print("Nenhuma maquina registrada ainda.")
        return 0
    print(f"{'host':<20}{'GPU':<28}{'estado':<14}{'visto ha'}")
    print("-" * 74)
    for w in st["workers"]:
        idade = w["idade_s"]
        quando = f"{idade}s" if idade < 90 else f"{idade // 60}min"
        frio = " (SEM CONTATO)" if idade > st["limite_seg"] else ""
        print(f"{w['hostname'][:19]:<20}{(w.get('gpu') or '—')[:27]:<28}"
              f"{w['status']:<14}{quando}{frio}")
    return 0


def cmd_resetar(args) -> int:
    st = estado()
    runs = _run_de(st)

    if args.tudo:
        alvos = [e["nome"] for e in st["experimentos"]]
    elif args.erro:
        alvos = [e["nome"] for e in st["experimentos"] if e["status"] == "erro"]
    elif args.nomes:
        validos = {e["nome"] for e in st["experimentos"]}
        alvos = [n for n in args.nomes if n in validos]
        for n in set(args.nomes) - validos:
            print(f"! experimento desconhecido, ignorado: {n}")
    else:
        print("informe nomes, --erro ou --tudo")
        return 2

    if not alvos:
        print("Nada a resetar.")
        return 0

    print(f"Devolver {len(alvos)} slot(s) para a fila:")
    for n in alvos:
        r = runs.get(n, {})
        print(f"  - {n}  ({r.get('status', 'pendente')}, {r.get('epocas', 0)} epocas)")
    if not args.sim:
        if input("\nConfirma? [s/N] ").strip().lower() != "s":
            print("cancelado.")
            return 1

    ok = 0
    for n in alvos:
        try:
            api(f"/api/experimentos/{n}/liberar", "POST", {})
            ok += 1
        except Exception as exc:
            print(f"! {n}: {exc}")
    print(f"\n{ok}/{len(alvos)} devolvidos a fila.")
    print("As maquinas pegam os slots no proximo claim (ate ~1 min).")
    return 0


def main() -> int:
    global URL, TOKEN
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("estado").set_defaults(fn=cmd_estado)
    sub.add_parser("erros").set_defaults(fn=cmd_erros)
    sub.add_parser("maquinas").set_defaults(fn=cmd_maquinas)

    r = sub.add_parser("resetar")
    r.add_argument("nomes", nargs="*")
    r.add_argument("--erro", action="store_true", help="todos com status erro/perdido")
    r.add_argument("--tudo", action="store_true", help="a fila inteira")
    r.add_argument("--sim", action="store_true", help="nao pedir confirmacao")
    r.set_defaults(fn=cmd_resetar)

    args = ap.parse_args()
    URL, TOKEN = credenciais()
    try:
        return args.fn(args)
    except urllib.error.HTTPError as exc:
        sys.exit(f"! painel recusou ({exc.code}). Token correto?")
    except urllib.error.URLError as exc:
        sys.exit(f"! painel inacessivel: {exc.reason}")


if __name__ == "__main__":
    codigo = main()
    if WINDOWS:
        input("\nPressione Enter para fechar...")
    raise SystemExit(codigo)

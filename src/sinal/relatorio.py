"""Apresentação dos resultados: tabelas Rich no terminal e JSON em disco."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sinal.comparador import ResultadoCaso, ResultadoComparacao
from sinal.config import Estatistica
from sinal.estatistica import Veredito

CORES = {
    Veredito.MELHOROU: "green",
    Veredito.PIOROU: "red",
    Veredito.EQUIVALENTE: "cyan",
    Veredito.INCONCLUSIVO: "yellow",
}


def _porcentagem(confianca: float) -> str:
    return f"{confianca * 100:g}%"  # 0.95 -> "95%", 0.975 -> "97.5%"


def _ic(inferior: float, superior: float) -> str:
    return f"[{inferior:+.2f}, {superior:+.2f}]"


def _media(notas: list[int]) -> str:
    return f"{sum(notas) / len(notas):.2f}" if notas else "—"


# ---------------------------------------------------------------------------
# Frases em português simples
# ---------------------------------------------------------------------------


def frase_do_veredito(resultado: ResultadoComparacao, estatistica: Estatistica) -> str:
    """Explica o veredito global numa frase, sem jargão."""
    i = resultado.intervalo
    faixa = f"IC {_porcentagem(estatistica.confianca)}: {i.inferior:+.2f} a {i.superior:+.2f}"
    if resultado.veredito == Veredito.MELHOROU:
        return f"O B é melhor que o A: em média {i.diferenca:+.2f} ponto na escala 1-5 ({faixa})."
    if resultado.veredito == Veredito.PIOROU:
        return f"O B é pior que o A: em média {i.diferenca:+.2f} ponto na escala 1-5 ({faixa})."
    if resultado.veredito == Veredito.EQUIVALENTE:
        return (
            f"A diferença entre A e B é irrelevante: o intervalo inteiro ({faixa}) "
            f"fica dentro de ±{estatistica.limiar_irrelevante:g} ponto."
        )
    return (
        f"Ainda não dá pra afirmar nada: o intervalo ({faixa}) inclui o zero. "
        "Pra estreitar, adicione mais CASOS (mais rodadas no mesmo caso ajudam pouco)."
    )


def avisos(resultado: ResultadoComparacao, estatistica: Estatistica) -> list[str]:
    """Alertas que merecem atenção de quem lê o resultado."""
    lista: list[str] = []
    alfa = 1 - estatistica.confianca
    ic_decisivo = resultado.veredito in (Veredito.MELHOROU, Veredito.PIOROU)
    permutacao_decisiva = resultado.p_valor < alfa
    if ic_decisivo != permutacao_decisiva:
        lista.append(
            f"O intervalo de confiança e o teste de permutação discordam (p = {resultado.p_valor:.3f}). "
            "Trate o resultado com cautela."
        )
    fora = len(resultado.casos) - resultado.casos_no_global
    if fora:
        lista.append(
            f"{fora} caso(s) ficaram fora do veredito global por falta de notas válidas em A ou B. "
            "Veja os erros no banco."
        )
    return lista


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def tabela_casos(casos: list[ResultadoCaso], confianca_parada: float) -> Table:
    tabela = Table(
        title="Resultado por caso (indicativo)",
        caption="O veredito por caso é uma pista pra diagnóstico. A decisão vale pelo veredito global.",
        caption_justify="left",
    )
    tabela.add_column("Caso")
    tabela.add_column("Rodadas", justify="right")
    tabela.add_column("Média A", justify="right")
    tabela.add_column("Média B", justify="right")
    tabela.add_column("B − A", justify="right")
    tabela.add_column(f"IC {_porcentagem(confianca_parada)}", justify="center")
    tabela.add_column("Veredito")
    tabela.add_column("Erros", justify="right")

    for caso in casos:
        intervalo = caso.intervalo
        tabela.add_row(
            caso.caso_id,
            str(caso.rodadas),
            _media(caso.notas_a),
            _media(caso.notas_b),
            f"{intervalo.diferenca:+.2f}" if intervalo else "—",
            _ic(intervalo.inferior, intervalo.superior) if intervalo else "—",
            Text(caso.veredito.value, style=CORES[caso.veredito]),
            Text(str(caso.erros), style="red" if caso.erros else "dim"),
        )
    return tabela


def painel_global(resultado: ResultadoComparacao, estatistica: Estatistica) -> Panel:
    cor = CORES[resultado.veredito]
    linhas = [
        Text(resultado.veredito.value, style=f"bold {cor}"),
        Text(frase_do_veredito(resultado, estatistica)),
        Text(
            f"p-valor (permutação): {resultado.p_valor:.3f} · "
            f"casos no global: {resultado.casos_no_global} de {len(resultado.casos)}",
            style="dim",
        ),
        Text(
            f"chamadas: {resultado.chamadas_feitas} de {resultado.chamadas_maximas} possíveis "
            f"(economia de {resultado.economia_percentual:.0f}%) · "
            f"custo: US$ {resultado.custo_total_usd:.4f} · erros: {resultado.erros}",
            style="dim",
        ),
    ]
    linhas += [Text(f"⚠ {aviso}", style="yellow") for aviso in avisos(resultado, estatistica)]
    return Panel(Group(*linhas), title="Veredito global", border_style=cor)


def mostrar_resultado(resultado: ResultadoComparacao, estatistica: Estatistica, console: Console) -> None:
    console.print(tabela_casos(resultado.casos, estatistica.confianca_parada))
    console.print(painel_global(resultado, estatistica))


def mostrar_historico(comparacoes: list[dict[str, Any]], console: Console) -> None:
    if not comparacoes:
        console.print("Nenhuma comparação ainda. Rode `sinal compara prompts/v1.txt prompts/v2.txt`.")
        return

    tabela = Table(title="Histórico de comparações")
    tabela.add_column("ID", justify="right")
    tabela.add_column("Data (UTC)")
    tabela.add_column("A → B")
    tabela.add_column("Veredito")
    tabela.add_column("B − A", justify="right")
    tabela.add_column("IC", justify="center")
    tabela.add_column("p", justify="right")
    tabela.add_column("Execuções", justify="right")
    tabela.add_column("Erros", justify="right")
    tabela.add_column("Custo US$", justify="right")

    for c in comparacoes:
        if c["status"] == "concluida":
            veredito = Veredito(c["veredito"])
            texto_veredito = Text(veredito.value, style=CORES[veredito])
            diferenca, ic, p = f"{c['diferenca']:+.2f}", _ic(c["ic_inferior"], c["ic_superior"]), f"{c['p_valor']:.3f}"
        else:
            texto_veredito = Text(c["status"], style="red" if c["status"] == "falhou" else "dim")
            diferenca = ic = p = "—"
        nomes = Text(f"{Path(c['prompt_a']).name} → {Path(c['prompt_b']).name}")
        nomes.append(f"  #{c['hash_a'][:4]}→#{c['hash_b'][:4]}", style="dim")
        tabela.add_row(
            str(c["id"]),
            c["criado_em"].replace("T", " ")[:16],
            nomes,
            texto_veredito,
            diferenca,
            ic,
            p,
            str(c["total_execucoes"]),
            Text(str(c["total_erros"]), style="red" if c["total_erros"] else "dim"),
            f"{c['custo_total_usd']:.4f}",
        )
    console.print(tabela)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def salvar_json(resultado: ResultadoComparacao, pasta: str | Path) -> Path:
    """Salva o resultado completo em JSON (pra CI, dashboards ou outro programa ler)."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    agora = datetime.now()
    caminho = pasta / f"comparacao-{resultado.comparacao_id}-{agora:%Y%m%d-%H%M%S}.json"
    dados = {"gerado_em": agora.isoformat(timespec="seconds"), **resultado.para_dict()}
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho

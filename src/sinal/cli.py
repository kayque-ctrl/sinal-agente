"""Linha de comando do Sinal (Typer + Rich).

Códigos de saída:
  0 = rodou normal (qualquer veredito)
  1 = PIOROU com --falhar-se-piorar (pra quebrar o CI)
  2 = erro (config, arquivo inexistente, tudo falhou...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from sinal.armazenamento import Armazenamento
from sinal.casos import ErroCasos
from sinal.comparador import ErroComparacao, comparar
from sinal.config import NOME_ARQUIVO_PADRAO, Config, ErroConfig, carregar_config
from sinal.estatistica import Veredito
from sinal.exemplos import criar_projeto
from sinal.juiz import ErroJuiz
from sinal.relatorio import mostrar_historico, mostrar_resultado, salvar_json

SAIDA_PIOROU = 1
SAIDA_ERRO = 2

app = typer.Typer(
    help="Sinal: testes de regressão estatísticos para prompts (sinal vs. ruído).",
    no_args_is_help=True,
    add_completion=False,
)

OpcaoConfig = Annotated[
    Path,
    typer.Option("--config", "-c", help="Caminho do sinal.yaml.", show_default=True),
]


def _console() -> Console:
    # Criado a cada comando (e não no import) pra respeitar a largura do terminal da vez.
    return Console()


def _falhar(console: Console, mensagem: str) -> typer.Exit:
    console.print(f"[bold red]Erro:[/] {mensagem}")
    return typer.Exit(SAIDA_ERRO)


def _carregar_config(console: Console, caminho: Path) -> Config:
    try:
        return carregar_config(caminho)
    except ErroConfig as erro:
        raise _falhar(console, str(erro)) from erro


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------


@app.command()
def init(
    pasta: Annotated[Path, typer.Argument(help="Onde criar o projeto de exemplo.")] = Path("."),
) -> None:
    """Cria um projeto de exemplo (agente de vendas) que funciona offline, com modelos fake."""
    console = _console()
    criados, ja_existiam = criar_projeto(pasta)
    for caminho in criados:
        console.print(f"[green]criado[/]   {caminho.as_posix()}")
    for caminho in ja_existiam:
        console.print(f"[yellow]mantido[/]  {caminho.as_posix()} (já existia, não foi alterado)")
    console.print("\nPróximo passo: [bold]sinal compara prompts/v1.txt prompts/v2.txt[/]")


@app.command()
def compara(
    prompt_a: Annotated[Path, typer.Argument(help="Prompt atual (A).")],
    prompt_b: Annotated[Path, typer.Argument(help="Prompt novo (B).")],
    casos: Annotated[Optional[Path], typer.Option("--casos", help="Arquivo de casos (padrão: o do sinal.yaml).")] = None,
    falhar_se_piorar: Annotated[
        bool, typer.Option("--falhar-se-piorar", help="Sai com código 1 se o veredito for PIOROU (pra CI).")
    ] = False,
    config: OpcaoConfig = Path(NOME_ARQUIVO_PADRAO),
) -> None:
    """Compara o prompt B contra o A e diz se melhorou, piorou ou não mudou."""
    console = _console()
    cfg = _carregar_config(console, config)

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console, transient=True
    ) as progresso:
        tarefa = progresso.add_task("Rodando...", total=None)
        feitas = {"execucoes": 0, "erros": 0}

        def ao_progredir(execucao) -> None:
            feitas["execucoes"] += 1
            feitas["erros"] += execucao.erro is not None
            progresso.update(tarefa, description=f"execuções: {feitas['execucoes']} · erros: {feitas['erros']}")

        try:
            resultado = comparar(cfg, prompt_a, prompt_b, caminho_casos=casos, ao_progredir=ao_progredir)
        except (ErroCasos, ErroJuiz, ErroComparacao) as erro:
            raise _falhar(console, str(erro)) from erro

    mostrar_resultado(resultado, cfg.estatistica, console)
    caminho_json = salvar_json(resultado, cfg.caminhos.resultados)
    console.print(f"Resultado salvo em [bold]{caminho_json}[/]")

    if falhar_se_piorar and resultado.veredito == Veredito.PIOROU:
        console.print("[bold red]PIOROU[/] com --falhar-se-piorar: saindo com código 1.")
        raise typer.Exit(SAIDA_PIOROU)


@app.command()
def historico(
    limite: Annotated[int, typer.Option("--limite", "-n", help="Quantas comparações mostrar.")] = 20,
    config: OpcaoConfig = Path(NOME_ARQUIVO_PADRAO),
) -> None:
    """Lista as comparações anteriores, da mais recente pra mais antiga."""
    console = _console()
    cfg = _carregar_config(console, config)
    if not cfg.caminhos.banco.exists():
        mostrar_historico([], console)
        return
    with Armazenamento(cfg.caminhos.banco) as banco:
        mostrar_historico(banco.listar_comparacoes(limite), console)

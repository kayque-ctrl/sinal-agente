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

from sinal import calibracao
from sinal.armazenamento import Armazenamento
from sinal.calibracao import ErroCalibracao
from sinal.casos import ErroCasos
from sinal.comparador import ErroComparacao, comparar
from sinal.config import NOME_ARQUIVO_PADRAO, Config, ErroConfig, carregar_config
from sinal.estatistica import Veredito
from sinal.exemplos import criar_projeto
from sinal.juiz import ErroJuiz
from sinal.relatorio import mostrar_calibracao, mostrar_historico, mostrar_resultado, salvar_json

SAIDA_PIOROU = 1
SAIDA_ERRO = 2
RESPOSTAS_SIM = {"s", "sim", "y", "yes"}

app = typer.Typer(
    help="Sinal: testes de regressão estatísticos para prompts (sinal vs. ruído).",
    no_args_is_help=True,
    add_completion=False,
)

app_calibra = typer.Typer(help="Calibração do juiz contra as suas notas (kappa ponderado).", no_args_is_help=True)
app.add_typer(app_calibra, name="calibra")

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


# ---------------------------------------------------------------------------
# sinal calibra ...
# ---------------------------------------------------------------------------


def _abrir_banco_existente(console: Console, cfg: Config) -> Armazenamento:
    if not cfg.caminhos.banco.exists():
        raise _falhar(console, "Nenhuma comparação ainda. Rode `sinal compara` primeiro.")
    return Armazenamento(cfg.caminhos.banco)


@app.command()
def agente(config: OpcaoConfig = Path(NOME_ARQUIVO_PADRAO)) -> None:
    """Chat com um assistente que usa o Sinal pra te ajudar a iterar prompts."""
    from rich.markdown import Markdown

    from sinal.agente import Agente

    console = _console()
    cfg = _carregar_config(console, config)

    def confirmar(mensagem: str) -> bool:
        # Confirmação própria: o typer.confirm só aceita y/yes, e aqui a gente digita "s".
        try:
            resposta = console.input(f"[yellow]{mensagem}[/] [s/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return resposta in RESPOSTAS_SIM

    assistente = Agente(cfg, confirmar)
    console.print(f"[bold]Agente do Sinal[/] (modelo: {cfg.modelos.agente}). Digite [bold]sair[/] pra encerrar.\n")
    while True:
        try:
            pedido = console.input("[bold cyan]você>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pedido.lower() in {"sair", "exit", "quit"}:
            break
        if not pedido:
            continue
        resposta = assistente.responder(
            pedido, ao_usar_ferramenta=lambda c: console.print(f"[dim]  ↳ ferramenta: {c.nome}({c.argumentos_json})[/]")
        )
        console.print(Markdown(resposta))
        console.print()
    if assistente.custo_total_usd:
        console.print(f"[dim]Custo do agente nesta sessão: US$ {assistente.custo_total_usd:.4f}[/]")
    console.print("Até mais!")


@app_calibra.command("exportar")
def calibra_exportar(
    saida: Annotated[Optional[Path], typer.Option("--saida", "-o", help="Onde salvar o CSV.")] = None,
    comparacao: Annotated[
        Optional[int], typer.Option("--comparacao", help="ID da comparação (padrão: a última concluída).")
    ] = None,
    tamanho: Annotated[int, typer.Option("--tamanho", "-n", help="Quantas respostas exportar.")] = (
        calibracao.PARES_RECOMENDADOS
    ),
    config: OpcaoConfig = Path(NOME_ARQUIVO_PADRAO),
) -> None:
    """Exporta respostas às cegas (sem a nota do juiz) pra você dar nota no Excel."""
    console = _console()
    cfg = _carregar_config(console, config)
    with _abrir_banco_existente(console, cfg) as banco:
        try:
            pasta = cfg.caminhos.banco.parent / "calibracao"
            destino = saida or pasta / f"calibracao-{comparacao or 'ultima'}.csv"
            caminho, quantidade = calibracao.exportar(
                banco, destino, comparacao_id=comparacao, tamanho=tamanho, semente=cfg.estatistica.semente
            )
        except ErroCalibracao as erro:
            raise _falhar(console, str(erro)) from erro
    console.print(f"[green]{quantidade} respostas exportadas[/] para [bold]{caminho}[/]")
    console.print(
        "Abra no Excel, preencha a coluna [bold]nota_humana[/] (1 a 5) seguindo a rubrica, salve "
        f"e rode: [bold]sinal calibra avaliar {caminho}[/]"
    )


@app_calibra.command("avaliar")
def calibra_avaliar(
    arquivo: Annotated[Path, typer.Argument(help="CSV com a coluna nota_humana preenchida.")],
    config: OpcaoConfig = Path(NOME_ARQUIVO_PADRAO),
) -> None:
    """Compara as suas notas com as do juiz: kappa, matriz de confusão e maiores discordâncias."""
    console = _console()
    cfg = _carregar_config(console, config)
    with _abrir_banco_existente(console, cfg) as banco:
        try:
            resultado = calibracao.avaliar(banco, arquivo)
        except ErroCalibracao as erro:
            raise _falhar(console, str(erro)) from erro
    mostrar_calibracao(resultado, console)

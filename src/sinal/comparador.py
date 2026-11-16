"""Orquestra a comparação entre dois prompts.

casos → roda A e B em paralelo → juiz → grava no banco → veredito por caso
→ roda mais só nos casos inconclusivos (amostragem adaptativa) → veredito global.
"""

from __future__ import annotations

import hashlib
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from sinal.armazenamento import Armazenamento, Execucao
from sinal.casos import Caso, carregar_casos, carregar_prompt, montar_mensagens
from sinal.config import Config
from sinal.estatistica import (
    Intervalo,
    Veredito,
    bootstrap_caso,
    bootstrap_hierarquico,
    decidir_veredito,
    teste_permutacao_pareado,
)
from sinal.juiz import avaliar, carregar_rubrica
from sinal.llm import chamar

CHAMADAS_POR_EXECUCAO = 2  # 1 ao modelo alvo + 1 ao juiz
VERSOES = ("A", "B")


class ErroComparacao(Exception):
    """A comparação não conseguiu chegar a um resultado (ex.: tudo deu erro)."""


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------


@dataclass
class EstadoCaso:
    """O que já sabemos de um caso durante a comparação (muda a cada lote)."""

    caso: Caso
    notas_a: list[int] = field(default_factory=list)
    notas_b: list[int] = field(default_factory=list)
    rodadas: int = 0  # rodadas já executadas POR VERSÃO
    erros: int = 0


@dataclass(frozen=True)
class ResultadoCaso:
    caso_id: str
    notas_a: list[int]
    notas_b: list[int]
    rodadas: int
    erros: int
    intervalo: Intervalo | None  # None se faltou nota válida em A ou B
    veredito: Veredito


@dataclass(frozen=True)
class ResultadoComparacao:
    comparacao_id: int
    prompt_a: str
    prompt_b: str
    casos: list[ResultadoCaso]
    intervalo: Intervalo
    p_valor: float
    veredito: Veredito
    casos_no_global: int
    execucoes: int
    erros: int
    chamadas_feitas: int
    chamadas_maximas: int
    custo_total_usd: float

    @property
    def chamadas_economizadas(self) -> int:
        return self.chamadas_maximas - self.execucoes * CHAMADAS_POR_EXECUCAO

    @property
    def economia_percentual(self) -> float:
        return 100 * self.chamadas_economizadas / self.chamadas_maximas if self.chamadas_maximas else 0.0

    def para_dict(self) -> dict[str, Any]:
        """Versão serializável em JSON (enums viram texto)."""
        dados = asdict(self)
        dados["veredito"] = self.veredito.value
        for caso in dados["casos"]:
            caso["veredito"] = caso["veredito"].value
        dados["chamadas_economizadas"] = self.chamadas_economizadas
        dados["economia_percentual"] = round(self.economia_percentual, 1)
        return dados


# ---------------------------------------------------------------------------
# Comparação
# ---------------------------------------------------------------------------


def comparar(
    config: Config,
    caminho_a: str | Path,
    caminho_b: str | Path,
    *,
    caminho_casos: str | Path | None = None,
    banco: Armazenamento | None = None,
    ao_progredir: Callable[[Execucao], None] | None = None,
) -> ResultadoComparacao:
    """Compara o prompt B contra o A e devolve o veredito.

    Args:
        caminho_casos: se None, usa o da config.
        banco: se None, abre o banco da config (e fecha no fim).
        ao_progredir: chamada a cada execução concluída (o CLI usa pra barra de progresso).

    Raises:
        ErroComparacao: nenhum caso teve notas válidas nas duas versões.
    """
    caminho_casos = Path(caminho_casos or config.caminhos.casos).resolve()
    casos = carregar_casos(caminho_casos)
    config_salva = asdict(config)
    config_salva["caminhos"]["casos"] = caminho_casos  # o arquivo de casos realmente usado (pode vir de --casos)
    prompts = {"A": carregar_prompt(caminho_a), "B": carregar_prompt(caminho_b)}
    rubrica = carregar_rubrica(config.caminhos.rubrica)

    banco_proprio = banco is None
    banco = banco or Armazenamento(config.caminhos.banco)
    try:
        comparacao_id = banco.criar_comparacao(
            prompt_a=str(caminho_a),
            prompt_b=str(caminho_b),
            conteudo_a=prompts["A"],
            conteudo_b=prompts["B"],
            modelo_alvo=config.modelos.alvo,
            modelo_juiz=config.modelos.juiz,
            config=config_salva,
        )
        try:
            resultado = _executar_comparacao(
                config, casos, prompts, rubrica, banco, comparacao_id, str(caminho_a), str(caminho_b), ao_progredir
            )
        except Exception as erro:
            banco.marcar_falha(comparacao_id, f"{type(erro).__name__}: {erro}")
            raise
        banco.finalizar_comparacao(
            comparacao_id,
            veredito=resultado.veredito.value,
            diferenca=resultado.intervalo.diferenca,
            ic_inferior=resultado.intervalo.inferior,
            ic_superior=resultado.intervalo.superior,
            p_valor=resultado.p_valor,
            chamadas_feitas=resultado.chamadas_feitas,
            chamadas_economizadas=resultado.chamadas_economizadas,
            resultado=resultado.para_dict(),
        )
        return resultado
    finally:
        if banco_proprio:
            banco.fechar()


def _executar_comparacao(
    config: Config,
    casos: list[Caso],
    prompts: dict[str, str],
    rubrica: str,
    banco: Armazenamento,
    comparacao_id: int,
    nome_a: str,
    nome_b: str,
    ao_progredir: Callable[[Execucao], None] | None,
) -> ResultadoComparacao:
    amostragem = config.amostragem
    estados = {caso.id: EstadoCaso(caso) for caso in casos}
    contadores = {"execucoes": 0, "chamadas": 0, "custo": 0.0}

    # Lote inicial: todos os casos, rodadas 1..rodadas_iniciais
    a_rodar = {caso.id: range(1, amostragem.rodadas_iniciais + 1) for caso in casos}

    while a_rodar:
        _executar_lote(config, prompts, rubrica, banco, comparacao_id, estados, a_rodar, contadores, ao_progredir)

        # Amostragem adaptativa: só os INCONCLUSIVOS que ainda não bateram o teto ganham mais rodadas.
        a_rodar = {}
        for indice, estado in enumerate(estados.values()):
            _, veredito = _avaliar_caso(estado, config, indice)
            if veredito == Veredito.INCONCLUSIVO and estado.rodadas < amostragem.rodadas_maximas:
                ultima = min(estado.rodadas + amostragem.tamanho_lote, amostragem.rodadas_maximas)
                a_rodar[estado.caso.id] = range(estado.rodadas + 1, ultima + 1)

    return _resultado_final(config, estados, comparacao_id, nome_a, nome_b, contadores)


def _executar_lote(
    config: Config,
    prompts: dict[str, str],
    rubrica: str,
    banco: Armazenamento,
    comparacao_id: int,
    estados: dict[str, EstadoCaso],
    a_rodar: dict[str, range],
    contadores: dict[str, Any],
    ao_progredir: Callable[[Execucao], None] | None,
) -> None:
    """Roda em paralelo todas as tarefas (caso, versão, rodada) do lote e grava cada uma ao terminar."""
    tarefas = [
        (estados[caso_id].caso, versao, rodada)
        for caso_id, rodadas in a_rodar.items()
        for rodada in rodadas
        for versao in VERSOES
    ]
    with ThreadPoolExecutor(max_workers=config.concorrencia) as executor:
        futuros = [
            executor.submit(_executar_tarefa, caso, versao, rodada, prompts[versao], rubrica, config)
            for caso, versao, rodada in tarefas
        ]
        concluidas: list[Execucao] = []
        # as_completed entrega na ordem em que TERMINAM, não na ordem em que foram enviados.
        for futuro in as_completed(futuros):
            execucao, chamadas = futuro.result()
            banco.salvar_execucao(comparacao_id, execucao)  # grava já: se cair depois, isso não se perde
            concluidas.append(execucao)
            contadores["execucoes"] += 1
            contadores["chamadas"] += chamadas
            contadores["custo"] += execucao.custo_usd + execucao.custo_juiz_usd
            if ao_progredir:
                ao_progredir(execucao)

    # A ordem de término varia a cada execução (threads). Registrando por rodada,
    # as listas de notas ficam sempre iguais para a mesma semente.
    for execucao in sorted(concluidas, key=lambda e: e.rodada):
        _registrar(estados[execucao.caso_id], execucao)
    for caso_id, rodadas in a_rodar.items():
        estados[caso_id].rodadas = rodadas[-1]


def _executar_tarefa(
    caso: Caso, versao: str, rodada: int, prompt: str, rubrica: str, config: Config
) -> tuple[Execucao, int]:
    """Uma execução: vendedor responde, juiz avalia. Nunca levanta exceção.

    Devolve a Execucao e quantas chamadas de LLM foram de fato feitas (1 ou 2).
    """
    gerador = _gerador_da_tarefa(config.estatistica.semente, caso.id, versao, rodada)
    try:
        resposta = chamar(config.modelos.alvo, montar_mensagens(prompt, caso), config.temperatura, gerador=gerador)
        base = dict(
            caso_id=caso.id,
            versao=versao,
            rodada=rodada,
            tokens_entrada=resposta.tokens_entrada,
            tokens_saida=resposta.tokens_saida,
            latencia_ms=resposta.latencia_ms,
            custo_usd=resposta.custo_usd,
        )
        if not resposta.ok:
            return Execucao(**base, erro=f"alvo: {resposta.erro}"), 1

        avaliacao = avaliar(
            caso, resposta.texto or "", config.modelos.juiz, rubrica, temperatura=config.temperatura_juiz, gerador=gerador
        )
        return (
            Execucao(
                **base,
                resposta=resposta.texto,
                nota=avaliacao.nota,
                justificativa=avaliacao.justificativa,
                metodo_parse=avaliacao.metodo,
                erro=avaliacao.erro,
                tokens_juiz=avaliacao.chamada.tokens_entrada + avaliacao.chamada.tokens_saida,
                custo_juiz_usd=avaliacao.chamada.custo_usd,
            ),
            2,
        )
    except Exception as erro:  # bug inesperado: vira registro, não derruba a rodada
        return Execucao(caso.id, versao, rodada, erro=f"interno: {type(erro).__name__}: {erro}"), 0


def _gerador_da_tarefa(semente: int | None, caso_id: str, versao: str, rodada: int) -> random.Random | None:
    """Sorteador próprio de cada tarefa, derivado de (semente, caso, versão, rodada).

    Com threads, a ordem em que as tarefas terminam muda a cada execução. Com uma semente
    por tarefa, o resultado fica igual mesmo assim. Só afeta o modo fake.
    """
    if semente is None:
        return None
    chave = f"{semente}|{caso_id}|{versao}|{rodada}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(chave).digest()[:8], "big"))


def _registrar(estado: EstadoCaso, execucao: Execucao) -> None:
    if execucao.nota is None:
        estado.erros += 1
    elif execucao.versao == "A":
        estado.notas_a.append(execucao.nota)
    else:
        estado.notas_b.append(execucao.nota)


# ---------------------------------------------------------------------------
# Vereditos
# ---------------------------------------------------------------------------


def _avaliar_caso(estado: EstadoCaso, config: Config, indice: int) -> tuple[Intervalo | None, Veredito]:
    """IC e veredito de um caso. Sem notas válidas em A ou B, é INCONCLUSIVO.

    Usa `confianca_parada` (mais exigente que a global): esse veredito decide se o caso
    para ou roda mais, e é consultado a cada lote. Por isso é só INDICATIVO; a decisão
    vale pelo veredito global.
    """
    if not estado.notas_a or not estado.notas_b:
        return None, Veredito.INCONCLUSIVO
    e = config.estatistica
    intervalo = bootstrap_caso(
        estado.notas_a,
        estado.notas_b,
        confianca=e.confianca_parada,
        reamostragens=e.reamostragens,
        semente=None if e.semente is None else e.semente + indice,
    )
    return intervalo, veredito_do_caso(intervalo, estado, config)


def veredito_do_caso(intervalo: Intervalo, estado: EstadoCaso, config: Config) -> Veredito:
    """Veredito com a trava do n mínimo: conta notas VÁLIDAS (erros não contam)."""
    n_valido = min(len(estado.notas_a), len(estado.notas_b))
    return decidir_veredito(
        intervalo,
        config.estatistica.limiar_irrelevante,
        permitir_equivalencia=n_valido >= config.amostragem.minimo_para_equivalencia,
    )


def _resultado_final(
    config: Config,
    estados: dict[str, EstadoCaso],
    comparacao_id: int,
    nome_a: str,
    nome_b: str,
    contadores: dict[str, Any],
) -> ResultadoComparacao:
    resultados: list[ResultadoCaso] = []
    for indice, estado in enumerate(estados.values()):
        intervalo, veredito = _avaliar_caso(estado, config, indice)
        resultados.append(
            ResultadoCaso(
                caso_id=estado.caso.id,
                notas_a=list(estado.notas_a),
                notas_b=list(estado.notas_b),
                rodadas=estado.rodadas,
                erros=estado.erros,
                intervalo=intervalo,
                veredito=veredito,
            )
        )

    validos = [r for r in resultados if r.intervalo is not None]
    if not validos:
        raise ErroComparacao(
            "Nenhum caso teve notas válidas nas duas versões. Veja os erros com `sinal historico` "
            "ou no banco (tabela execucoes)."
        )

    e = config.estatistica
    intervalo_global = bootstrap_hierarquico(
        [(r.notas_a, r.notas_b) for r in validos],
        confianca=e.confianca,
        reamostragens=e.reamostragens,
        semente=e.semente,
    )
    p_valor = teste_permutacao_pareado(
        [r.intervalo.diferenca for r in validos], reamostragens=e.reamostragens, semente=e.semente
    )
    # Equivalência global só se nenhum caso ficou "pouco amostrado" sem conclusão firme.
    minimo = config.amostragem.minimo_para_equivalencia
    bem_amostrados = all(
        min(len(r.notas_a), len(r.notas_b)) >= minimo or r.veredito in (Veredito.MELHOROU, Veredito.PIOROU)
        for r in validos
    )
    veredito_global = decidir_veredito(
        intervalo_global, e.limiar_irrelevante, permitir_equivalencia=bem_amostrados
    )

    chamadas_maximas = len(estados) * len(VERSOES) * config.amostragem.rodadas_maximas * CHAMADAS_POR_EXECUCAO
    return ResultadoComparacao(
        comparacao_id=comparacao_id,
        prompt_a=nome_a,
        prompt_b=nome_b,
        casos=resultados,
        intervalo=intervalo_global,
        p_valor=p_valor,
        veredito=veredito_global,
        casos_no_global=len(validos),
        execucoes=contadores["execucoes"],
        erros=sum(r.erros for r in resultados),
        chamadas_feitas=contadores["chamadas"],
        chamadas_maximas=chamadas_maximas,
        custo_total_usd=contadores["custo"],
    )

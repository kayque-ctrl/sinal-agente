"""Calibração do juiz: compara as notas do juiz com notas humanas.

1. `exportar`: amostra de respostas já avaliadas vai pra um CSV, ÀS CEGAS
   (sem a nota do juiz, sem a justificativa, sem a versão A/B).
2. Você preenche a coluna `nota_humana` (no Excel, Google Sheets...).
3. `avaliar`: junta as suas notas com as do juiz e calcula o kappa ponderado quadrático.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sinal.armazenamento import Armazenamento
from sinal.estatistica import NOTAS_POSSIVEIS, interpretar_kappa, kappa_ponderado_quadratico, matriz_confusao
from sinal.llm import PADRAO_QUALIDADE

COLUNAS_CSV = ["id_execucao", "caso_id", "entrada", "criterio", "referencia", "resposta", "nota_humana"]
SEPARADOR = ";"  # o Excel em pt-BR espera ";" (a vírgula é o separador decimal)
CODIFICACAO = "utf-8-sig"  # UTF-8 com BOM: sem ele, o Excel embaralha os acentos
PARES_RECOMENDADOS = 30


class ErroCalibracao(Exception):
    """Problema na exportação ou na leitura do CSV de calibração."""


@dataclass(frozen=True)
class Discordancia:
    id_execucao: int
    caso_id: str
    nota_juiz: int
    nota_humana: int
    resposta: str
    justificativa_juiz: str | None

    @property
    def distancia(self) -> int:
        return abs(self.nota_juiz - self.nota_humana)


@dataclass(frozen=True)
class ResultadoCalibracao:
    pares: int
    ignoradas: int  # linhas sem nota humana (ainda não avaliadas)
    kappa: float
    interpretacao: str
    concordancia_exata: float  # % de pares com a mesma nota
    concordancia_vizinha: float  # % de pares com diferença de no máximo 1
    matriz: list[list[int]]  # linha = juiz, coluna = humano
    discordancias: list[Discordancia]  # as maiores primeiro


# ---------------------------------------------------------------------------
# Exportar
# ---------------------------------------------------------------------------


def exportar(
    banco: Armazenamento,
    caminho_csv: str | Path,
    *,
    comparacao_id: int | None = None,
    tamanho: int = PARES_RECOMENDADOS,
    semente: int | None = None,
) -> tuple[Path, int]:
    """Exporta uma amostra às cegas pra você dar nota.

    Args:
        comparacao_id: de qual comparação tirar a amostra (padrão: a última concluída).
        tamanho: quantas respostas exportar.

    Returns:
        (caminho do CSV, quantas linhas foram exportadas)
    """
    if tamanho < 1:
        raise ErroCalibracao("O tamanho da amostra deve ser pelo menos 1.")
    comparacao_id = comparacao_id if comparacao_id is not None else _ultima_concluida(banco)
    comparacao = banco.buscar_comparacao(comparacao_id)
    if comparacao is None:
        raise ErroCalibracao(f"A comparação {comparacao_id} não existe.")

    avaliadas = [e for e in banco.ler_execucoes(comparacao_id) if e["nota"] is not None]
    if not avaliadas:
        raise ErroCalibracao(f"A comparação {comparacao_id} não tem respostas avaliadas pelo juiz.")

    casos = _casos_da_comparacao(comparacao)
    escolhidas = amostra_estratificada(avaliadas, tamanho, random.Random(semente))

    caminho = Path(caminho_csv)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding=CODIFICACAO, newline="") as arquivo:
        escritor = csv.writer(arquivo, delimiter=SEPARADOR)
        escritor.writerow(COLUNAS_CSV)
        for execucao in escolhidas:
            caso = casos.get(execucao["caso_id"], {})
            escritor.writerow(
                [
                    execucao["id"],
                    execucao["caso_id"],
                    caso.get("entrada", ""),
                    caso.get("criterio", ""),
                    caso.get("referencia") or "",
                    _sem_marcador_fake(execucao["resposta"] or ""),
                    "",  # nota_humana: é você quem preenche
                ]
            )
            # De propósito NÃO vão pro CSV: nota, justificativa e versão (A/B).
    return caminho, len(escolhidas)


def amostra_estratificada(execucoes: list[dict[str, Any]], tamanho: int, gerador: random.Random) -> list[dict]:
    """Sorteia um pouco de cada nota do juiz, em rodízio, e embaralha no fim.

    Sem o rodízio, se o juiz dá 4 pra quase tudo, a amostra seria quase só de 4
    e o kappa ficaria instável. Sem o embaralhamento final, a ORDEM das linhas
    revelaria a nota do juiz (1ª = nota 1, 2ª = nota 2...).
    """
    grupos: dict[int, list[dict]] = defaultdict(list)
    for execucao in execucoes:
        grupos[execucao["nota"]].append(execucao)
    for grupo in grupos.values():
        gerador.shuffle(grupo)

    escolhidas: list[dict] = []
    while len(escolhidas) < tamanho and any(grupos.values()):
        for nota in sorted(grupos):
            if grupos[nota] and len(escolhidas) < tamanho:
                escolhidas.append(grupos[nota].pop())
    gerador.shuffle(escolhidas)
    return escolhidas


def _ultima_concluida(banco: Armazenamento) -> int:
    for comparacao in banco.listar_comparacoes(limite=1000):
        if comparacao["status"] == "concluida":
            return comparacao["id"]
    raise ErroCalibracao("Nenhuma comparação concluída ainda. Rode `sinal compara` primeiro.")


def _casos_da_comparacao(comparacao: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Entrada, critério e referência de cada caso, lidos do arquivo de casos da config salva."""
    from sinal.casos import ErroCasos, carregar_casos  # import local: só a exportação precisa

    caminho = ((comparacao.get("config") or {}).get("caminhos") or {}).get("casos")
    if not caminho:
        return {}
    try:
        return {c.id: {"entrada": c.entrada, "criterio": c.criterio, "referencia": c.referencia}
                for c in carregar_casos(caminho)}
    except ErroCasos:
        return {}  # arquivo mudou de lugar: exporta só com a resposta


def _sem_marcador_fake(resposta: str) -> str:
    """No modo fake, o marcador [qualidade=X] seria uma 'cola' pro humano. Tira."""
    return PADRAO_QUALIDADE.sub("", resposta).strip()


# ---------------------------------------------------------------------------
# Avaliar
# ---------------------------------------------------------------------------


def avaliar(banco: Armazenamento, caminho_csv: str | Path, *, maximo_discordancias: int = 5) -> ResultadoCalibracao:
    """Lê o CSV com as suas notas e compara com as notas do juiz guardadas no banco."""
    notas_humanas, ignoradas = ler_notas_humanas(caminho_csv)
    if not notas_humanas:
        raise ErroCalibracao(
            f"Nenhuma nota humana preenchida em '{caminho_csv}'. Preencha a coluna `nota_humana` com 1 a 5."
        )

    execucoes = banco.buscar_execucoes(list(notas_humanas))
    faltando = sorted(set(notas_humanas) - set(execucoes))
    if faltando:
        raise ErroCalibracao(
            f"Execuções não encontradas no banco: {faltando[:10]}. O CSV é deste projeto (deste sinal.db)?"
        )

    ids = list(notas_humanas)
    juiz = [execucoes[i]["nota"] for i in ids]
    humano = [notas_humanas[i] for i in ids]
    kappa = kappa_ponderado_quadratico(juiz, humano)

    discordancias = sorted(
        (
            Discordancia(
                id_execucao=i,
                caso_id=execucoes[i]["caso_id"],
                nota_juiz=execucoes[i]["nota"],
                nota_humana=notas_humanas[i],
                resposta=_sem_marcador_fake(execucoes[i]["resposta"] or ""),
                justificativa_juiz=execucoes[i]["justificativa"],
            )
            for i in ids
            if execucoes[i]["nota"] != notas_humanas[i]
        ),
        key=lambda d: (-d.distancia, d.id_execucao),
    )

    total = len(ids)
    return ResultadoCalibracao(
        pares=total,
        ignoradas=ignoradas,
        kappa=kappa,
        interpretacao=interpretar_kappa(kappa),
        concordancia_exata=100 * sum(j == h for j, h in zip(juiz, humano)) / total,
        concordancia_vizinha=100 * sum(abs(j - h) <= 1 for j, h in zip(juiz, humano)) / total,
        matriz=matriz_confusao(juiz, humano).tolist(),
        discordancias=discordancias[:maximo_discordancias],
    )


def ler_notas_humanas(caminho_csv: str | Path) -> tuple[dict[int, int], int]:
    """Lê o CSV (separador ; ou ,) e devolve ({id_execucao: nota_humana}, linhas ignoradas).

    Linhas com nota vazia são ignoradas (ainda não avaliadas). Nota inválida é erro,
    com o número da linha, pra você achar no Excel.
    """
    caminho = Path(caminho_csv)
    if not caminho.is_file():
        raise ErroCalibracao(f"Não achei o arquivo '{caminho}'.")
    texto = caminho.read_text(encoding=CODIFICACAO)
    primeira_linha = texto.splitlines()[0] if texto else ""
    separador = ";" if primeira_linha.count(";") >= primeira_linha.count(",") else ","

    leitor = csv.DictReader(texto.splitlines(), delimiter=separador)
    faltando = {"id_execucao", "nota_humana"} - set(leitor.fieldnames or [])
    if faltando:
        raise ErroCalibracao(f"O CSV não tem a(s) coluna(s): {', '.join(sorted(faltando))}.")

    notas: dict[int, int] = {}
    ignoradas = 0
    problemas: list[str] = []
    for numero_linha, linha in enumerate(leitor, start=2):  # linha 1 é o cabeçalho
        bruto = (linha.get("nota_humana") or "").strip()
        if not bruto:
            ignoradas += 1
            continue
        nota = _converter_nota(bruto)
        if nota is None:
            problemas.append(f"linha {numero_linha}: nota_humana {bruto!r} não é um inteiro de 1 a 5.")
            continue
        try:
            notas[int(linha["id_execucao"])] = nota
        except (TypeError, ValueError):
            problemas.append(f"linha {numero_linha}: id_execucao {linha.get('id_execucao')!r} inválido.")

    if problemas:
        raise ErroCalibracao("Problemas no CSV:\n  - " + "\n  - ".join(problemas))
    return notas, ignoradas


def _converter_nota(bruto: str) -> int | None:
    """Aceita "4", "4.0" e "4,0" (o Excel às vezes salva assim). Recusa "3.5", "6", "boa"."""
    try:
        valor = float(bruto.replace(",", "."))
    except ValueError:
        return None
    if not valor.is_integer() or int(valor) not in NOTAS_POSSIVEIS:
        return None
    return int(valor)

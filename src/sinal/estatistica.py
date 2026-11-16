"""Estatística do Sinal: só numpy, sem I/O.

Recebe notas e devolve números. Não sabe o que é LLM, YAML ou banco de dados,
por isso dá pra testar tudo com dados sintéticos em que a resposta certa é conhecida.

Ferramentas:
- bootstrap_caso: IC da diferença de médias (B - A) dentro de um caso.
- bootstrap_hierarquico: IC global, reamostrando casos e depois rodadas.
- teste_permutacao_pareado: p-valor bicaudal por troca de sinais.
- decidir_veredito: transforma um IC em MELHOROU / PIOROU / EQUIVALENTE / INCONCLUSIVO.
- kappa_ponderado_quadratico: concordância entre juiz e humano.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

Notas = Sequence[float]
NOTAS_POSSIVEIS = (1, 2, 3, 4, 5)


class Veredito(str, Enum):
    MELHOROU = "MELHOROU"
    PIOROU = "PIOROU"
    EQUIVALENTE = "EQUIVALENTE"
    INCONCLUSIVO = "INCONCLUSIVO"


@dataclass(frozen=True)
class Intervalo:
    """Diferença estimada (B - A) e seu intervalo de confiança."""

    diferenca: float
    inferior: float
    superior: float

    @property
    def largura(self) -> float:
        return self.superior - self.inferior


# ---------------------------------------------------------------------------
# [1] Bootstrap por caso
# ---------------------------------------------------------------------------


def bootstrap_caso(
    notas_a: Notas,
    notas_b: Notas,
    confianca: float = 0.95,
    reamostragens: int = 10_000,
    semente: int | None = None,
) -> Intervalo:
    """IC da diferença de médias (B - A) num único caso.

    Dentro de um caso, as rodadas de A e de B são independentes (a rodada 1 de A
    não tem relação com a rodada 1 de B), então cada lado é reamostrado separadamente.
    """
    a = _como_array(notas_a, "notas_a")
    b = _como_array(notas_b, "notas_b")
    _validar_parametros(confianca, reamostragens)
    gerador = np.random.default_rng(semente)

    medias_a = _medias_reamostradas(a, reamostragens, gerador)
    medias_b = _medias_reamostradas(b, reamostragens, gerador)
    inferior, superior = _limites_percentis(medias_b - medias_a, confianca)
    return Intervalo(float(b.mean() - a.mean()), inferior, superior)


def _medias_reamostradas(notas: np.ndarray, reamostragens: int, gerador: np.random.Generator) -> np.ndarray:
    """Faz `reamostragens` sorteios com reposição de uma vez e devolve a média de cada um."""
    # Matriz (reamostragens x n): cada linha é um sorteio de n posições, com reposição.
    indices = gerador.integers(0, notas.size, size=(reamostragens, notas.size))
    return notas[indices].mean(axis=1)


# ---------------------------------------------------------------------------
# [2] Bootstrap hierárquico (global)
# ---------------------------------------------------------------------------


def bootstrap_hierarquico(
    casos: Sequence[tuple[Notas, Notas]],
    confianca: float = 0.95,
    reamostragens: int = 10_000,
    semente: int | None = None,
    *,
    elementos_por_bloco: int = 2_000_000,
) -> Intervalo:
    """IC global da diferença (B - A), reamostrando em 2 níveis.

    Nível 1: sorteia QUAIS casos entram (com reposição). Captura a variação entre casos.
    Nível 2: dentro de cada caso sorteado, sorteia as rodadas. Captura o ruído do LLM.

    A diferença global é a média das diferenças por caso: cada caso vale 1 voto,
    não importa quantas rodadas ele teve (a amostragem adaptativa dá mais rodadas a alguns).

    Args:
        casos: lista de (notas_a, notas_b), uma tupla por caso. Os tamanhos podem variar.
        elementos_por_bloco: limite de memória; as reamostragens são feitas em blocos.
    """
    if len(casos) == 0:
        raise ValueError("É preciso pelo menos um caso.")
    _validar_parametros(confianca, reamostragens)

    matriz_a, n_a = _empilhar([a for a, _ in casos], "A")
    matriz_b, n_b = _empilhar([b for _, b in casos], "B")
    total_casos = len(casos)

    # Estimativa pontual: média (entre casos) da diferença de médias de cada caso.
    medias_reais_a = matriz_a.sum(axis=1) / n_a
    medias_reais_b = matriz_b.sum(axis=1) / n_b
    diferenca = float(np.mean(medias_reais_b - medias_reais_a))

    gerador = np.random.default_rng(semente)
    largura_maxima = max(matriz_a.shape[1], matriz_b.shape[1])
    tamanho_bloco = max(1, elementos_por_bloco // (total_casos * largura_maxima))

    diferencas = np.empty(reamostragens)
    for inicio in range(0, reamostragens, tamanho_bloco):
        quantas = min(tamanho_bloco, reamostragens - inicio)

        # NÍVEL 1: matriz (quantas x total_casos) com os índices dos casos sorteados.
        # O MESMO sorteio vale para A e B: o caso é a unidade pareada.
        escolhidos = gerador.integers(0, total_casos, size=(quantas, total_casos))

        # NÍVEL 2: reamostra as rodadas dentro de cada caso sorteado.
        medias_a = _medias_dentro_dos_casos(matriz_a, n_a, escolhidos, gerador)
        medias_b = _medias_dentro_dos_casos(matriz_b, n_b, escolhidos, gerador)

        diferencas[inicio : inicio + quantas] = (medias_b - medias_a).mean(axis=1)

    inferior, superior = _limites_percentis(diferencas, confianca)
    return Intervalo(diferenca, inferior, superior)


def _empilhar(listas: list[Notas], lado: str) -> tuple[np.ndarray, np.ndarray]:
    """Junta listas de tamanhos diferentes numa matriz retangular, completando com 0.

    Devolve a matriz (casos x maior_n) e o vetor com o n real de cada caso.
    Ex.: [[4, 5], [3, 3, 4]] -> [[4, 5, 0], [3, 3, 4]] e n = [2, 3]
    """
    arrays = [_como_array(notas, f"notas {lado} do caso #{i + 1}") for i, notas in enumerate(listas)]
    n = np.array([arr.size for arr in arrays])
    matriz = np.zeros((len(arrays), n.max()))
    for linha, arr in enumerate(arrays):
        matriz[linha, : arr.size] = arr
    return matriz, n


def _medias_dentro_dos_casos(
    matriz: np.ndarray,
    n: np.ndarray,
    escolhidos: np.ndarray,
    gerador: np.random.Generator,
) -> np.ndarray:
    """Para cada caso sorteado, reamostra suas rodadas com reposição e tira a média.

    Tudo vetorizado: nada de loop em Python por caso ou por reamostragem.
    Formatos: escolhidos (R, K) -> resultado (R, K).
    """
    largura = matriz.shape[1]
    n_escolhidos = n[escolhidos]  # (R, K): quantas rodadas cada caso sorteado tem

    # Sorteia um número em [0, 1) para cada posição e escala pelo n do caso:
    # floor(u * n) dá um índice uniforme entre 0 e n-1. Assim casos com n diferentes
    # convivem na mesma matriz.
    u = gerador.random((*escolhidos.shape, largura))  # (R, K, largura)
    colunas = (u * n_escolhidos[..., None]).astype(int)
    valores = matriz[escolhidos[..., None], colunas]  # (R, K, largura)

    # Um caso com n rodadas usa só as n primeiras posições sorteadas; o resto é descartado.
    validas = np.arange(largura) < n_escolhidos[..., None]
    return np.where(validas, valores, 0.0).sum(axis=-1) / n_escolhidos


# ---------------------------------------------------------------------------
# [3] Teste de permutação pareado
# ---------------------------------------------------------------------------


def teste_permutacao_pareado(
    diferencas_por_caso: Notas,
    reamostragens: int = 10_000,
    semente: int | None = None,
) -> float:
    """p-valor bicaudal por troca de sinais das diferenças por caso.

    Hipótese nula: A e B são iguais, então o sinal de cada diferença é cara ou coroa.
    Se 2^K combinações cabem em `reamostragens`, testa TODAS (p-valor exato);
    senão, sorteia `reamostragens` combinações (Monte Carlo).
    """
    d = _como_array(diferencas_por_caso, "diferencas_por_caso")
    if reamostragens < 1:
        raise ValueError("reamostragens deve ser >= 1.")
    total_casos = d.size
    observado = abs(d.mean())
    tolerancia = 1e-12  # evita que erro de arredondamento conte um empate como "menor"

    if 2**total_casos <= reamostragens:
        # Exato: cada número de 0 a 2^K - 1, em binário, é uma combinação de sinais.
        # Ex. K=3: 5 = 101 -> bits [1, 0, 1] -> sinais [-1, +1, -1]
        bits = (np.arange(2**total_casos)[:, None] >> np.arange(total_casos)) & 1
        sinais = 1 - 2 * bits  # bit 0 -> +1, bit 1 -> -1
        estatisticas = np.abs((sinais * d).mean(axis=1))
        return float(np.mean(estatisticas >= observado - tolerancia))

    gerador = np.random.default_rng(semente)
    sinais = gerador.choice([-1.0, 1.0], size=(reamostragens, total_casos))
    estatisticas = np.abs((sinais * d).mean(axis=1))
    extremos = np.count_nonzero(estatisticas >= observado - tolerancia)
    # O +1 conta a própria amostra observada: o p-valor nunca dá exatamente 0.
    return (extremos + 1) / (reamostragens + 1)


# ---------------------------------------------------------------------------
# [4] Veredito
# ---------------------------------------------------------------------------


def decidir_veredito(intervalo: Intervalo, limiar: float, *, permitir_equivalencia: bool = True) -> Veredito:
    """Transforma o IC num veredito.

    - IC inteiro dentro de [-limiar, +limiar] -> EQUIVALENTE (diferença irrelevante)
    - IC inteiro abaixo de 0                  -> PIOROU
    - IC inteiro acima de 0                   -> MELHOROU
    - IC cruza o 0                            -> INCONCLUSIVO

    Args:
        permitir_equivalencia: False quando há poucas amostras. Com 3 notas iguais
            o IC tem largura zero e "EQUIVALENTE" seria uma conclusão falsa.
    """
    if limiar < 0:
        raise ValueError("limiar deve ser >= 0.")
    if permitir_equivalencia and -limiar <= intervalo.inferior and intervalo.superior <= limiar:
        return Veredito.EQUIVALENTE
    if intervalo.superior < 0:
        return Veredito.PIOROU
    if intervalo.inferior > 0:
        return Veredito.MELHOROU
    return Veredito.INCONCLUSIVO


# ---------------------------------------------------------------------------
# [5] Kappa de Cohen ponderado quadrático
# ---------------------------------------------------------------------------


def matriz_confusao(notas_1: Notas, notas_2: Notas, categorias: Sequence[int] = NOTAS_POSSIVEIS) -> np.ndarray:
    """Contagem de pares: linha = nota do avaliador 1, coluna = nota do avaliador 2."""
    if len(notas_1) != len(notas_2):
        raise ValueError(f"As listas têm tamanhos diferentes ({len(notas_1)} e {len(notas_2)}).")
    if len(notas_1) == 0:
        raise ValueError("É preciso pelo menos um par de notas.")
    posicao = {categoria: i for i, categoria in enumerate(categorias)}
    invalidas = sorted({n for n in [*notas_1, *notas_2] if n not in posicao})
    if invalidas:
        raise ValueError(f"Notas fora das categorias {tuple(categorias)}: {invalidas}")

    linhas = np.array([posicao[n] for n in notas_1])
    colunas = np.array([posicao[n] for n in notas_2])
    matriz = np.zeros((len(categorias), len(categorias)), dtype=int)
    np.add.at(matriz, (linhas, colunas), 1)  # soma 1 em cada (linha, coluna), inclusive repetidos
    return matriz


def kappa_ponderado_quadratico(
    notas_1: Notas, notas_2: Notas, categorias: Sequence[int] = NOTAS_POSSIVEIS
) -> float:
    """Kappa de Cohen com pesos quadráticos.

    kappa = 1 - (discordância observada) / (discordância esperada por acaso)
    peso de discordar entre as categorias i e j = (i - j)² / (k - 1)²

    Devolve NaN quando não há variação nenhuma (ex.: os dois deram 4 para tudo),
    porque aí a discordância esperada é zero e a conta não tem sentido.
    """
    observada = matriz_confusao(notas_1, notas_2, categorias) / len(notas_1)  # proporções
    k = len(categorias)
    i, j = np.indices((k, k))  # i = índice da linha, j = índice da coluna, em cada célula
    pesos = (i - j) ** 2 / (k - 1) ** 2

    # Por acaso: se os avaliadores fossem independentes, P(i, j) = P(linha i) * P(coluna j).
    esperada = np.outer(observada.sum(axis=1), observada.sum(axis=0))

    discordancia_esperada = float((pesos * esperada).sum())
    if discordancia_esperada == 0:
        return float("nan")
    return 1 - float((pesos * observada).sum()) / discordancia_esperada


def interpretar_kappa(kappa: float) -> str:
    """Faixas de Landis & Koch (1977), as mais usadas na literatura."""
    if np.isnan(kappa):
        return "indefinido (sem variação nas notas)"
    if kappa < 0:
        return "pior que o acaso"
    for teto, descricao in ((0.20, "leve"), (0.40, "razoável"), (0.60, "moderada"), (0.80, "substancial")):
        if kappa <= teto:
            return f"concordância {descricao}"
    return "concordância quase perfeita"


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------


def _como_array(notas: Notas, nome: str) -> np.ndarray:
    arr = np.asarray(notas, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{nome} precisa ter pelo menos uma nota.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{nome} tem valores inválidos (NaN ou infinito).")
    return arr


def _validar_parametros(confianca: float, reamostragens: int) -> None:
    if not 0 < confianca < 1:
        raise ValueError("confianca deve estar entre 0 e 1.")
    if reamostragens < 1:
        raise ValueError("reamostragens deve ser >= 1.")


def _limites_percentis(distribuicao: np.ndarray, confianca: float) -> tuple[float, float]:
    """Corta as pontas: com 95%, pega os percentis 2.5 e 97.5."""
    alfa = 1 - confianca
    inferior, superior = np.quantile(distribuicao, [alfa / 2, 1 - alfa / 2])
    return float(inferior), float(superior)

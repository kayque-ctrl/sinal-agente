"""Testes da estatística com dados sintéticos onde a resposta certa é CONHECIDA."""

import math

import numpy as np
import pytest

from sinal.estatistica import (
    Intervalo,
    Veredito,
    bootstrap_caso,
    bootstrap_hierarquico,
    decidir_veredito,
    interpretar_kappa,
    kappa_ponderado_quadratico,
    matriz_confusao,
    teste_permutacao_pareado,
)

R = 4000  # menos reamostragens nos testes = mais rápido, ainda preciso o bastante
LIMIAR = 0.25


# ===========================================================================
# [1] Bootstrap por caso
# ===========================================================================


def test_caso_diferenca_clara_melhorou():
    intervalo = bootstrap_caso([2, 2, 3, 2, 2, 3], [5, 4, 5, 5, 4, 5], reamostragens=R, semente=1)

    assert intervalo.diferenca == pytest.approx(28 / 6 - 14 / 6)  # média B (4.67) - média A (2.33)
    assert decidir_veredito(intervalo, LIMIAR) == Veredito.MELHOROU


def test_caso_diferenca_clara_piorou():
    intervalo = bootstrap_caso([5, 4, 5, 5], [2, 1, 2, 2], reamostragens=R, semente=1)
    assert decidir_veredito(intervalo, LIMIAR) == Veredito.PIOROU


def test_caso_amostras_iguais_nao_melhorou():
    notas = [3, 4, 2, 5, 3, 4]
    intervalo = bootstrap_caso(notas, notas, reamostragens=R, semente=1)

    assert intervalo.diferenca == 0
    assert intervalo.inferior < 0 < intervalo.superior  # IC contém o zero
    assert decidir_veredito(intervalo, LIMIAR) != Veredito.MELHOROU


def test_tres_notas_iguais_dao_ic_de_largura_zero():
    """O perigo que justifica o n mínimo: com [4,4,4] vs [4,4,4], todo sorteio dá 0."""
    intervalo = bootstrap_caso([4, 4, 4], [4, 4, 4], reamostragens=R, semente=1)

    assert intervalo.largura == 0
    assert decidir_veredito(intervalo, LIMIAR) == Veredito.EQUIVALENTE  # conclusão FALSA...
    assert decidir_veredito(intervalo, LIMIAR, permitir_equivalencia=False) == Veredito.INCONCLUSIVO  # ...bloqueada


def test_mesma_semente_mesmo_resultado():
    a, b = [3, 4, 2, 5], [4, 4, 5, 3]
    assert bootstrap_caso(a, b, semente=7, reamostragens=R) == bootstrap_caso(a, b, semente=7, reamostragens=R)


def test_ic_cobre_a_diferenca_real_em_cerca_de_95_por_cento():
    """Se o IC é de 95%, em ~95% dos experimentos ele deve conter a diferença verdadeira."""
    gerador = np.random.default_rng(0)
    diferenca_real = 0.7
    acertos = 0
    experimentos = 200
    for i in range(experimentos):
        a = gerador.normal(3.0, 1.0, size=30)
        b = gerador.normal(3.0 + diferenca_real, 1.0, size=30)
        intervalo = bootstrap_caso(a, b, reamostragens=2000, semente=i)
        acertos += intervalo.inferior <= diferenca_real <= intervalo.superior
    assert 0.88 <= acertos / experimentos <= 0.99


@pytest.mark.parametrize("a, b", [([], [1]), ([1], []), ([1, float("nan")], [1])])
def test_caso_notas_invalidas(a, b):
    with pytest.raises(ValueError):
        bootstrap_caso(a, b)


@pytest.mark.parametrize("confianca, reamostragens", [(1.0, 100), (0.0, 100), (0.95, 0)])
def test_parametros_invalidos(confianca, reamostragens):
    with pytest.raises(ValueError):
        bootstrap_caso([1, 2], [2, 3], confianca=confianca, reamostragens=reamostragens)


# ===========================================================================
# [2] Bootstrap hierárquico
# ===========================================================================


def bootstrap_saco_unico(casos, reamostragens, semente):
    """O jeito ERRADO, só pra comparação: junta todas as notas num saco e ignora os casos."""
    a = np.concatenate([np.asarray(x, dtype=float) for x, _ in casos])
    b = np.concatenate([np.asarray(y, dtype=float) for _, y in casos])
    return bootstrap_caso(a, b, reamostragens=reamostragens, semente=semente)


def simular_casos_sem_efeito_medio(gerador, total_casos=8, rodadas=20):
    """Cada caso tem seu próprio efeito (uns melhoram, outros pioram), mas a MÉDIA real é 0.

    É o cenário típico: o v2 ajuda em umas situações e atrapalha em outras.
    A resposta certa para "o v2 é melhor em geral?" é: NÃO DÁ PRA AFIRMAR.
    """
    efeitos = gerador.normal(0.0, 1.0, size=total_casos)
    bases = gerador.uniform(2.0, 4.0, size=total_casos)
    return [
        (
            bases[c] + gerador.normal(0, 0.5, size=rodadas),
            bases[c] + efeitos[c] + gerador.normal(0, 0.5, size=rodadas),
        )
        for c in range(total_casos)
    ]


def test_saco_unico_da_confianca_falsa_e_hierarquico_nao():
    """A prova do porquê: quantas vezes cada método 'inventa' uma diferença que não existe?"""
    gerador = np.random.default_rng(42)
    experimentos = 200
    falsos_saco = falsos_hierarquico = 0
    for i in range(experimentos):
        casos = simular_casos_sem_efeito_medio(gerador)
        saco = bootstrap_saco_unico(casos, reamostragens=2000, semente=i)
        hier = bootstrap_hierarquico(casos, reamostragens=2000, semente=i)
        falsos_saco += not (saco.inferior <= 0 <= saco.superior)
        falsos_hierarquico += not (hier.inferior <= 0 <= hier.superior)

    taxa_saco = falsos_saco / experimentos
    taxa_hier = falsos_hierarquico / experimentos
    print(f"\nalarme falso: saco único = {taxa_saco:.0%} | hierárquico = {taxa_hier:.0%} (ideal: 5%)")
    assert taxa_saco > 0.35  # o saco único erra MUITO mais que os 5% prometidos
    assert taxa_hier < 0.15  # o hierárquico fica perto do prometido


def test_hierarquico_diferenca_clara_em_todos_os_casos_melhorou():
    gerador = np.random.default_rng(3)
    casos = [(gerador.normal(2.5, 0.7, 5), gerador.normal(3.8, 0.7, 5)) for _ in range(8)]
    intervalo = bootstrap_hierarquico(casos, reamostragens=R, semente=1)
    assert decidir_veredito(intervalo, LIMIAR) == Veredito.MELHOROU


def test_hierarquico_mesmas_notas_nao_melhorou():
    casos = [([3, 4, 2], [3, 4, 2]), ([5, 4, 5], [5, 4, 5]), ([1, 2, 2], [1, 2, 2])]
    intervalo = bootstrap_hierarquico(casos, reamostragens=R, semente=1)

    assert intervalo.diferenca == 0
    assert decidir_veredito(intervalo, LIMIAR) != Veredito.MELHOROU


def test_hierarquico_cada_caso_vale_um_voto():
    """Caso 1: +4 com 2 rodadas. Caso 2: 0 com 20 rodadas.
    Hierárquico: (4 + 0) / 2 = 2.0. Saco único: (2*4 + 20*0) / 22 ≈ 0.36."""
    casos = [([1, 1], [5, 5]), ([3] * 20, [3] * 20)]

    assert bootstrap_hierarquico(casos, reamostragens=R, semente=1).diferenca == pytest.approx(2.0)
    assert bootstrap_saco_unico(casos, reamostragens=R, semente=1).diferenca == pytest.approx(8 / 22)


def test_hierarquico_aceita_casos_com_tamanhos_diferentes():
    casos = [([3, 4], [4, 5, 4, 5, 4]), ([2, 2, 3, 2, 3, 2, 2], [3, 3, 4])]
    intervalo = bootstrap_hierarquico(casos, reamostragens=R, semente=1)
    assert intervalo.inferior <= intervalo.diferenca <= intervalo.superior


def test_hierarquico_em_blocos_pequenos_da_o_mesmo_tipo_de_resultado():
    """O processamento em blocos é só pra economizar memória: não pode mudar a conclusão."""
    gerador = np.random.default_rng(5)
    casos = [(gerador.normal(3, 1, 6), gerador.normal(3.5, 1, 6)) for _ in range(6)]
    grande = bootstrap_hierarquico(casos, reamostragens=R, semente=1)
    pequeno = bootstrap_hierarquico(casos, reamostragens=R, semente=1, elementos_por_bloco=50)

    assert pequeno.diferenca == grande.diferenca
    assert pequeno.inferior == pytest.approx(grande.inferior, abs=0.1)
    assert pequeno.superior == pytest.approx(grande.superior, abs=0.1)


def test_hierarquico_mesma_semente_mesmo_resultado():
    casos = [([3, 4, 2], [4, 4, 5]), ([2, 3], [3, 3])]
    assert bootstrap_hierarquico(casos, semente=9, reamostragens=R) == bootstrap_hierarquico(
        casos, semente=9, reamostragens=R
    )


def test_hierarquico_sem_casos_ou_caso_vazio():
    with pytest.raises(ValueError):
        bootstrap_hierarquico([])
    with pytest.raises(ValueError, match="caso #2"):
        bootstrap_hierarquico([([1], [2]), ([], [3])])


# ===========================================================================
# [3] Teste de permutação pareado
# ===========================================================================


def test_permutacao_todos_positivos_da_p_pequeno_e_exato():
    """10 casos, todos positivos: só 2 das 1024 combinações (todos + e todos -) são tão extremas."""
    p = teste_permutacao_pareado([1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.9, 1.0], reamostragens=R)
    assert p == pytest.approx(2 / 1024)


def test_permutacao_com_6_casos_o_menor_p_possivel_e_2_de_64():
    """Com poucos casos, nunca dá pra ter certeza muito forte, por melhor que sejam as notas."""
    p = teste_permutacao_pareado([3.0, 3.0, 3.0, 3.0, 3.0, 3.0], reamostragens=R)
    assert p == pytest.approx(2 / 64)


def test_permutacao_diferencas_simetricas_da_p_alto():
    p = teste_permutacao_pareado([1.0, -1.0, 0.5, -0.5, 0.2, -0.2], reamostragens=R)
    assert p == 1.0


def test_permutacao_tudo_zero_da_p_1():
    assert teste_permutacao_pareado([0.0, 0.0, 0.0]) == 1.0


def test_permutacao_monte_carlo_com_muitos_casos():
    """20 casos -> 2^20 combinações > reamostragens -> sorteio. O p nunca dá exatamente 0."""
    gerador = np.random.default_rng(0)
    p_claro = teste_permutacao_pareado(gerador.normal(1.0, 0.5, 20), reamostragens=2000, semente=1)
    p_nulo = teste_permutacao_pareado(gerador.normal(0.0, 0.5, 20), reamostragens=2000, semente=1)

    assert 0 < p_claro < 0.01
    assert p_nulo > 0.05


def test_permutacao_sem_diferenca_alarme_falso_perto_de_5_por_cento():
    gerador = np.random.default_rng(1)
    experimentos = 300
    rejeicoes = sum(
        teste_permutacao_pareado(gerador.normal(0, 1, 12), reamostragens=R) < 0.05 for _ in range(experimentos)
    )
    assert rejeicoes / experimentos < 0.10


def test_permutacao_entradas_invalidas():
    with pytest.raises(ValueError):
        teste_permutacao_pareado([])
    with pytest.raises(ValueError):
        teste_permutacao_pareado([1.0], reamostragens=0)


# ===========================================================================
# [4] Veredito
# ===========================================================================


@pytest.mark.parametrize(
    "inferior, superior, esperado",
    [
        (-0.10, 0.20, Veredito.EQUIVALENTE),  # inteiro dentro de [-0.25, 0.25]
        (0.05, 0.15, Veredito.EQUIVALENTE),  # acima de 0, mas irrelevante: equivalência vem primeiro
        (0.30, 1.20, Veredito.MELHOROU),
        (0.10, 0.60, Veredito.MELHOROU),  # acima de 0 e sai da faixa
        (-1.20, -0.30, Veredito.PIOROU),
        (-0.40, 0.50, Veredito.INCONCLUSIVO),  # cruza o 0
        (-0.25, 0.25, Veredito.EQUIVALENTE),  # exatamente nas bordas conta como dentro
    ],
)
def test_veredito(inferior, superior, esperado):
    intervalo = Intervalo(diferenca=(inferior + superior) / 2, inferior=inferior, superior=superior)
    assert decidir_veredito(intervalo, LIMIAR) == esperado


def test_veredito_sem_permitir_equivalencia():
    dentro_da_faixa = Intervalo(0.0, -0.1, 0.1)
    acima_de_zero = Intervalo(0.1, 0.05, 0.15)

    assert decidir_veredito(dentro_da_faixa, LIMIAR, permitir_equivalencia=False) == Veredito.INCONCLUSIVO
    assert decidir_veredito(acima_de_zero, LIMIAR, permitir_equivalencia=False) == Veredito.MELHOROU


def test_veredito_limiar_negativo():
    with pytest.raises(ValueError):
        decidir_veredito(Intervalo(0, 0, 0), -0.1)


# ===========================================================================
# [5] Kappa ponderado quadrático
# ===========================================================================


def test_kappa_concordancia_perfeita():
    notas = [1, 2, 3, 4, 5, 3, 4]
    assert kappa_ponderado_quadratico(notas, notas) == pytest.approx(1.0)


def test_kappa_independencia_da_zero():
    """Com 2 categorias e todos os pares igualmente frequentes, a concordância é a do acaso."""
    assert kappa_ponderado_quadratico([1, 1, 2, 2], [1, 2, 1, 2], categorias=(1, 2)) == pytest.approx(0.0)


def test_kappa_discordancia_total_da_menos_um():
    """Feito à mão: observada = [[0, .5], [.5, 0]], esperada = .25 em tudo, pesos = [[0,1],[1,0]].
    kappa = 1 - 1.0 / 0.5 = -1."""
    assert kappa_ponderado_quadratico([1, 2], [2, 1], categorias=(1, 2)) == pytest.approx(-1.0)


def test_kappa_errar_por_pouco_custa_menos_que_errar_por_muito():
    humano = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    juiz_perto = [2, 2, 3, 5, 5, 1, 3, 3, 4, 4]  # erra por 1 às vezes
    juiz_longe = [5, 2, 3, 1, 5, 1, 5, 3, 4, 1]  # erra por 3-4 às vezes

    k_perto = kappa_ponderado_quadratico(humano, juiz_perto)
    k_longe = kappa_ponderado_quadratico(humano, juiz_longe)
    assert k_perto > 0.8
    assert k_longe < k_perto


def test_kappa_sem_variacao_e_indefinido():
    assert math.isnan(kappa_ponderado_quadratico([4, 4, 4], [4, 4, 4]))


def test_kappa_aceita_notas_float_inteiras():
    """Notas lidas de CSV podem vir como 4.0."""
    assert kappa_ponderado_quadratico([1.0, 5.0, 3.0], [1, 5, 3]) == pytest.approx(1.0)


def test_matriz_confusao():
    matriz = matriz_confusao([1, 1, 5, 3], [1, 2, 5, 3])
    assert matriz[0, 0] == 1  # (1, 1)
    assert matriz[0, 1] == 1  # (1, 2)
    assert matriz[4, 4] == 1  # (5, 5)
    assert matriz.sum() == 4


@pytest.mark.parametrize("notas_1, notas_2", [([1, 2], [1]), ([], []), ([1, 6], [1, 2]), ([0], [1])])
def test_kappa_entradas_invalidas(notas_1, notas_2):
    with pytest.raises(ValueError):
        kappa_ponderado_quadratico(notas_1, notas_2)


@pytest.mark.parametrize(
    "kappa, trecho",
    [
        (float("nan"), "indefinido"),
        (-0.2, "pior que o acaso"),
        (0.1, "leve"),
        (0.3, "razoável"),
        (0.5, "moderada"),
        (0.7, "substancial"),
        (0.9, "quase perfeita"),
    ],
)
def test_interpretar_kappa(kappa, trecho):
    assert trecho in interpretar_kappa(kappa)

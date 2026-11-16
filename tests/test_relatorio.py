"""Testes do relatorio.py."""

import json

import pytest
from rich.console import Console

from sinal.comparador import ResultadoCaso, ResultadoComparacao
from sinal.config import Estatistica
from sinal.estatistica import Intervalo, Veredito
from sinal.relatorio import avisos, frase_do_veredito, mostrar_historico, mostrar_resultado, salvar_json

ESTATISTICA = Estatistica()


def resultado_exemplo(veredito=Veredito.MELHOROU, intervalo=Intervalo(1.36, 0.90, 1.89), p_valor=0.031, **extra):
    casos = extra.pop(
        "casos",
        [
            ResultadoCaso("objecao-preco", [3, 3, 3], [4, 5, 4], 3, 0, Intervalo(1.33, 1.0, 2.0), Veredito.MELHOROU),
            ResultadoCaso("duvida-parcelamento", [2, 3, 3], [], 12, 12, None, Veredito.INCONCLUSIVO),
        ],
    )
    dados = dict(
        comparacao_id=7,
        prompt_a="prompts/v1.txt",
        prompt_b="prompts/v2.txt",
        casos=casos,
        intervalo=intervalo,
        p_valor=p_valor,
        veredito=veredito,
        casos_no_global=1,
        execucoes=24,  # 24 execuções × 2 chamadas = 48 de 96 → economia de 50%
        erros=12,
        chamadas_feitas=48,
        chamadas_maximas=96,
        custo_total_usd=0.0123,
    )
    dados.update(extra)
    return ResultadoComparacao(**dados)


def texto_impresso(funcao, *args):
    console = Console(record=True, width=160, color_system=None)
    funcao(*args, console)
    return console.export_text()


# --- frases -----------------------------------------------------------------


@pytest.mark.parametrize(
    "veredito, intervalo, trecho",
    [
        (Veredito.MELHOROU, Intervalo(1.36, 0.9, 1.89), "O B é melhor que o A: em média +1.36"),
        (Veredito.PIOROU, Intervalo(-0.8, -1.2, -0.4), "O B é pior que o A: em média -0.80"),
        (Veredito.EQUIVALENTE, Intervalo(0.0, -0.1, 0.1), "irrelevante"),
        (Veredito.INCONCLUSIVO, Intervalo(0.2, -0.3, 0.7), "mais CASOS"),
    ],
)
def test_frase_do_veredito(veredito, intervalo, trecho):
    frase = frase_do_veredito(resultado_exemplo(veredito, intervalo), ESTATISTICA)
    assert trecho in frase
    assert "IC 95%" in frase


# --- avisos -----------------------------------------------------------------


def test_aviso_quando_ic_e_permutacao_discordam():
    melhorou_mas_p_alto = resultado_exemplo(Veredito.MELHOROU, p_valor=0.20, casos_no_global=2, casos=[])
    assert any("discordam" in a for a in avisos(melhorou_mas_p_alto, ESTATISTICA))


def test_sem_aviso_quando_concordam_e_todos_os_casos_entraram():
    resultado = resultado_exemplo(Veredito.MELHOROU, p_valor=0.01, casos_no_global=2)
    assert avisos(resultado, ESTATISTICA) == []


def test_aviso_quando_caso_fica_fora_do_global():
    assert any("fora do veredito global" in a for a in avisos(resultado_exemplo(p_valor=0.01), ESTATISTICA))


# --- terminal ---------------------------------------------------------------


def test_mostrar_resultado():
    texto = texto_impresso(mostrar_resultado, resultado_exemplo(), ESTATISTICA)

    assert "Resultado por caso (indicativo)" in texto
    assert "A decisão vale pelo veredito global" in texto  # opção C
    assert "IC 99%" in texto  # o IC por caso usa a confianca_parada
    assert "objecao-preco" in texto and "[+1.00, +2.00]" in texto
    assert "duvida-parcelamento" in texto and "—" in texto  # caso sem notas em B
    assert "Veredito global" in texto and "MELHOROU" in texto
    assert "economia de 50%" in texto
    assert "US$ 0.0123" in texto
    assert "fora do veredito global" in texto


def test_mostrar_historico_vazio():
    assert "Nenhuma comparação ainda" in texto_impresso(mostrar_historico, [])


def test_mostrar_historico():
    linha_base = dict(
        criado_em="2026-11-16T15:02:03+00:00",
        prompt_a="prompts/v1.txt",
        prompt_b="prompts/v2.txt",
        hash_a="abcd1234abcd",
        hash_b="ef567890ef56",
        total_execucoes=64,
        total_erros=0,
        custo_total_usd=0.0,
    )
    comparacoes = [
        dict(linha_base, id=2, status="concluida", veredito="MELHOROU", diferenca=1.36, ic_inferior=0.9,
             ic_superior=1.89, p_valor=0.031),
        dict(linha_base, id=1, status="falhou", veredito=None, diferenca=None, ic_inferior=None,
             ic_superior=None, p_valor=None, total_erros=12),
    ]
    texto = texto_impresso(mostrar_historico, comparacoes)

    assert "v1.txt → v2.txt" in texto and "#abcd→#ef56" in texto
    assert "MELHOROU" in texto and "+1.36" in texto and "0.031" in texto
    assert "falhou" in texto
    assert "2026-11-16 15:02" in texto


# --- JSON -------------------------------------------------------------------


def test_salvar_json(tmp_path):
    caminho = salvar_json(resultado_exemplo(), tmp_path / "resultados")

    assert caminho.parent == tmp_path / "resultados"
    assert caminho.name.startswith("comparacao-7-")
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    assert dados["veredito"] == "MELHOROU"
    assert dados["casos"][1]["intervalo"] is None
    assert dados["economia_percentual"] == 50.0
    assert "gerado_em" in dados

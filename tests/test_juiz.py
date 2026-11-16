"""Testes do juiz.py: principalmente o parser, que precisa aguentar LLM desobediente."""

import inspect
import random

import pytest

from sinal import juiz
from sinal.casos import Caso
from sinal.juiz import (
    RUBRICA_PADRAO,
    TEMPERATURA_JUIZ,
    ErroJuiz,
    avaliar,
    carregar_rubrica,
    interpretar_resposta_juiz,
    montar_mensagens_juiz,
)
from sinal.llm import RespostaLLM

CASO = Caso(
    id="pedido-desconto",
    entrada="Se eu fechar hoje, vocês me dão 30% de desconto?",
    criterio="Não promete desconto fora da política.",
    referencia="No máximo 10%, só no plano anual.",
)
CASO_SEM_REFERENCIA = Caso(id="pergunta-vaga", entrada="Me fala mais sobre vocês.", criterio="Faz pergunta de descoberta.")


# ===========================================================================
# Parser: caminho feliz (JSON)
# ===========================================================================


def test_json_limpo():
    r = interpretar_resposta_juiz('{"justificativa": "Tratou bem a objeção.", "nota": 4}')
    assert (r.nota, r.justificativa, r.metodo, r.erro) == (4, "Tratou bem a objeção.", "json", None)


def test_json_em_cerca_de_codigo():
    r = interpretar_resposta_juiz('```json\n{"justificativa": "ok", "nota": 5}\n```')
    assert (r.nota, r.metodo) == (5, "json")


def test_json_no_meio_de_texto():
    r = interpretar_resposta_juiz('Claro! Aqui está a avaliação:\n{"justificativa": "ok", "nota": 2}\nEspero ter ajudado.')
    assert (r.nota, r.metodo) == (2, "json_extraido")


@pytest.mark.parametrize("nota", ['"4"', "4.0", '" 4 "'])
def test_nota_como_texto_ou_float_inteiro(nota):
    assert interpretar_resposta_juiz(f'{{"justificativa": "ok", "nota": {nota}}}').nota == 4


def test_chaves_em_maiusculas():
    r = interpretar_resposta_juiz('{"Justificativa": "ok", "Nota": 3}')
    assert (r.nota, r.justificativa) == (3, "ok")


# ===========================================================================
# Parser: notas inválidas viram ERRO (nada de "corrigir")
# ===========================================================================


@pytest.mark.parametrize(
    "nota, trecho",
    [("7", "fora da escala"), ("0", "fora da escala"), ("3.5", "inteira"), ('"boa"', "inválida"), ("true", "inválida"),
     ("null", "inválida")],
)
def test_nota_invalida_no_json_e_erro(nota, trecho):
    r = interpretar_resposta_juiz(f'{{"justificativa": "ok", "nota": {nota}}}')
    assert r.nota is None and r.metodo is None
    assert trecho in r.erro


# ===========================================================================
# Parser: plano B (regex)
# ===========================================================================


def test_json_quebrado_cai_no_regex_e_recupera_justificativa():
    # vírgula sobrando no fim: json.loads recusa
    r = interpretar_resposta_juiz('{"justificativa": "Não inventou desconto.", "nota": 5,}')
    assert (r.nota, r.justificativa, r.metodo) == (5, "Não inventou desconto.", "regex")


def test_aspas_simples_cai_no_regex():
    r = interpretar_resposta_juiz("{'justificativa': 'ok', 'nota': 3}")
    assert (r.nota, r.metodo) == (3, "regex")


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("Justificativa: tratou a objeção.\nNota: 4/5", 4),
        ("A resposta foi boa. Nota: 4.", 4),
        ("nota = 2", 2),
        ("Nota final: 3", 3),
        ("Uma nota 5 seria exagero. Nota final: 3", 3),  # "nota 5" sem dois-pontos não conta
        ("Nota: 4.0", 4),
    ],
)
def test_texto_livre_com_regex(texto, esperado):
    r = interpretar_resposta_juiz(texto)
    assert (r.nota, r.metodo) == (esperado, "regex")


@pytest.mark.parametrize("texto", ["Nota: 3.5", "Nota: 45", "Nota: 9"])
def test_regex_recusa_nota_quebrada_ou_fora_da_escala(texto):
    r = interpretar_resposta_juiz(texto)
    assert r.nota is None and r.erro


def test_json_valido_sem_nota_e_sem_regex_e_erro():
    r = interpretar_resposta_juiz('{"justificativa": "esqueci a nota"}')
    assert r.nota is None and "não consegui" in r.erro


@pytest.mark.parametrize("texto", ["", "   ", "Não sei avaliar isso."])
def test_sem_nota_nenhuma_e_erro(texto):
    r = interpretar_resposta_juiz(texto)
    assert r.nota is None and r.metodo is None and r.erro


def test_json_que_nao_e_objeto_cai_no_regex_ou_erro():
    assert interpretar_resposta_juiz("[4]").erro


# ===========================================================================
# Rubrica e mensagens
# ===========================================================================


def test_rubrica_padrao_quando_nao_ha_arquivo():
    assert carregar_rubrica(None) == RUBRICA_PADRAO


def test_rubrica_customizada_de_arquivo(tmp_path):
    arquivo = tmp_path / "rubrica.txt"
    arquivo.write_text("1 - ruim\n5 - ótimo\n", encoding="utf-8")
    rubrica = carregar_rubrica(arquivo)

    assert rubrica == "1 - ruim\n5 - ótimo"
    assert "1 - ruim" in montar_mensagens_juiz(CASO, "resposta", rubrica)[0]["content"]


def test_rubrica_inexistente_ou_vazia(tmp_path):
    with pytest.raises(ErroJuiz, match="Não achei"):
        carregar_rubrica(tmp_path / "nada.txt")
    vazio = tmp_path / "vazia.txt"
    vazio.write_text("\n", encoding="utf-8")
    with pytest.raises(ErroJuiz, match="vazio"):
        carregar_rubrica(vazio)


def test_mensagens_do_juiz_tem_tudo_que_ele_precisa():
    system, user = montar_mensagens_juiz(CASO, "Posso oferecer 10% no anual.")

    assert system["role"] == "system" and user["role"] == "user"
    assert RUBRICA_PADRAO in system["content"]
    assert '"justificativa"' in system["content"]
    assert system["content"].index('"justificativa"') < system["content"].index('"nota"')  # raciocínio primeiro
    for trecho in (CASO.entrada, CASO.criterio, CASO.referencia, "Posso oferecer 10% no anual."):
        assert trecho in user["content"]
    assert "<resposta>\nPosso oferecer 10% no anual.\n</resposta>" in user["content"]


def test_sem_referencia_nao_aparece_a_tag():
    _, user = montar_mensagens_juiz(CASO_SEM_REFERENCIA, "Me conta o que você procura?")
    assert "<referencia>" not in user["content"]


def test_marcador_rubrica_nao_sobra_no_prompt():
    system, _ = montar_mensagens_juiz(CASO, "x")
    assert "{rubrica}" not in system["content"]


# ===========================================================================
# avaliar()
# ===========================================================================


def test_juiz_e_cego_por_construcao():
    """avaliar() não tem como receber a versão (A/B): o parâmetro nem existe."""
    parametros = set(inspect.signature(avaliar).parameters)
    assert not any("versao" in p or "prompt" in p for p in parametros)


def test_avaliar_com_juiz_fake():
    resposta = "Entendo! Posso oferecer até 10% no plano anual.\n\n[qualidade=4.90]"
    avaliacao = avaliar(CASO, resposta, "fake/juiz", gerador=random.Random(0))

    assert avaliacao.ok
    assert avaliacao.nota in (4, 5)
    assert avaliacao.metodo == "json"
    assert avaliacao.chamada.tokens_entrada > 0


def test_avaliar_falha_de_api_vira_erro():
    avaliacao = avaliar(CASO, "qualquer", "fake/erro")

    assert not avaliacao.ok
    assert avaliacao.nota is None
    assert avaliacao.erro.startswith("juiz: RuntimeError")


def test_avaliar_usa_temperatura_zero_e_registra_erro_de_parse(monkeypatch):
    recebido = {}

    def chamar_falso(modelo, mensagens, temperatura, gerador=None):
        recebido["temperatura"] = temperatura
        return RespostaLLM(texto="Desculpe, não posso avaliar.")

    monkeypatch.setattr(juiz, "chamar", chamar_falso)
    avaliacao = avaliar(CASO, "resposta", "gpt-4o-mini")

    assert recebido["temperatura"] == TEMPERATURA_JUIZ == 0.0
    assert not avaliacao.ok
    assert "não consegui extrair" in avaliacao.erro
    assert avaliacao.chamada.texto == "Desculpe, não posso avaliar."  # texto bruto guardado pra diagnóstico

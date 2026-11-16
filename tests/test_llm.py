"""Testes do llm.py (modo fake e integração com LiteLLM simulada)."""

import json
import random
from types import SimpleNamespace

import litellm
import pytest

from sinal.casos import carregar_casos, carregar_prompt, montar_mensagens
from sinal.llm import chamar, dificuldade_do_caso, is_fake, qualidade_oculta

MENSAGENS = [{"role": "system", "content": "Você é vendedor."}, {"role": "user", "content": "Achei caro."}]


def test_is_fake():
    assert is_fake("fake/vendedor")
    assert not is_fake("gpt-4o-mini")


# --- vendedor fake ----------------------------------------------------------


def test_vendedor_fake_responde_sem_custo():
    resposta = chamar("fake/vendedor", MENSAGENS, 0.7, gerador=random.Random(1))

    assert resposta.ok
    assert "[qualidade=" in resposta.texto
    assert resposta.custo_usd == 0.0
    assert resposta.tokens_entrada > 0 and resposta.tokens_saida > 0
    assert resposta.latencia_ms >= 0


def test_mesma_semente_mesma_resposta():
    a = chamar("fake/vendedor", MENSAGENS, 0.7, gerador=random.Random(42))
    b = chamar("fake/vendedor", MENSAGENS, 0.7, gerador=random.Random(42))
    assert a.texto == b.texto


def test_qualquer_fake_desconhecido_age_como_vendedor():
    resposta = chamar("fake/qualquer-coisa", MENSAGENS, gerador=random.Random(1))
    assert "[qualidade=" in resposta.texto


def test_dificuldade_e_estavel_e_varia_entre_casos():
    assert dificuldade_do_caso("Achei caro.") == dificuldade_do_caso("Achei caro.")
    valores = {dificuldade_do_caso(f"caso {i}") for i in range(20)}
    assert len(valores) == 20
    assert all(2.0 <= v <= 3.5 for v in valores)


def test_qualidade_fica_entre_1_e_5():
    gerador = random.Random(0)
    valores = [qualidade_oculta(MENSAGENS, 2.0, gerador) for _ in range(2000)]  # temperatura alta = muito ruído
    assert min(valores) >= 1.0 and max(valores) <= 5.0


def test_mesmo_caso_tem_mesma_dificuldade_nos_dois_modos():
    """System+user e {entrada} devem enxergar o mesmo 'trecho do caso'."""
    modo_system = [{"role": "system", "content": "x"}, {"role": "user", "content": "Achei caro."}]
    modo_marcador = [{"role": "user", "content": "x\n\nAchei caro."}]
    # Mesma semente nos dois lados: o ruído é igual, então só a dificuldade poderia diferir.
    gerador_a, gerador_b = random.Random(5), random.Random(5)
    assert qualidade_oculta(modo_system, 0.0, gerador_a) == qualidade_oculta(modo_marcador, 0.0, gerador_b)


def test_prompt_v2_gera_qualidade_media_maior_que_v1(pasta_dados):
    """A diferença simulada precisa ser REAL e detectável: é ela que o Sinal vai medir."""
    casos = carregar_casos(pasta_dados / "casos_vendas.yaml")
    v1 = carregar_prompt(pasta_dados / "prompt_v1.txt")
    v2 = carregar_prompt(pasta_dados / "prompt_v2.txt")
    gerador = random.Random(123)

    def media(prompt):
        valores = [
            qualidade_oculta(montar_mensagens(prompt, caso), 0.7, gerador) for caso in casos for _ in range(100)
        ]
        return sum(valores) / len(valores)

    assert media(v2) - media(v1) > 0.8  # v2 tem 5 regras → bônus esperado de ~1.25


# --- juiz fake --------------------------------------------------------------


def test_juiz_fake_le_o_marcador_e_devolve_json_com_justificativa_primeiro():
    mensagens = [{"role": "user", "content": "Avalie a resposta:\nTexto...\n[qualidade=4.80]"}]
    resposta = chamar("fake/juiz", mensagens, 0.0, gerador=random.Random(0))

    dados = json.loads(resposta.texto)
    assert list(dados) == ["justificativa", "nota"]  # raciocínio antes da nota
    assert dados["nota"] in (4, 5)


def test_juiz_fake_nota_sempre_entre_1_e_5():
    gerador = random.Random(0)
    for qualidade in ("1.00", "5.00"):
        mensagens = [{"role": "user", "content": f"[qualidade={qualidade}]"}]
        notas = {json.loads(chamar("fake/juiz", mensagens, gerador=gerador).texto)["nota"] for _ in range(200)}
        assert notas <= {1, 2, 3, 4, 5}


def test_juiz_fake_sem_marcador_da_nota_neutra():
    resposta = chamar("fake/juiz", [{"role": "user", "content": "sem marcador"}])
    assert json.loads(resposta.texto)["nota"] == 3


# --- falhas -----------------------------------------------------------------


def test_falha_vira_registro_e_nao_excecao():
    resposta = chamar("fake/erro", MENSAGENS)

    assert not resposta.ok
    assert resposta.texto is None
    assert "falha simulada" in resposta.erro


# --- provedor real (LiteLLM substituído por um dublê) ----------------------


def test_chamada_real_extrai_texto_tokens_e_custo(monkeypatch):
    recebido = {}

    def completion_falso(**kwargs):
        recebido.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Olá, cliente!"))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        )

    monkeypatch.setattr(litellm, "completion", completion_falso)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0021)

    resposta = chamar("gpt-4o-mini", MENSAGENS, 0.3)

    assert resposta.ok
    assert resposta.texto == "Olá, cliente!"
    assert (resposta.tokens_entrada, resposta.tokens_saida) == (12, 3)
    assert resposta.custo_usd == pytest.approx(0.0021)
    assert recebido["model"] == "gpt-4o-mini" and recebido["temperature"] == 0.3


def test_temperatura_none_nem_envia_o_parametro(monkeypatch):
    """Modelos como Claude Opus 5.5 / Sonnet 5.5 recusam `temperature`: com None, ele nem vai."""
    recebido = {}

    def completion_falso(**kwargs):
        recebido.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="oi"))], usage=None)

    monkeypatch.setattr(litellm, "completion", completion_falso)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0)

    assert chamar("anthropic/claude-opus-5-5", MENSAGENS, None).ok
    assert "temperature" not in recebido


def test_fake_com_temperatura_none_funciona():
    resposta = chamar("fake/vendedor", MENSAGENS, None, gerador=random.Random(1))
    assert resposta.ok and "[qualidade=" in resposta.texto


def test_modelo_sem_preco_nao_falha(monkeypatch):
    def custo_falha(completion_response):
        raise ValueError("modelo não está na tabela de preços")

    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="oi"))], usage=None),
    )
    monkeypatch.setattr(litellm, "completion_cost", custo_falha)

    resposta = chamar("modelo-local", MENSAGENS)
    assert resposta.ok and resposta.custo_usd == 0.0 and resposta.tokens_entrada == 0


def test_erro_de_api_vira_registro(monkeypatch):
    def completion_quebrado(**kwargs):
        raise ConnectionError("sem internet")

    monkeypatch.setattr(litellm, "completion", completion_quebrado)

    resposta = chamar("gpt-4o-mini", MENSAGENS)
    assert not resposta.ok
    assert resposta.erro == "ConnectionError: sem internet"

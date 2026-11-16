"""Testes do agente: ferramentas, travas de segurança e o laço de tool calling."""

import json
from types import SimpleNamespace

import litellm
import pytest

from sinal import agente as modulo_agente
from sinal.agente import (
    ESQUEMAS_FERRAMENTAS,
    PROMPT_SISTEMA,
    Agente,
    ErroFerramenta,
    FerramentasSinal,
    validar_nome_prompt,
)
from sinal.config import montar_config
from sinal.exemplos import criar_projeto
from sinal.llm import ChamadaFerramenta, RespostaComFerramentas, chamar_com_ferramentas


@pytest.fixture
def config(tmp_path):
    criar_projeto(tmp_path)
    return montar_config({"estatistica": {"reamostragens": 500, "semente": 1}}, pasta_base=tmp_path)


def ferramentas(config, confirmar=True):
    confirmacoes = []

    def responder(mensagem):
        confirmacoes.append(mensagem)
        return confirmar

    return FerramentasSinal(config, responder), confirmacoes


def executar(ferr, ferramenta, /, **argumentos):
    # "/" = só-posicional: senão o argumento `nome` de ler_prompt/salvar_prompt colidiria com o parâmetro.
    return json.loads(ferr.executar(ChamadaFerramenta("id1", ferramenta, json.dumps(argumentos))))


# ===========================================================================
# Validação de nomes
# ===========================================================================


@pytest.mark.parametrize("nome", ["v3.txt", "v3-objecoes.md", "prompt_final_2.txt"])
def test_nomes_validos(nome):
    assert validar_nome_prompt(nome) == nome


@pytest.mark.parametrize(
    "nome", ["../sinal.yaml", "../../x.txt", "sub/v3.txt", "sub\\v3.txt", "v3.py", "", ".txt", "a" * 70 + ".txt",
             "v3.txt\n", 42]
)
def test_nomes_invalidos(nome):
    with pytest.raises(ErroFerramenta):
        validar_nome_prompt(nome)


# ===========================================================================
# Ferramentas
# ===========================================================================


def test_listar_e_ler_prompts(config):
    ferr, _ = ferramentas(config)

    nomes = [p["nome"] for p in executar(ferr, "listar_prompts")["prompts"]]
    assert nomes == ["v1.txt", "v2.txt"]
    assert executar(ferr, "ler_prompt", nome="v1.txt")["conteudo"].startswith("Você é um vendedor")


def test_listar_prompts_sem_pasta(tmp_path):
    config = montar_config({}, pasta_base=tmp_path)
    ferr, _ = ferramentas(config)
    assert executar(ferr, "listar_prompts") == {"prompts": []}


def test_ler_prompt_inexistente_ou_fora_da_pasta(config):
    ferr, _ = ferramentas(config)
    assert "não existe" in executar(ferr, "ler_prompt", nome="v9.txt")["erro"]
    assert "inválido" in executar(ferr, "ler_prompt", nome="../sinal.yaml")["erro"]


def test_segunda_camada_contra_sair_da_pasta(config, monkeypatch):
    """Mesmo se a validação do nome falhasse, o caminho resolvido não pode sair de prompts/."""
    monkeypatch.setattr(modulo_agente, "validar_nome_prompt", lambda nome: nome)  # desliga a 1ª camada
    ferr, _ = ferramentas(config)
    assert "fora da pasta" in executar(ferr, "ler_prompt", nome="../sinal.yaml")["erro"]


def test_listar_casos(config):
    ferr, _ = ferramentas(config)
    casos = executar(ferr, "listar_casos")["casos"]
    assert len(casos) == 6 and casos[2]["tem_referencia"] is True


def test_salvar_prompt_cria_nova_versao(config):
    ferr, _ = ferramentas(config)
    resultado = executar(ferr, "salvar_prompt", nome="v3.txt", conteudo="Você é vendedor.\n- Regra nova.")

    assert resultado == {"salvo": "v3.txt", "linhas": 2}
    assert (config.caminhos.prompts / "v3.txt").read_text(encoding="utf-8") == "Você é vendedor.\n- Regra nova.\n"


def test_salvar_prompt_nunca_sobrescreve(config):
    ferr, _ = ferramentas(config)
    original = (config.caminhos.prompts / "v2.txt").read_text(encoding="utf-8")

    resultado = executar(ferr, "salvar_prompt", nome="v2.txt", conteudo="apaguei tudo")

    assert "não será sobrescrito" in resultado["erro"]
    assert (config.caminhos.prompts / "v2.txt").read_text(encoding="utf-8") == original


@pytest.mark.parametrize("nome, conteudo", [("../sinal.yaml", "x"), ("v3.txt", "   ")])
def test_salvar_prompt_recusa_nome_ruim_ou_vazio(config, nome, conteudo):
    ferr, _ = ferramentas(config)
    assert "erro" in executar(ferr, "salvar_prompt", nome=nome, conteudo=conteudo)
    assert not (config.caminhos.prompts.parent / "sinal.yaml").read_text(encoding="utf-8") == "x"


def test_comparar_pede_confirmacao_e_respeita_o_nao(config):
    ferr, confirmacoes = ferramentas(config, confirmar=False)

    resultado = executar(ferr, "comparar_prompts", prompt_a="v1.txt", prompt_b="v2.txt")

    assert resultado["cancelado"] is True
    assert len(confirmacoes) == 1
    assert "6 casos" in confirmacoes[0] and "custo zero" in confirmacoes[0]
    assert not config.caminhos.banco.exists()  # nada rodou


def test_comparar_com_confirmacao(config):
    ferr, _ = ferramentas(config, confirmar=True)
    resultado = executar(ferr, "comparar_prompts", prompt_a="v1.txt", prompt_b="v2.txt")

    assert resultado["veredito_global"] == "MELHOROU"
    assert len(resultado["intervalo_de_confianca"]) == 2
    assert {c["id"] for c in resultado["casos"]} >= {"objecao-preco", "pergunta-vaga"}


def test_comparar_prompt_inexistente_nem_pergunta(config):
    ferr, confirmacoes = ferramentas(config)
    assert "não existe" in executar(ferr, "comparar_prompts", prompt_a="v1.txt", prompt_b="v9.txt")["erro"]
    assert confirmacoes == []  # não faz sentido pedir confirmação de algo que nem vai rodar


def test_mensagem_de_custo_com_modelo_real(tmp_path):
    criar_projeto(tmp_path)
    config = montar_config({"modelos": {"alvo": "gpt-4o-mini"}}, pasta_base=tmp_path)
    ferr, confirmacoes = ferramentas(config, confirmar=False)
    executar(ferr, "comparar_prompts", prompt_a="v1.txt", prompt_b="v2.txt")
    assert "custo zero" not in confirmacoes[0]
    assert "entre 72 e 288 chamadas" in confirmacoes[0]  # 6 casos × 2 versões × 2 chamadas × 3..12 rodadas


def test_ler_execucoes_e_historico(config):
    ferr, _ = ferramentas(config)
    comparacao_id = executar(ferr, "comparar_prompts", prompt_a="v1.txt", prompt_b="v2.txt")["comparacao_id"]

    execucoes = executar(ferr, "ler_execucoes", comparacao_id=comparacao_id, caso_id="objecao-preco", versao="B")
    assert execucoes["execucoes"] and all(e["versao"] == "B" for e in execucoes["execucoes"])
    assert all(e["justificativa_do_juiz"] for e in execucoes["execucoes"])

    assert len(executar(ferr, "ler_execucoes", comparacao_id=comparacao_id, limite=10_000)["execucoes"]) <= 50
    assert "não existe" in executar(ferr, "ler_execucoes", comparacao_id=999)["erro"]

    historico = executar(ferr, "historico")["comparacoes"]
    assert historico[0]["veredito"] == "MELHOROU"


def test_historico_sem_banco(config):
    ferr, _ = ferramentas(config)
    assert executar(ferr, "historico") == {"comparacoes": []}


def test_respostas_longas_sao_cortadas():
    assert modulo_agente._cortar("x" * 600).endswith("… [cortado]")
    assert modulo_agente._cortar("curto") == "curto"
    assert modulo_agente._cortar(None) is None


@pytest.mark.parametrize(
    "nome, argumentos_json, trecho",
    [
        ("apagar_tudo", "{}", "desconhecida"),
        ("ler_prompt", "{nao é json", "JSON"),
        ("ler_prompt", "[1, 2]", "objeto JSON"),
        ("ler_prompt", "{}", "Argumentos errados"),  # faltou "nome"
        ("listar_casos", '{"extra": 1}', "Argumentos errados"),  # sobrou argumento
    ],
)
def test_erros_de_chamada_voltam_pro_llm(config, nome, argumentos_json, trecho):
    ferr, _ = ferramentas(config)
    assert trecho in json.loads(ferr.executar(ChamadaFerramenta("id", nome, argumentos_json)))["erro"]


def test_erro_inesperado_de_ferramenta_vira_json(config):
    (config.caminhos.casos).write_text("isto: nao e uma lista", encoding="utf-8")
    ferr, _ = ferramentas(config)
    assert "ErroCasos" in executar(ferr, "listar_casos")["erro"]


def test_esquemas_cobrem_todas_as_ferramentas(config):
    ferr, _ = ferramentas(config)
    for esquema in ESQUEMAS_FERRAMENTAS:
        assert hasattr(ferr, f"_ferramenta_{esquema['function']['name']}")
    assert len(ESQUEMAS_FERRAMENTAS) == 7


def test_prompt_de_sistema_tem_as_regras():
    assert "Nunca afirme que um prompt melhorou sem um veredito global MELHOROU" in PROMPT_SISTEMA
    assert "linguagem simples" in PROMPT_SISTEMA
    assert "NOVA versão" in PROMPT_SISTEMA and "comparar_prompts" in PROMPT_SISTEMA


# ===========================================================================
# O laço do agente (com fake/agente)
# ===========================================================================


def test_agente_usa_ferramenta_e_responde(config):
    usadas = []
    assistente = Agente(config, lambda m: True)

    resposta = assistente.responder("liste os prompts", ao_usar_ferramenta=usadas.append)

    assert [c.nome for c in usadas] == ["listar_prompts"]
    assert "v1.txt" in resposta and "v2.txt" in resposta
    papeis = [m["role"] for m in assistente.mensagens]
    assert papeis == ["system", "user", "assistant", "tool", "assistant"]
    assert assistente.mensagens[2]["tool_calls"][0]["function"]["name"] == "listar_prompts"
    assert assistente.mensagens[3]["tool_call_id"] == assistente.mensagens[2]["tool_calls"][0]["id"]


def test_agente_fluxo_completo_salvar_e_comparar(config):
    assistente = Agente(config, lambda m: True)

    assistente.responder("salve v3.txt: Você é vendedor.\n- Pergunte a necessidade.\n- Seja objetivo.")
    resposta = assistente.responder("compare v1.txt com v3.txt")

    assert (config.caminhos.prompts / "v3.txt").is_file()
    assert '"veredito_global"' in resposta
    assert "só dá pra afirmar melhora quando o veredito global é MELHOROU" in resposta


@pytest.mark.parametrize(
    "pedido, ferramenta",
    [
        ("mostre v2.txt", "ler_prompt"),
        ("liste os casos", "listar_casos"),
        ("histórico", "historico"),
        ("leia as execuções da comparação 1", "ler_execucoes"),
        ("leia as execuções", "ler_execucoes"),
    ],
)
def test_roteador_do_agente_fake(config, pedido, ferramenta):
    usadas = []
    Agente(config, lambda m: True).responder(pedido, ao_usar_ferramenta=usadas.append)
    assert [c.nome for c in usadas] == [ferramenta]


def test_agente_fake_sem_ferramenta_explica_o_que_entende(config):
    assert "Entendo pedidos como" in Agente(config, lambda m: True).responder("oi, tudo bem?")


def test_agente_para_depois_do_maximo_de_passos(config, monkeypatch):
    sempre_ferramenta = RespostaComFerramentas(texto=None, chamadas=(ChamadaFerramenta("x", "listar_prompts", "{}"),))
    monkeypatch.setattr(modulo_agente, "chamar_com_ferramentas", lambda *a: sempre_ferramenta)

    resposta = Agente(config, lambda m: True, maximo_passos=3).responder("loop")
    assert "Parei depois de 3 passos" in resposta


def test_agente_erro_do_modelo_vira_mensagem(config, monkeypatch):
    monkeypatch.setattr(modulo_agente, "chamar_com_ferramentas", lambda *a: RespostaComFerramentas(None, erro="Timeout"))
    assert "Não consegui falar com o modelo" in Agente(config, lambda m: True).responder("oi")


# ===========================================================================
# chamar_com_ferramentas com provedor real (LiteLLM substituído por um dublê)
# ===========================================================================


def test_chamada_real_com_tool_calls(monkeypatch):
    recebido = {}

    def completion_falso(**kwargs):
        recebido.update(kwargs)
        chamada = SimpleNamespace(id="call_1", function=SimpleNamespace(name="listar_casos", arguments="{}"))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[chamada]))],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=5),
        )

    monkeypatch.setattr(litellm, "completion", completion_falso)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)

    resposta = chamar_com_ferramentas("gpt-4o-mini", [{"role": "user", "content": "oi"}], ESQUEMAS_FERRAMENTAS)

    assert resposta.ok and resposta.chamadas == (ChamadaFerramenta("call_1", "listar_casos", "{}"),)
    assert recebido["tool_choice"] == "auto" and "temperature" not in recebido
    assert resposta.mensagem_assistente() == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "listar_casos", "arguments": "{}"}}],
    }


def test_chamada_real_so_texto_e_sem_preco(monkeypatch):
    def custo_falha(completion_response):
        raise ValueError("sem preço")

    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Olá!", tool_calls=None))],
                                     usage=None),
    )
    monkeypatch.setattr(litellm, "completion_cost", custo_falha)

    resposta = chamar_com_ferramentas("modelo-local", [], [])
    assert resposta.texto == "Olá!" and resposta.chamadas == () and resposta.custo_usd == 0.0
    assert "tool_calls" not in resposta.mensagem_assistente()


def test_chamada_real_com_erro(monkeypatch):
    def quebrado(**kwargs):
        raise ConnectionError("sem internet")

    monkeypatch.setattr(litellm, "completion", quebrado)
    resposta = chamar_com_ferramentas("gpt-4o-mini", [], [])
    assert not resposta.ok and resposta.erro == "ConnectionError: sem internet"

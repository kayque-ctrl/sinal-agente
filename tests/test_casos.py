"""Testes do casos.py."""

import pytest

from sinal.casos import Caso, ErroCasos, carregar_casos, carregar_prompt, montar_casos, montar_mensagens

CASO = Caso(id="objecao-preco", entrada="Achei caro.", criterio="Trata a objeção.", referencia="Até 10%.")


# --- carregar casos ---------------------------------------------------------


def test_carrega_os_casos_de_vendas(pasta_dados):
    casos = carregar_casos(pasta_dados / "casos_vendas.yaml")

    assert len(casos) == 6
    assert casos[0].id == "objecao-preco"
    assert casos[0].referencia is None  # referência é opcional
    assert casos[2].referencia.startswith("A política permite")


def test_id_numerico_vira_texto():
    casos = montar_casos([{"id": 7, "entrada": "Oi", "criterio": "Cumprimenta"}])
    assert casos[0].id == "7"


def test_espacos_extras_sao_removidos():
    casos = montar_casos([{"id": " a ", "entrada": " Oi \n", "criterio": "Cumprimenta "}])
    assert casos[0] == Caso(id="a", entrada="Oi", criterio="Cumprimenta")


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(ErroCasos, match="Não achei"):
        carregar_casos(tmp_path / "nada.yaml")


def test_yaml_quebrado(tmp_path):
    arquivo = tmp_path / "casos.yaml"
    arquivo.write_text("- id: [sem fechar", encoding="utf-8")
    with pytest.raises(ErroCasos, match="YAML válido"):
        carregar_casos(arquivo)


@pytest.mark.parametrize("dados", [None, [], {"id": "a"}, "texto"])
def test_raiz_precisa_ser_lista_nao_vazia(dados):
    with pytest.raises(ErroCasos, match="lista"):
        montar_casos(dados)


def test_campo_obrigatorio_faltando_mostra_o_id():
    with pytest.raises(ErroCasos, match="caso 'sem-criterio'.*criterio"):
        montar_casos([{"id": "sem-criterio", "entrada": "Oi"}])


def test_sem_id_usa_a_posicao():
    with pytest.raises(ErroCasos, match="caso #2.*id"):
        montar_casos([{"id": "a", "entrada": "Oi", "criterio": "x"}, {"entrada": "Oi", "criterio": "x"}])


def test_item_que_nao_e_mapa():
    with pytest.raises(ErroCasos, match="caso #1"):
        montar_casos(["só um texto"])


def test_chave_desconhecida():
    with pytest.raises(ErroCasos, match="criterios"):
        montar_casos([{"id": "a", "entrada": "Oi", "criterios": "x"}])


def test_referencia_invalida():
    with pytest.raises(ErroCasos, match="referencia"):
        montar_casos([{"id": "a", "entrada": "Oi", "criterio": "x", "referencia": 10}])


def test_id_repetido():
    item = {"id": "a", "entrada": "Oi", "criterio": "x"}
    with pytest.raises(ErroCasos, match="id repetido"):
        montar_casos([item, dict(item)])


def test_todos_os_problemas_aparecem_juntos():
    with pytest.raises(ErroCasos) as erro:
        montar_casos([{"id": "a"}, {"id": "b", "entrada": "Oi", "criterio": "x", "extra": 1}])
    assert "caso 'a'" in str(erro.value)
    assert "caso 'b'" in str(erro.value)


# --- prompts ----------------------------------------------------------------


def test_carrega_prompt(pasta_dados):
    assert carregar_prompt(pasta_dados / "prompt_v1.txt").startswith("Você é um vendedor")


def test_prompt_inexistente(tmp_path):
    with pytest.raises(ErroCasos, match="Não achei"):
        carregar_prompt(tmp_path / "v9.txt")


def test_prompt_vazio(tmp_path):
    arquivo = tmp_path / "vazio.txt"
    arquivo.write_text("   \n", encoding="utf-8")
    with pytest.raises(ErroCasos, match="vazio"):
        carregar_prompt(arquivo)


# --- montar mensagens -------------------------------------------------------


def test_sem_marcador_vira_system_mais_user():
    mensagens = montar_mensagens("Você é vendedor.", CASO)
    assert mensagens == [
        {"role": "system", "content": "Você é vendedor."},
        {"role": "user", "content": "Achei caro."},
    ]


def test_com_marcador_vira_uma_mensagem_so():
    mensagens = montar_mensagens("Você é vendedor.\n\nCliente: {entrada}", CASO)
    assert mensagens == [{"role": "user", "content": "Você é vendedor.\n\nCliente: Achei caro."}]


def test_outras_chaves_no_prompt_nao_quebram():
    # Com .format() isso daria KeyError por causa do {"nota": 5}.
    prompt = 'Responda em JSON tipo {"nota": 5}.\nCliente: {entrada}'
    conteudo = montar_mensagens(prompt, CASO)[0]["content"]
    assert '{"nota": 5}' in conteudo
    assert "Cliente: Achei caro." in conteudo


@pytest.mark.parametrize("prompt", ["Você é vendedor.", "Cliente: {entrada}"])
def test_gabarito_nunca_vai_para_o_modelo_alvo(prompt):
    texto = " ".join(m["content"] for m in montar_mensagens(prompt, CASO))
    assert CASO.criterio not in texto
    assert CASO.referencia not in texto

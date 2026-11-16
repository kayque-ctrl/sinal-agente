"""Testes do config.py."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from sinal.config import Config, ErroConfig, carregar_config, montar_config


def escrever_yaml(pasta: Path, conteudo: str) -> Path:
    caminho = pasta / "sinal.yaml"
    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


# --- padrões ----------------------------------------------------------------


def test_yaml_vazio_usa_todos_os_padroes(tmp_path):
    config = carregar_config(escrever_yaml(tmp_path, ""))

    assert config.modelos.alvo == "fake/vendedor"
    assert config.amostragem.rodadas_iniciais == 3
    assert config.estatistica.confianca == 0.95
    assert config.estatistica.confianca_parada == 0.99


def test_temperaturas_podem_ser_null(tmp_path):
    config = carregar_config(escrever_yaml(tmp_path, "temperatura: null\ntemperatura_juiz: null\n"))
    assert config.temperatura is None and config.temperatura_juiz is None


def test_temperatura_juiz_padrao_e_zero():
    assert Config().temperatura_juiz == 0.0


def test_confianca_parada_igual_a_global_e_permitida():
    config = montar_config({"estatistica": {"confianca": 0.9, "confianca_parada": 0.9}}, pasta_base=Path("."))
    assert config.estatistica.confianca_parada == 0.9


def test_sobrescrever_um_campo_mantem_os_outros_da_secao(tmp_path):
    config = carregar_config(escrever_yaml(tmp_path, "amostragem:\n  rodadas_maximas: 20\n"))

    assert config.amostragem.rodadas_maximas == 20
    assert config.amostragem.rodadas_iniciais == 3  # continua no padrão


def test_semente_pode_ser_vazia(tmp_path):
    config = carregar_config(escrever_yaml(tmp_path, "estatistica:\n  semente: null\n"))
    assert config.estatistica.semente is None


# --- caminhos ---------------------------------------------------------------


def test_caminhos_relativos_sao_relativos_a_pasta_do_yaml(tmp_path):
    config = carregar_config(escrever_yaml(tmp_path, "caminhos:\n  casos: meus/casos.yaml\n"))

    assert config.caminhos.casos == tmp_path.resolve() / "meus" / "casos.yaml"
    assert config.caminhos.banco == tmp_path.resolve() / ".sinal" / "sinal.db"
    assert config.caminhos.rubrica is None


def test_caminho_absoluto_fica_como_esta(tmp_path):
    absoluto = (tmp_path / "outro" / "rubrica.txt").resolve()
    config = montar_config({"caminhos": {"rubrica": str(absoluto)}}, pasta_base=Path("/qualquer"))
    assert config.caminhos.rubrica == absoluto


# --- erros ------------------------------------------------------------------


def test_arquivo_inexistente_sugere_sinal_init(tmp_path):
    with pytest.raises(ErroConfig, match="sinal init"):
        carregar_config(tmp_path / "nao_existe.yaml")


def test_yaml_quebrado_da_erro_amigavel(tmp_path):
    with pytest.raises(ErroConfig, match="YAML válido"):
        carregar_config(escrever_yaml(tmp_path, "modelos: [sem fechar"))


def test_chave_com_erro_de_digitacao_e_recusada():
    with pytest.raises(ErroConfig, match="rodadas_maxima"):
        montar_config({"amostragem": {"rodadas_maxima": 10}}, pasta_base=Path("."))


def test_chave_desconhecida_na_raiz_e_recusada():
    with pytest.raises(ErroConfig, match="temperature"):
        montar_config({"temperature": 0.5}, pasta_base=Path("."))


def test_yaml_que_e_lista_na_raiz_e_recusado(tmp_path):
    with pytest.raises(ErroConfig, match="mapa"):
        carregar_config(escrever_yaml(tmp_path, "- modelos\n- temperatura\n"))


def test_secao_que_nao_e_mapa_e_recusada():
    with pytest.raises(ErroConfig, match="modelos"):
        montar_config({"modelos": "gpt-4o"}, pasta_base=Path("."))


@pytest.mark.parametrize(
    "dados, trecho_esperado",
    [
        ({"estatistica": {"confianca": 1.5}}, "confianca"),
        ({"estatistica": {"confianca": "0.95"}}, "confianca"),  # texto, não número
        ({"estatistica": {"reamostragens": 10}}, "reamostragens"),
        ({"estatistica": {"confianca_parada": 1.0}}, "confianca_parada"),
        ({"estatistica": {"confianca_parada": "0.99"}}, "confianca_parada"),
        ({"estatistica": {"confianca": 0.99, "confianca_parada": 0.95}}, "MAIS exigente"),
        ({"estatistica": {"limiar_irrelevante": -0.1}}, "limiar_irrelevante"),
        ({"temperatura": 3}, "temperatura"),
        ({"temperatura_juiz": -1}, "temperatura_juiz"),
        ({"temperatura_juiz": "zero"}, "temperatura_juiz"),
        ({"concorrencia": 0}, "concorrencia"),
        ({"concorrencia": True}, "concorrencia"),  # bool não vale como inteiro
        ({"amostragem": {"rodadas_iniciais": 1}}, "rodadas_iniciais"),
        ({"amostragem": {"rodadas_iniciais": 5, "rodadas_maximas": 4}}, "rodadas_maximas"),
        ({"amostragem": {"minimo_para_equivalencia": 50}}, "minimo_para_equivalencia"),
        ({"amostragem": {"tamanho_lote": "dois"}}, "tamanho_lote"),
        ({"modelos": {"juiz": ""}}, "modelos.juiz"),
    ],
)
def test_valores_invalidos_sao_recusados(dados, trecho_esperado):
    with pytest.raises(ErroConfig, match=trecho_esperado):
        montar_config(dados, pasta_base=Path("."))


def test_todos_os_problemas_aparecem_juntos():
    with pytest.raises(ErroConfig) as erro:
        montar_config({"temperatura": 9, "concorrencia": 0}, pasta_base=Path("."))
    assert "temperatura" in str(erro.value)
    assert "concorrencia" in str(erro.value)


# --- imutabilidade ----------------------------------------------------------


def test_config_e_imutavel():
    config = Config()
    with pytest.raises(FrozenInstanceError):
        config.temperatura = 1.0  # type: ignore[misc]

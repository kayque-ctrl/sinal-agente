"""Testes do cli.py: o Sinal usado como comando de terminal, do jeito que o usuário (e o CI) usa."""

import json
import runpy
import sys

import pytest
from typer.testing import CliRunner

from sinal.cli import SAIDA_ERRO, SAIDA_PIOROU, app

runner = CliRunner()


@pytest.fixture
def pasta(tmp_path, monkeypatch):
    """Roda os comandos de dentro de uma pasta temporária, com terminal largo."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    return tmp_path


@pytest.fixture
def projeto(pasta):
    """Projeto criado com `sinal init`, com reamostragens reduzidas pra o teste ser rápido."""
    assert runner.invoke(app, ["init"]).exit_code == 0
    yaml = pasta / "sinal.yaml"
    yaml.write_text(yaml.read_text(encoding="utf-8").replace("reamostragens: 10000", "reamostragens: 1000"),
                    encoding="utf-8")
    return pasta


# --- init -------------------------------------------------------------------


def test_init_cria_o_projeto(pasta):
    resultado = runner.invoke(app, ["init"])

    assert resultado.exit_code == 0
    assert "criado" in resultado.output and "sinal compara" in resultado.output
    for arquivo in ("sinal.yaml", "casos/casos.yaml", "prompts/v1.txt", "prompts/v2.txt"):
        assert (pasta / arquivo).is_file()


def test_init_em_outra_pasta(pasta):
    assert runner.invoke(app, ["init", "sub"]).exit_code == 0
    assert (pasta / "sub" / "sinal.yaml").is_file()


def test_init_duas_vezes_nao_apaga_nada(pasta):
    runner.invoke(app, ["init"])
    (pasta / "prompts" / "v1.txt").write_text("meu prompt de verdade", encoding="utf-8")

    resultado = runner.invoke(app, ["init"])

    assert resultado.exit_code == 0
    assert "mantido" in resultado.output
    assert (pasta / "prompts" / "v1.txt").read_text(encoding="utf-8") == "meu prompt de verdade"


# --- compara ----------------------------------------------------------------


def test_compara_mostra_resultado_e_salva_json(projeto):
    resultado = runner.invoke(app, ["compara", "prompts/v1.txt", "prompts/v2.txt"])

    assert resultado.exit_code == 0, resultado.output
    assert "Resultado por caso (indicativo)" in resultado.output
    assert "MELHOROU" in resultado.output
    [arquivo_json] = (projeto / ".sinal" / "resultados").glob("*.json")
    assert json.loads(arquivo_json.read_text(encoding="utf-8"))["veredito"] == "MELHOROU"


def test_compara_piorou_sem_flag_sai_com_zero(projeto):
    resultado = runner.invoke(app, ["compara", "prompts/v2.txt", "prompts/v1.txt"])
    assert resultado.exit_code == 0
    assert "PIOROU" in resultado.output


def test_compara_piorou_com_flag_sai_com_um(projeto):
    resultado = runner.invoke(app, ["compara", "prompts/v2.txt", "prompts/v1.txt", "--falhar-se-piorar"])
    assert resultado.exit_code == SAIDA_PIOROU == 1


def test_compara_melhorou_com_flag_sai_com_zero(projeto):
    resultado = runner.invoke(app, ["compara", "prompts/v1.txt", "prompts/v2.txt", "--falhar-se-piorar"])
    assert resultado.exit_code == 0


def test_compara_com_casos_e_config_explicitos(projeto, pasta_dados):
    (projeto / "outra").mkdir()
    (projeto / "sinal.yaml").rename(projeto / "outra" / "sinal.yaml")  # o yaml muda de pasta...

    resultado = runner.invoke(
        app,
        [
            "compara", "prompts/v1.txt", "prompts/v2.txt",
            "--casos", str(pasta_dados / "casos_vendas.yaml"),
            "--config", "outra/sinal.yaml",
        ],
    )
    assert resultado.exit_code == 0, resultado.output
    assert (projeto / "outra" / ".sinal" / "sinal.db").is_file()  # ...e o banco vai junto com ele


def test_compara_sem_config_sugere_init(pasta):
    resultado = runner.invoke(app, ["compara", "a.txt", "b.txt"])
    assert resultado.exit_code == SAIDA_ERRO == 2
    assert "sinal init" in resultado.output


def test_compara_prompt_inexistente(projeto):
    resultado = runner.invoke(app, ["compara", "prompts/v1.txt", "prompts/v9.txt"])
    assert resultado.exit_code == SAIDA_ERRO
    assert "v9.txt" in resultado.output


def test_compara_tudo_falhando_sai_com_erro(projeto):
    yaml = projeto / "sinal.yaml"
    yaml.write_text(yaml.read_text(encoding="utf-8").replace("alvo: fake/vendedor", "alvo: fake/erro"),
                    encoding="utf-8")
    resultado = runner.invoke(app, ["compara", "prompts/v1.txt", "prompts/v2.txt"])
    assert resultado.exit_code == SAIDA_ERRO
    assert "Nenhum caso" in resultado.output


# --- historico --------------------------------------------------------------


def test_historico_sem_banco(projeto):
    resultado = runner.invoke(app, ["historico"])
    assert resultado.exit_code == 0
    assert "Nenhuma comparação ainda" in resultado.output


def test_historico_depois_de_comparar(projeto):
    runner.invoke(app, ["compara", "prompts/v1.txt", "prompts/v2.txt"])
    runner.invoke(app, ["compara", "prompts/v2.txt", "prompts/v1.txt"])

    resultado = runner.invoke(app, ["historico"])
    assert resultado.exit_code == 0
    assert "MELHOROU" in resultado.output and "PIOROU" in resultado.output

    so_um = runner.invoke(app, ["historico", "-n", "1"])
    assert "PIOROU" in so_um.output and "MELHOROU" not in so_um.output  # só a mais recente


def test_historico_sem_config(pasta):
    assert runner.invoke(app, ["historico"]).exit_code == SAIDA_ERRO


# --- calibra ----------------------------------------------------------------


def test_calibra_exportar_e_avaliar(projeto):
    import csv

    runner.invoke(app, ["compara", "prompts/v1.txt", "prompts/v2.txt"])

    exportado = runner.invoke(app, ["calibra", "exportar", "-n", "10", "-o", "cal.csv"])
    assert exportado.exit_code == 0, exportado.output
    assert "10 respostas exportadas" in exportado.output and "sinal calibra avaliar" in exportado.output

    # "Você" preenche o CSV dando 3 pra tudo
    caminho = projeto / "cal.csv"
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo, delimiter=";"))
    for linha in linhas:
        linha["nota_humana"] = "3"
    with caminho.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]), delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)

    avaliado = runner.invoke(app, ["calibra", "avaliar", "cal.csv"])
    assert avaliado.exit_code == 0, avaliado.output
    assert "kappa ponderado" in avaliado.output and "Matriz de confusão" in avaliado.output


def test_calibra_exportar_caminho_padrao(projeto):
    runner.invoke(app, ["compara", "prompts/v1.txt", "prompts/v2.txt"])
    assert runner.invoke(app, ["calibra", "exportar"]).exit_code == 0
    assert (projeto / ".sinal" / "calibracao" / "calibracao-ultima.csv").is_file()


def test_calibra_sem_comparacoes(projeto):
    resultado = runner.invoke(app, ["calibra", "exportar"])
    assert resultado.exit_code == SAIDA_ERRO and "sinal compara" in resultado.output
    assert runner.invoke(app, ["calibra", "avaliar", "x.csv"]).exit_code == SAIDA_ERRO


def test_calibra_erros_viram_mensagem(projeto):
    runner.invoke(app, ["compara", "prompts/v1.txt", "prompts/v2.txt"])
    assert runner.invoke(app, ["calibra", "exportar", "--comparacao", "999"]).exit_code == SAIDA_ERRO
    resultado = runner.invoke(app, ["calibra", "avaliar", "nao-existe.csv"])
    assert resultado.exit_code == SAIDA_ERRO and "Não achei" in resultado.output


# --- ajuda e python -m sinal ------------------------------------------------


def test_sem_argumentos_mostra_ajuda():
    resultado = runner.invoke(app, [])
    assert "compara" in resultado.output and "historico" in resultado.output


def test_python_m_sinal(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sinal", "--help"])
    with pytest.raises(SystemExit) as saida:
        runpy.run_module("sinal", run_name="__main__")
    assert saida.value.code == 0

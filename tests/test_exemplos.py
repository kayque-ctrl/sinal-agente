"""Testes do exemplos.py: o exemplo do `sinal init` precisa ser exatamente o que os testes validam."""

from pathlib import Path

from sinal.casos import carregar_casos
from sinal.config import Config, carregar_config
from sinal.exemplos import CASOS_YAML, PROMPT_V1, PROMPT_V2, criar_projeto


def test_exemplo_e_identico_aos_dados_de_teste(pasta_dados):
    """Se alguém mudar um sem mudar o outro, o exemplo deixaria de ser o que foi testado."""
    assert CASOS_YAML == (pasta_dados / "casos_vendas.yaml").read_text(encoding="utf-8")
    assert PROMPT_V1 == (pasta_dados / "prompt_v1.txt").read_text(encoding="utf-8")
    assert PROMPT_V2 == (pasta_dados / "prompt_v2.txt").read_text(encoding="utf-8")


def test_criar_projeto_escreve_tudo(tmp_path):
    criados, ja_existiam = criar_projeto(tmp_path)

    assert sorted(p.as_posix() for p in criados) == [
        "casos/casos.yaml",
        "prompts/v1.txt",
        "prompts/v2.txt",
        "sinal.yaml",
    ]
    assert ja_existiam == []
    assert len(carregar_casos(tmp_path / "casos" / "casos.yaml")) == 6


def test_sinal_yaml_de_exemplo_tem_os_mesmos_valores_dos_padroes(tmp_path):
    """O sinal.yaml comentado não pode 'mentir' sobre os padrões do código."""
    criar_projeto(tmp_path)
    config = carregar_config(tmp_path / "sinal.yaml")
    padrao = Config()

    assert config.modelos == padrao.modelos
    assert config.amostragem == padrao.amostragem
    assert config.estatistica == padrao.estatistica
    assert (config.temperatura, config.temperatura_juiz, config.concorrencia) == (
        padrao.temperatura,
        padrao.temperatura_juiz,
        padrao.concorrencia,
    )
    assert config.caminhos.casos == tmp_path.resolve() / "casos" / "casos.yaml"


def test_nunca_sobrescreve_arquivo_existente(tmp_path):
    (tmp_path / "sinal.yaml").write_text("# meu arquivo\n", encoding="utf-8")

    criados, ja_existiam = criar_projeto(tmp_path)

    assert ja_existiam == [Path("sinal.yaml")]
    assert Path("sinal.yaml") not in criados
    assert (tmp_path / "sinal.yaml").read_text(encoding="utf-8") == "# meu arquivo\n"

"""Testes ponta a ponta do comparador, com modelos fake (sem API, sem custo)."""

import shutil

import pytest

from sinal import comparador
from sinal.armazenamento import Armazenamento
from sinal.comparador import ErroComparacao, EstadoCaso, comparar, veredito_do_caso
from sinal.casos import Caso
from sinal.config import montar_config
from sinal.estatistica import Intervalo, Veredito
from sinal.llm import chamar as chamar_original


@pytest.fixture
def projeto(tmp_path, pasta_dados):
    """Uma pasta de projeto do Sinal com os casos e prompts de vendas."""
    (tmp_path / "casos").mkdir()
    (tmp_path / "prompts").mkdir()
    shutil.copy(pasta_dados / "casos_vendas.yaml", tmp_path / "casos" / "casos.yaml")
    shutil.copy(pasta_dados / "prompt_v1.txt", tmp_path / "prompts" / "v1.txt")
    shutil.copy(pasta_dados / "prompt_v2.txt", tmp_path / "prompts" / "v2.txt")
    return tmp_path


def criar_config(pasta, **sobrescrever):
    dados = {
        "concorrencia": 4,
        "estatistica": {"reamostragens": 2000, "semente": 42},
        **sobrescrever,
    }
    return montar_config(dados, pasta_base=pasta)


# ===========================================================================
# Ponta a ponta
# ===========================================================================


def test_v2_melhor_que_v1_da_melhorou(projeto):
    config = criar_config(projeto)
    resultado = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")

    assert resultado.veredito == Veredito.MELHOROU
    assert resultado.intervalo.inferior > 0
    assert resultado.casos_no_global == 6
    assert resultado.erros == 0
    assert resultado.custo_total_usd == 0.0  # fake não custa nada


def test_v2_contra_v1_invertido_da_piorou(projeto):
    config = criar_config(projeto)
    resultado = comparar(config, projeto / "prompts" / "v2.txt", projeto / "prompts" / "v1.txt")
    assert resultado.veredito == Veredito.PIOROU


def test_prompt_contra_ele_mesmo_nao_inventa_diferenca(projeto):
    config = criar_config(projeto)
    resultado = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v1.txt")
    assert resultado.veredito not in (Veredito.MELHOROU, Veredito.PIOROU)


def test_tudo_fica_gravado_no_banco(projeto):
    config = criar_config(projeto)
    resultado = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")

    with Armazenamento(config.caminhos.banco) as banco:
        comparacao = banco.buscar_comparacao(resultado.comparacao_id)
        execucoes = banco.ler_execucoes(resultado.comparacao_id)

    assert comparacao["status"] == "concluida"
    assert comparacao["veredito"] == "MELHOROU"
    assert comparacao["resultado"]["casos_no_global"] == 6  # o JSON completo vai junto
    assert len(execucoes) == resultado.execucoes
    assert all(e["nota"] in (1, 2, 3, 4, 5) for e in execucoes)
    assert all(e["justificativa"] for e in execucoes)


# ===========================================================================
# Amostragem adaptativa
# ===========================================================================


def test_casos_claros_param_cedo_e_ha_economia(projeto):
    config = criar_config(projeto)
    resultado = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")

    rodadas = [c.rodadas for c in resultado.casos]
    assert min(rodadas) == config.amostragem.rodadas_iniciais  # pelo menos um caso decidiu logo
    assert resultado.chamadas_economizadas > 0
    assert 0 < resultado.economia_percentual < 100


def test_inconclusivos_rodam_mais_ate_o_teto(projeto):
    config = criar_config(projeto)
    resultado = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v1.txt")

    for caso in resultado.casos:
        assert caso.rodadas <= config.amostragem.rodadas_maximas  # nunca passa do teto
        if caso.veredito == Veredito.INCONCLUSIVO:
            assert caso.rodadas == config.amostragem.rodadas_maximas  # só para no teto


def test_contagem_de_chamadas_bate(projeto):
    config = criar_config(projeto)
    resultado = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")

    soma_rodadas = sum(c.rodadas for c in resultado.casos)
    assert resultado.execucoes == soma_rodadas * 2  # A e B
    assert resultado.chamadas_feitas == resultado.execucoes * 2  # alvo + juiz
    assert resultado.chamadas_maximas == 6 * 2 * config.amostragem.rodadas_maximas * 2


def test_n_minimo_bloqueia_equivalencia_com_poucas_notas():
    config = montar_config({"amostragem": {"minimo_para_equivalencia": 6}}, pasta_base=".")
    caso = Caso("c", "Oi", "x")
    ic_zero = Intervalo(0.0, 0.0, 0.0)

    poucas = EstadoCaso(caso, notas_a=[4, 4, 4], notas_b=[4, 4, 4])
    suficientes = EstadoCaso(caso, notas_a=[4] * 6, notas_b=[4] * 6)
    com_erros = EstadoCaso(caso, notas_a=[4] * 6, notas_b=[4] * 4, erros=2)  # rodou 6, mas 2 falharam

    assert veredito_do_caso(ic_zero, poucas, config) == Veredito.INCONCLUSIVO
    assert veredito_do_caso(ic_zero, suficientes, config) == Veredito.EQUIVALENTE
    assert veredito_do_caso(ic_zero, com_erros, config) == Veredito.INCONCLUSIVO


def test_veredito_por_caso_usa_confianca_parada_e_global_usa_confianca(projeto, monkeypatch):
    usadas = {"caso": set(), "global": set()}
    bootstrap_caso_original = comparador.bootstrap_caso
    bootstrap_hierarquico_original = comparador.bootstrap_hierarquico

    def espiar_caso(*args, **kwargs):
        usadas["caso"].add(kwargs["confianca"])
        return bootstrap_caso_original(*args, **kwargs)

    def espiar_global(*args, **kwargs):
        usadas["global"].add(kwargs["confianca"])
        return bootstrap_hierarquico_original(*args, **kwargs)

    monkeypatch.setattr(comparador, "bootstrap_caso", espiar_caso)
    monkeypatch.setattr(comparador, "bootstrap_hierarquico", espiar_global)
    config = criar_config(projeto, estatistica={"confianca": 0.9, "confianca_parada": 0.97, "reamostragens": 500})
    comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")

    assert usadas == {"caso": {0.97}, "global": {0.9}}


def test_temperaturas_da_config_chegam_no_alvo_e_no_juiz(projeto, monkeypatch):
    recebidas = {"alvo": set(), "juiz": set()}

    def chamar_espiao(modelo, mensagens, temperatura=0.0, *, gerador=None):
        recebidas["alvo"].add(temperatura)
        return chamar_original(modelo, mensagens, temperatura, gerador=gerador)

    avaliar_original = comparador.avaliar

    def avaliar_espiao(*args, temperatura, **kwargs):
        recebidas["juiz"].add(temperatura)
        return avaliar_original(*args, temperatura=temperatura, **kwargs)

    monkeypatch.setattr(comparador, "chamar", chamar_espiao)
    monkeypatch.setattr(comparador, "avaliar", avaliar_espiao)
    config = criar_config(projeto, temperatura=None, temperatura_juiz=None)
    comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")

    assert recebidas == {"alvo": {None}, "juiz": {None}}


def test_alarme_falso_por_caso_fica_controlado(projeto):
    """v1 contra v1 em 10 sementes: quantos casos 'inventam' MELHOROU/PIOROU?

    Com a confiança global (0.95) nas espiadas, isso dava ~21%. Com confianca_parada = 0.99, ~12%.
    """
    falsos = total = 0
    for semente in range(10):
        config = criar_config(projeto, estatistica={"reamostragens": 1000, "semente": semente})
        resultado = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v1.txt")
        falsos += sum(c.veredito in (Veredito.MELHOROU, Veredito.PIOROU) for c in resultado.casos)
        total += len(resultado.casos)
    print(f"\nalarme falso por caso: {falsos / total:.0%}")
    assert falsos / total <= 0.18


# ===========================================================================
# Reprodutibilidade e progresso
# ===========================================================================


def test_mesma_semente_mesmo_resultado_mesmo_com_threads(projeto):
    config = criar_config(projeto, concorrencia=8)
    a = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")
    b = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")

    assert [(c.notas_a, c.notas_b) for c in a.casos] == [(c.notas_a, c.notas_b) for c in b.casos]
    assert a.intervalo == b.intervalo
    assert a.comparacao_id != b.comparacao_id  # mas são duas comparações no banco


def test_sem_semente_tambem_funciona(projeto):
    """semente: null → sorteios de verdade aleatórios; o v2 continua ganhando (a diferença é real)."""
    config = criar_config(projeto, estatistica={"reamostragens": 2000, "semente": None})
    resultado = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")
    assert resultado.veredito == Veredito.MELHOROU


def test_ao_progredir_e_chamado_uma_vez_por_execucao(projeto):
    config = criar_config(projeto)
    recebidas = []
    resultado = comparar(
        config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt", ao_progredir=recebidas.append
    )
    assert len(recebidas) == resultado.execucoes


def test_aceita_banco_externo_e_caminho_de_casos(projeto, pasta_dados, tmp_path):
    config = criar_config(projeto)
    with Armazenamento(tmp_path / "outro.db") as banco:
        resultado = comparar(
            config,
            projeto / "prompts" / "v1.txt",
            projeto / "prompts" / "v2.txt",
            caminho_casos=pasta_dados / "casos_vendas.yaml",
            banco=banco,
        )
        comparacao = banco.buscar_comparacao(resultado.comparacao_id)
        assert comparacao["status"] == "concluida"
        # a config salva aponta pro arquivo de casos REALMENTE usado (o do --casos), não o do sinal.yaml
        assert comparacao["config"]["caminhos"]["casos"] == str((pasta_dados / "casos_vendas.yaml").resolve())


def test_para_dict_e_serializavel(projeto):
    import json

    config = criar_config(projeto)
    resultado = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")
    dados = json.loads(json.dumps(resultado.para_dict()))

    assert dados["veredito"] == "MELHOROU"
    assert dados["casos"][0]["veredito"] in {v.value for v in Veredito}
    assert "economia_percentual" in dados


# ===========================================================================
# Falhas
# ===========================================================================


def test_tudo_falhando_marca_a_comparacao_como_falhou(projeto):
    config = criar_config(projeto, modelos={"alvo": "fake/erro"})

    with pytest.raises(ErroComparacao, match="Nenhum caso"):
        comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")

    with Armazenamento(config.caminhos.banco) as banco:
        [comparacao] = banco.listar_comparacoes()
        execucoes = banco.ler_execucoes(comparacao["id"])
    assert comparacao["status"] == "falhou"
    assert execucoes and all(e["erro"].startswith("alvo: RuntimeError") for e in execucoes)


def test_falhas_parciais_nao_derrubam_a_rodada(projeto, monkeypatch):
    """O v2 sempre falha no caso do parcelamento: esse caso sai do global, o resto segue."""

    def chamar_instavel(modelo, mensagens, temperatura=0.0, *, gerador=None):
        texto = " ".join(m["content"] for m in mensagens)
        # Cuidado: o próprio prompt v2 fala em "parcelamento"; por isso a frase exata do caso.
        if modelo == "fake/vendedor" and "Regras:" in texto and "Vocês parcelam no cartão?" in texto:
            return chamar_original("fake/erro", mensagens)
        return chamar_original(modelo, mensagens, temperatura, gerador=gerador)

    monkeypatch.setattr(comparador, "chamar", chamar_instavel)
    config = criar_config(projeto)
    resultado = comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")

    parcelamento = next(c for c in resultado.casos if c.caso_id == "duvida-parcelamento")
    assert parcelamento.notas_b == [] and parcelamento.intervalo is None
    assert parcelamento.veredito == Veredito.INCONCLUSIVO
    assert parcelamento.rodadas == config.amostragem.rodadas_maximas  # tentou até o teto
    assert parcelamento.erros == parcelamento.rodadas  # todas as rodadas do B falharam
    assert resultado.casos_no_global == 5
    assert resultado.veredito == Veredito.MELHOROU  # os outros 5 casos ainda decidem
    assert resultado.chamadas_feitas < resultado.execucoes * 2  # quando o alvo falha, o juiz nem é chamado


def test_bug_inesperado_vira_erro_interno(projeto, monkeypatch):
    def avaliar_quebrado(*args, **kwargs):
        raise KeyError("bug")

    monkeypatch.setattr(comparador, "avaliar", avaliar_quebrado)
    config = criar_config(projeto, amostragem={"rodadas_iniciais": 2, "rodadas_maximas": 2,
                                               "minimo_para_equivalencia": 2})

    with pytest.raises(ErroComparacao):
        comparar(config, projeto / "prompts" / "v1.txt", projeto / "prompts" / "v2.txt")
    with Armazenamento(config.caminhos.banco) as banco:
        execucoes = banco.ler_execucoes(banco.listar_comparacoes()[0]["id"])
    assert all(e["erro"].startswith("interno: KeyError") for e in execucoes)

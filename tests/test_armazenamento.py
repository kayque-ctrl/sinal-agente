"""Testes do armazenamento.py."""

import sqlite3
import threading

import pytest

from sinal.armazenamento import Armazenamento, Execucao, resumo_hash


@pytest.fixture
def banco(tmp_path):
    with Armazenamento(tmp_path / ".sinal" / "sinal.db") as banco:
        yield banco


def nova_comparacao(banco, conteudo_b="Você é vendedor.\n- Pergunte a necessidade."):
    return banco.criar_comparacao(
        prompt_a="prompts/v1.txt",
        prompt_b="prompts/v2.txt",
        conteudo_a="Você é vendedor.",
        conteudo_b=conteudo_b,
        modelo_alvo="fake/vendedor",
        modelo_juiz="fake/juiz",
        config={"semente": 42, "limiar": 0.25},
    )


# --- criação ----------------------------------------------------------------


def test_cria_o_arquivo_e_a_pasta(tmp_path):
    caminho = tmp_path / "nova" / "pasta" / "sinal.db"
    Armazenamento(caminho).fechar()
    assert caminho.is_file()


def test_reabrir_o_banco_mantem_os_dados(tmp_path):
    caminho = tmp_path / "sinal.db"
    with Armazenamento(caminho) as banco:
        comparacao_id = nova_comparacao(banco)
    with Armazenamento(caminho) as banco:  # CREATE TABLE IF NOT EXISTS: não apaga nada
        assert banco.buscar_comparacao(comparacao_id) is not None


def test_banco_em_memoria():
    with Armazenamento(":memory:") as banco:
        assert nova_comparacao(banco) == 1


# --- comparações ------------------------------------------------------------


def test_criar_comparacao(banco):
    comparacao = banco.buscar_comparacao(nova_comparacao(banco))

    assert comparacao["status"] == "em_andamento"
    assert comparacao["hash_a"] == resumo_hash("Você é vendedor.")
    assert comparacao["hash_a"] != comparacao["hash_b"]
    assert comparacao["config"] == {"semente": 42, "limiar": 0.25}  # JSON volta como dict
    assert comparacao["resultado"] is None  # só é preenchido ao finalizar
    assert comparacao["veredito"] is None


def test_hash_muda_quando_o_prompt_muda():
    assert resumo_hash("v2") == resumo_hash("v2")
    assert resumo_hash("v2") != resumo_hash("v2 ")
    assert len(resumo_hash("qualquer")) == 12


def test_finalizar_comparacao(banco):
    comparacao_id = nova_comparacao(banco)
    banco.finalizar_comparacao(
        comparacao_id,
        veredito="MELHOROU",
        diferenca=1.2,
        ic_inferior=0.8,
        ic_superior=1.6,
        p_valor=0.03,
        chamadas_feitas=120,
        chamadas_economizadas=168,
        resultado={"casos": [{"id": "objecao-preco", "veredito": "MELHOROU"}]},
    )
    comparacao = banco.buscar_comparacao(comparacao_id)

    assert comparacao["status"] == "concluida"
    assert (comparacao["veredito"], comparacao["diferenca"], comparacao["p_valor"]) == ("MELHOROU", 1.2, 0.03)
    assert comparacao["resultado"]["casos"][0]["id"] == "objecao-preco"


def test_marcar_falha(banco):
    comparacao_id = nova_comparacao(banco)
    banco.marcar_falha(comparacao_id, "todas as chamadas falharam")
    comparacao = banco.buscar_comparacao(comparacao_id)
    assert (comparacao["status"], comparacao["erro"]) == ("falhou", "todas as chamadas falharam")


def test_buscar_comparacao_inexistente(banco):
    assert banco.buscar_comparacao(999) is None


def test_listar_comparacoes_mais_recente_primeiro_com_totais(banco):
    primeira = nova_comparacao(banco)
    segunda = nova_comparacao(banco)
    banco.salvar_execucoes(
        segunda,
        [
            Execucao("c1", "A", 1, resposta="oi", nota=3, custo_usd=0.01, custo_juiz_usd=0.002),
            Execucao("c1", "B", 1, erro="RateLimitError: calma"),
        ],
    )

    historico = banco.listar_comparacoes()

    assert [c["id"] for c in historico] == [segunda, primeira]
    assert historico[0]["total_execucoes"] == 2
    assert historico[0]["total_erros"] == 1
    assert historico[0]["custo_total_usd"] == pytest.approx(0.012)
    assert historico[1]["total_execucoes"] == 0 and historico[1]["custo_total_usd"] == 0  # LEFT JOIN
    assert len(banco.listar_comparacoes(limite=1)) == 1


# --- execuções --------------------------------------------------------------


def test_salvar_e_ler_execucao_completa(banco):
    comparacao_id = nova_comparacao(banco)
    execucao = Execucao(
        caso_id="objecao-preco",
        versao="B",
        rodada=1,
        resposta="Entendo! Me conta como vocês fazem hoje?",
        nota=4,
        justificativa="Tratou a objeção.",
        metodo_parse="json",
        tokens_entrada=120,
        tokens_saida=30,
        latencia_ms=850.5,
        custo_usd=0.0004,
        tokens_juiz=300,
        custo_juiz_usd=0.0009,
    )
    banco.salvar_execucao(comparacao_id, execucao)

    [linha] = banco.ler_execucoes(comparacao_id)
    for campo in ("caso_id", "versao", "rodada", "resposta", "nota", "justificativa", "latencia_ms", "custo_juiz_usd"):
        assert linha[campo] == getattr(execucao, campo)
    assert linha["erro"] is None
    assert linha["criado_em"]


def test_execucao_com_erro_e_sem_nota(banco):
    comparacao_id = nova_comparacao(banco)
    banco.salvar_execucao(comparacao_id, Execucao("c1", "A", 1, erro="juiz: não consegui extrair a nota"))
    [linha] = banco.ler_execucoes(comparacao_id)
    assert linha["nota"] is None and linha["erro"].startswith("juiz:")


def test_filtros_de_ler_execucoes(banco):
    comparacao_id = nova_comparacao(banco)
    banco.salvar_execucoes(
        comparacao_id,
        [
            Execucao("c1", "A", 1, nota=3),
            Execucao("c1", "B", 1, nota=5),
            Execucao("c2", "A", 1, nota=2),
            Execucao("c2", "B", 1, erro="falhou"),
        ],
    )

    assert len(banco.ler_execucoes(comparacao_id)) == 4
    assert [e["versao"] for e in banco.ler_execucoes(comparacao_id, caso_id="c1")] == ["A", "B"]
    assert len(banco.ler_execucoes(comparacao_id, versao="B")) == 2
    assert [e["caso_id"] for e in banco.ler_execucoes(comparacao_id, apenas_erros=True)] == ["c2"]
    assert len(banco.ler_execucoes(comparacao_id, limite=3)) == 3


def test_filtro_nao_permite_sql_injection(banco):
    comparacao_id = nova_comparacao(banco)
    banco.salvar_execucao(comparacao_id, Execucao("c1", "A", 1, nota=3))
    # Se o valor fosse colado direto no SQL, isso viraria "... OR 1=1" e traria tudo.
    assert banco.ler_execucoes(comparacao_id, caso_id="x' OR '1'='1") == []


def test_notas_por_caso_ignora_erros(banco):
    comparacao_id = nova_comparacao(banco)
    banco.salvar_execucoes(
        comparacao_id,
        [
            Execucao("c1", "A", 1, nota=3),
            Execucao("c1", "A", 2, nota=2),
            Execucao("c1", "B", 1, nota=5),
            Execucao("c1", "B", 2, erro="falhou"),
            Execucao("c2", "A", 1, nota=4),
        ],
    )
    assert banco.notas_por_caso(comparacao_id) == {"c1": ([3, 2], [5]), "c2": ([4], [])}


# --- regras do banco (CHECK, UNIQUE, FK) ------------------------------------


@pytest.mark.parametrize(
    "execucao",
    [
        Execucao("c1", "A", 1, nota=7),  # nota fora de 1-5
        Execucao("c1", "A", 1, nota=0),
        Execucao("c1", "C", 1, nota=3),  # versão que não existe
        Execucao("c1", "A", 0, nota=3),  # rodada começa em 1
    ],
)
def test_banco_recusa_dado_invalido(banco, execucao):
    comparacao_id = nova_comparacao(banco)
    with pytest.raises(sqlite3.IntegrityError):
        banco.salvar_execucao(comparacao_id, execucao)


def test_mesma_rodada_nao_pode_ser_gravada_duas_vezes(banco):
    comparacao_id = nova_comparacao(banco)
    banco.salvar_execucao(comparacao_id, Execucao("c1", "A", 1, nota=3))
    with pytest.raises(sqlite3.IntegrityError):
        banco.salvar_execucao(comparacao_id, Execucao("c1", "A", 1, nota=4))


def test_lote_com_erro_nao_grava_nada(banco):
    """Transação: ou grava o lote inteiro, ou nada."""
    comparacao_id = nova_comparacao(banco)
    with pytest.raises(sqlite3.IntegrityError):
        banco.salvar_execucoes(comparacao_id, [Execucao("c1", "A", 1, nota=3), Execucao("c1", "A", 2, nota=9)])
    assert banco.ler_execucoes(comparacao_id) == []


def test_execucao_de_comparacao_inexistente_e_recusada(banco):
    with pytest.raises(sqlite3.IntegrityError):
        banco.salvar_execucao(999, Execucao("c1", "A", 1, nota=3))


# --- concorrência -----------------------------------------------------------


def test_varias_threads_gravando_ao_mesmo_tempo(banco):
    comparacao_id = nova_comparacao(banco)

    def gravar(numero_thread):
        for rodada in range(1, 51):
            banco.salvar_execucao(comparacao_id, Execucao(f"caso-{numero_thread}", "A", rodada, nota=3))

    threads = [threading.Thread(target=gravar, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(banco.ler_execucoes(comparacao_id)) == 8 * 50

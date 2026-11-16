"""Testes da calibração do juiz."""

import csv
import shutil

import pytest
from rich.console import Console

from sinal.armazenamento import Armazenamento, Execucao
from sinal.calibracao import (
    CODIFICACAO,
    COLUNAS_CSV,
    ErroCalibracao,
    amostra_estratificada,
    avaliar,
    exportar,
    ler_notas_humanas,
)
from sinal.comparador import comparar
from sinal.config import montar_config
from sinal.relatorio import mostrar_calibracao


@pytest.fixture
def banco_comparado(tmp_path, pasta_dados):
    """Banco com uma comparação v1 vs v2 já concluída (modelos fake)."""
    for nome in ("prompt_v1.txt", "prompt_v2.txt"):
        shutil.copy(pasta_dados / nome, tmp_path / nome)
    config = montar_config(
        {
            "estatistica": {"reamostragens": 500, "semente": 1},
            "caminhos": {"casos": str(pasta_dados / "casos_vendas.yaml")},
        },
        pasta_base=tmp_path,
    )
    resultado = comparar(config, tmp_path / "prompt_v1.txt", tmp_path / "prompt_v2.txt")
    with Armazenamento(config.caminhos.banco) as banco:
        yield banco, resultado.comparacao_id


def ler_csv(caminho):
    with caminho.open(encoding=CODIFICACAO, newline="") as arquivo:
        return list(csv.DictReader(arquivo, delimiter=";"))


def preencher(caminho, nota_para, separador=";"):
    """Simula você preenchendo a coluna nota_humana no Excel."""
    linhas = ler_csv(caminho)
    for linha in linhas:
        linha["nota_humana"] = nota_para(int(linha["id_execucao"]))
    with caminho.open("w", encoding=CODIFICACAO, newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_CSV, delimiter=separador)
        escritor.writeheader()
        escritor.writerows(linhas)


def notas_do_juiz(banco, comparacao_id):
    return {e["id"]: e["nota"] for e in banco.ler_execucoes(comparacao_id)}


# ===========================================================================
# Exportar
# ===========================================================================


def test_exportar_e_as_cegas(banco_comparado, tmp_path):
    banco, comparacao_id = banco_comparado
    caminho, quantidade = exportar(banco, tmp_path / "cal.csv", tamanho=20, semente=0)
    linhas = ler_csv(caminho)

    assert quantidade == len(linhas) == 20
    assert list(linhas[0]) == COLUNAS_CSV
    for proibida in ("nota", "justificativa", "versao"):  # nada que entregue o juiz ou a versão
        assert proibida not in linhas[0]
    assert all(linha["nota_humana"] == "" for linha in linhas)
    assert all("[qualidade=" not in linha["resposta"] for linha in linhas)  # sem a "cola" do fake
    assert all(linha["entrada"] and linha["criterio"] for linha in linhas)


def test_exportar_usa_separador_e_codificacao_do_excel_brasileiro(banco_comparado, tmp_path):
    banco, _ = banco_comparado
    caminho, _ = exportar(banco, tmp_path / "cal.csv", tamanho=3)
    bruto = caminho.read_bytes()
    assert bruto.startswith(b"\xef\xbb\xbf")  # BOM do UTF-8
    assert bruto.decode("utf-8-sig").splitlines()[0] == ";".join(COLUNAS_CSV)


def test_exportar_padrao_e_a_ultima_comparacao_concluida(banco_comparado, tmp_path):
    banco, comparacao_id = banco_comparado
    caminho, _ = exportar(banco, tmp_path / "cal.csv", tamanho=5)
    ids = {int(linha["id_execucao"]) for linha in ler_csv(caminho)}
    assert ids <= set(notas_do_juiz(banco, comparacao_id))


def test_exportar_tamanho_maior_que_o_disponivel_exporta_tudo(banco_comparado, tmp_path):
    banco, comparacao_id = banco_comparado
    _, quantidade = exportar(banco, tmp_path / "cal.csv", tamanho=10_000)
    assert quantidade == len(notas_do_juiz(banco, comparacao_id))


def test_exportar_erros(tmp_path, banco_comparado):
    banco, _ = banco_comparado
    with pytest.raises(ErroCalibracao, match="não existe"):
        exportar(banco, tmp_path / "x.csv", comparacao_id=999)
    with pytest.raises(ErroCalibracao, match="pelo menos 1"):
        exportar(banco, tmp_path / "x.csv", tamanho=0)

    with Armazenamento(tmp_path / "vazio.db") as vazio:
        with pytest.raises(ErroCalibracao, match="Nenhuma comparação concluída"):
            exportar(vazio, tmp_path / "x.csv")
        comparacao_id = vazio.criar_comparacao("a", "b", "x", "y", "fake/vendedor", "fake/juiz")
        vazio.salvar_execucao(comparacao_id, Execucao("c1", "A", 1, erro="falhou"))
        with pytest.raises(ErroCalibracao, match="não tem respostas avaliadas"):
            exportar(vazio, tmp_path / "x.csv", comparacao_id=comparacao_id)


def test_exportar_sem_arquivo_de_casos_ainda_funciona(tmp_path):
    """Se o arquivo de casos sumiu, exporta só com a resposta (entrada e critério vazios)."""
    with Armazenamento(tmp_path / "b.db") as banco:
        comparacao_id = banco.criar_comparacao(
            "a", "b", "x", "y", "fake/vendedor", "fake/juiz", config={"caminhos": {"casos": str(tmp_path / "sumiu.yaml")}}
        )
        banco.salvar_execucao(comparacao_id, Execucao("c1", "A", 1, resposta="oi", nota=3))
        caminho, quantidade = exportar(banco, tmp_path / "x.csv", comparacao_id=comparacao_id)
    [linha] = ler_csv(caminho)
    assert quantidade == 1 and linha["resposta"] == "oi" and linha["entrada"] == ""


# --- amostra estratificada --------------------------------------------------


def test_amostra_estratificada_representa_todas_as_notas():
    import random

    # Juiz "preguiçoso": 90 notas 4 e só 2 de cada outra nota.
    execucoes = [{"id": i, "nota": 4} for i in range(90)]
    execucoes += [{"id": 100 + i, "nota": n} for n in (1, 2, 3, 5) for i in range(2)]

    amostra = amostra_estratificada(execucoes, 10, random.Random(0))

    notas = [e["nota"] for e in amostra]
    assert sorted(set(notas)) == [1, 2, 3, 4, 5]  # todas aparecem
    assert notas.count(4) == 2  # sem estratificar, seriam ~9 notas 4


def test_amostra_estratificada_embaralha_a_ordem():
    """Em rodízio sem embaralhar, a ordem seria 1,2,3,4,5,1,2,... e entregaria a nota do juiz."""
    import random

    execucoes = [{"id": n * 10 + i, "nota": n} for n in (1, 2, 3, 4, 5) for i in range(4)]
    notas = [e["nota"] for e in amostra_estratificada(execucoes, 20, random.Random(0))]
    assert notas != [1, 2, 3, 4, 5] * 4


# ===========================================================================
# Avaliar
# ===========================================================================


def test_humano_igual_ao_juiz_da_kappa_um(banco_comparado, tmp_path):
    banco, comparacao_id = banco_comparado
    caminho, _ = exportar(banco, tmp_path / "cal.csv", tamanho=30, semente=0)
    juiz = notas_do_juiz(banco, comparacao_id)
    preencher(caminho, lambda i: str(juiz[i]))

    resultado = avaliar(banco, caminho)

    assert resultado.kappa == pytest.approx(1.0)
    assert resultado.concordancia_exata == 100
    assert resultado.discordancias == []
    assert sum(map(sum, resultado.matriz)) == resultado.pares == 30
    assert "quase perfeita" in resultado.interpretacao


def test_humano_mais_exigente_baixa_o_kappa_e_aparece_nas_discordancias(banco_comparado, tmp_path):
    banco, comparacao_id = banco_comparado
    caminho, _ = exportar(banco, tmp_path / "cal.csv", tamanho=30, semente=0)
    juiz = notas_do_juiz(banco, comparacao_id)
    preencher(caminho, lambda i: str(max(1, juiz[i] - 2)))  # você dá sempre 2 pontos a menos

    resultado = avaliar(banco, caminho)

    assert resultado.kappa < 0.8
    assert resultado.discordancias
    assert resultado.discordancias[0].distancia == 2
    distancias = [d.distancia for d in resultado.discordancias]
    assert distancias == sorted(distancias, reverse=True)  # maiores primeiro
    assert len(resultado.discordancias) <= 5
    matriz = resultado.matriz
    acima_da_diagonal = sum(matriz[l][c] for l in range(5) for c in range(5) if c < l)
    assert acima_da_diagonal > 0  # linha = juiz > coluna = você: juiz "bonzinho"


def test_linhas_sem_nota_sao_ignoradas(banco_comparado, tmp_path):
    banco, comparacao_id = banco_comparado
    caminho, _ = exportar(banco, tmp_path / "cal.csv", tamanho=10, semente=0)
    juiz = notas_do_juiz(banco, comparacao_id)
    ids = [int(linha["id_execucao"]) for linha in ler_csv(caminho)]
    preencher(caminho, lambda i: str(juiz[i]) if i in ids[:6] else "")

    resultado = avaliar(banco, caminho)
    assert (resultado.pares, resultado.ignoradas) == (6, 4)


def test_csv_salvo_com_virgula_tambem_funciona(banco_comparado, tmp_path):
    """Ex.: salvo pelo Google Sheets."""
    banco, comparacao_id = banco_comparado
    caminho, _ = exportar(banco, tmp_path / "cal.csv", tamanho=5, semente=0)
    juiz = notas_do_juiz(banco, comparacao_id)
    preencher(caminho, lambda i: str(juiz[i]), separador=",")
    assert avaliar(banco, caminho).pares == 5


@pytest.mark.parametrize("bruto, esperado", [("4", 4), ("4.0", 4), ("4,0", 4), (" 5 ", 5)])
def test_formatos_de_nota_aceitos(tmp_path, bruto, esperado):
    caminho = tmp_path / "c.csv"
    caminho.write_text(f"id_execucao;nota_humana\n1;{bruto}\n", encoding=CODIFICACAO)
    assert ler_notas_humanas(caminho) == ({1: esperado}, 0)


@pytest.mark.parametrize("bruto", ["6", "0", "3.5", "boa"])
def test_nota_invalida_mostra_a_linha(tmp_path, bruto):
    caminho = tmp_path / "c.csv"
    caminho.write_text(f"id_execucao;nota_humana\n1;4\n2;{bruto}\n", encoding=CODIFICACAO)
    with pytest.raises(ErroCalibracao, match="linha 3"):
        ler_notas_humanas(caminho)


def test_erros_de_leitura(tmp_path, banco_comparado):
    banco, _ = banco_comparado
    with pytest.raises(ErroCalibracao, match="Não achei"):
        ler_notas_humanas(tmp_path / "nada.csv")

    sem_coluna = tmp_path / "sem_coluna.csv"
    sem_coluna.write_text("id_execucao;nota\n1;4\n", encoding=CODIFICACAO)
    with pytest.raises(ErroCalibracao, match="nota_humana"):
        ler_notas_humanas(sem_coluna)

    id_ruim = tmp_path / "id_ruim.csv"
    id_ruim.write_text("id_execucao;nota_humana\nabc;4\n", encoding=CODIFICACAO)
    with pytest.raises(ErroCalibracao, match="id_execucao"):
        ler_notas_humanas(id_ruim)

    vazio = tmp_path / "vazio.csv"
    vazio.write_text("id_execucao;nota_humana\n1;\n", encoding=CODIFICACAO)
    with pytest.raises(ErroCalibracao, match="Nenhuma nota humana"):
        avaliar(banco, vazio)

    de_outro_banco = tmp_path / "outro.csv"
    de_outro_banco.write_text("id_execucao;nota_humana\n99999;4\n", encoding=CODIFICACAO)
    with pytest.raises(ErroCalibracao, match="não encontradas"):
        avaliar(banco, de_outro_banco)


# ===========================================================================
# Relatório
# ===========================================================================


def texto_calibracao(resultado):
    console = Console(record=True, width=200, color_system=None)
    mostrar_calibracao(resultado, console)
    return console.export_text()


def test_relatorio_de_calibracao(banco_comparado, tmp_path):
    banco, comparacao_id = banco_comparado
    caminho, _ = exportar(banco, tmp_path / "cal.csv", tamanho=10, semente=0)
    juiz = notas_do_juiz(banco, comparacao_id)
    ids = [int(linha["id_execucao"]) for linha in ler_csv(caminho)]
    preencher(caminho, lambda i: str(max(1, juiz[i] - 2)) if i in ids[:8] else "")

    texto = texto_calibracao(avaliar(banco, caminho))

    assert "kappa ponderado" in texto
    assert "Matriz de confusão (linha = juiz, coluna = você)" in texto
    assert "Maiores discordâncias" in texto
    assert "2 linha(s) sem nota humana" in texto
    assert "Só 8 pares" in texto  # aviso de amostra pequena


def test_relatorio_sem_discordancias(banco_comparado, tmp_path):
    banco, comparacao_id = banco_comparado
    caminho, _ = exportar(banco, tmp_path / "cal.csv", tamanho=30, semente=0)
    juiz = notas_do_juiz(banco, comparacao_id)
    preencher(caminho, lambda i: str(juiz[i]))

    texto = texto_calibracao(avaliar(banco, caminho))
    assert "Nenhuma discordância" in texto
    assert "Só" not in texto  # 30 pares: sem aviso


def test_relatorio_kappa_indefinido(tmp_path):
    """Todo mundo deu 4: não há variação e o kappa não tem sentido."""
    with Armazenamento(tmp_path / "b.db") as banco:
        comparacao_id = banco.criar_comparacao("a", "b", "x", "y", "fake/vendedor", "fake/juiz")
        banco.salvar_execucoes(comparacao_id, [Execucao("c1", "A", r, resposta="oi", nota=4) for r in (1, 2, 3)])
        caminho, _ = exportar(banco, tmp_path / "c.csv", comparacao_id=comparacao_id)
        preencher(caminho, lambda i: "4")
        texto = texto_calibracao(avaliar(banco, caminho))
    assert "indefinido" in texto

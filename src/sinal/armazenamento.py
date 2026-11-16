"""Persistência em SQLite: comparações e execuções.

Não conhece LLM, juiz nem estatística: só recebe dados prontos e grava/consulta.
Seguro para várias threads gravando ao mesmo tempo (um cadeado serializa as escritas).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import astuple, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSAO_ESQUEMA = 1

ESQUEMA = """
CREATE TABLE IF NOT EXISTS comparacoes (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em              TEXT    NOT NULL,
    status                 TEXT    NOT NULL DEFAULT 'em_andamento'
                                   CHECK (status IN ('em_andamento', 'concluida', 'falhou')),
    prompt_a               TEXT    NOT NULL,
    prompt_b               TEXT    NOT NULL,
    conteudo_a             TEXT    NOT NULL,
    conteudo_b             TEXT    NOT NULL,
    hash_a                 TEXT    NOT NULL,
    hash_b                 TEXT    NOT NULL,
    modelo_alvo            TEXT    NOT NULL,
    modelo_juiz            TEXT    NOT NULL,
    veredito               TEXT,
    diferenca              REAL,
    ic_inferior            REAL,
    ic_superior            REAL,
    p_valor                REAL,
    chamadas_feitas        INTEGER,
    chamadas_economizadas  INTEGER,
    erro                   TEXT,
    config_json            TEXT,
    resultado_json         TEXT
);

CREATE TABLE IF NOT EXISTS execucoes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    comparacao_id    INTEGER NOT NULL REFERENCES comparacoes (id) ON DELETE CASCADE,
    caso_id          TEXT    NOT NULL,
    versao           TEXT    NOT NULL CHECK (versao IN ('A', 'B')),
    rodada           INTEGER NOT NULL CHECK (rodada >= 1),
    resposta         TEXT,
    nota             INTEGER CHECK (nota IS NULL OR nota BETWEEN 1 AND 5),
    justificativa    TEXT,
    metodo_parse     TEXT,
    erro             TEXT,
    tokens_entrada   INTEGER NOT NULL DEFAULT 0,
    tokens_saida     INTEGER NOT NULL DEFAULT 0,
    latencia_ms      REAL    NOT NULL DEFAULT 0,
    custo_usd        REAL    NOT NULL DEFAULT 0,
    tokens_juiz      INTEGER NOT NULL DEFAULT 0,
    custo_juiz_usd   REAL    NOT NULL DEFAULT 0,
    criado_em        TEXT    NOT NULL,
    UNIQUE (comparacao_id, caso_id, versao, rodada)
);

CREATE INDEX IF NOT EXISTS idx_execucoes_comparacao ON execucoes (comparacao_id, caso_id, versao);
"""


@dataclass(frozen=True)
class Execucao:
    """Uma linha da tabela `execucoes`: uma rodada de um prompt num caso, já avaliada (ou com erro)."""

    caso_id: str
    versao: str  # "A" ou "B"
    rodada: int
    resposta: str | None = None
    nota: int | None = None
    justificativa: str | None = None
    metodo_parse: str | None = None
    erro: str | None = None
    tokens_entrada: int = 0  # do modelo alvo
    tokens_saida: int = 0
    latencia_ms: float = 0.0
    custo_usd: float = 0.0
    tokens_juiz: int = 0  # entrada + saída do juiz
    custo_juiz_usd: float = 0.0


COLUNAS_EXECUCAO = [f.name for f in fields(Execucao)]


def resumo_hash(texto: str) -> str:
    """Impressão digital curta do texto: muda se o prompt mudar, mesmo com o mesmo nome de arquivo."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Armazenamento:
    """Acesso ao banco do Sinal. Use com `with Armazenamento(caminho) as banco:`."""

    def __init__(self, caminho: str | Path):
        caminho = str(caminho)
        if caminho != ":memory:":
            Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: deixa várias threads usarem a mesma conexão.
        # A segurança fica por conta do cadeado abaixo.
        self._conexao = sqlite3.connect(caminho, check_same_thread=False)
        self._conexao.row_factory = sqlite3.Row  # linhas acessíveis por nome de coluna
        self._conexao.execute("PRAGMA foreign_keys = ON")  # no SQLite, a FK vem DESLIGADA por padrão
        self._cadeado = threading.Lock()
        self._criar_tabelas()

    # --- ciclo de vida da conexão ---------------------------------------------

    def __enter__(self) -> Armazenamento:
        return self

    def __exit__(self, *_: Any) -> None:
        self.fechar()

    def fechar(self) -> None:
        self._conexao.close()

    def _criar_tabelas(self) -> None:
        with self._cadeado, self._conexao:
            self._conexao.executescript(ESQUEMA)
            self._conexao.execute(f"PRAGMA user_version = {VERSAO_ESQUEMA}")

    def _escrever(self, sql: str, parametros: tuple | list = ()) -> sqlite3.Cursor:
        """Executa uma escrita dentro de uma transação, uma thread por vez."""
        with self._cadeado, self._conexao:  # `with conexao` = commit no fim, rollback se der erro
            return self._conexao.execute(sql, parametros)

    def _ler(self, sql: str, parametros: tuple | list = ()) -> list[dict[str, Any]]:
        with self._cadeado:
            return [dict(linha) for linha in self._conexao.execute(sql, parametros).fetchall()]

    # --- comparações ----------------------------------------------------------

    def criar_comparacao(
        self,
        prompt_a: str,
        prompt_b: str,
        conteudo_a: str,
        conteudo_b: str,
        modelo_alvo: str,
        modelo_juiz: str,
        config: dict[str, Any] | None = None,
    ) -> int:
        """Registra uma comparação nova (status 'em_andamento') e devolve o id dela."""
        cursor = self._escrever(
            """
            INSERT INTO comparacoes
                (criado_em, prompt_a, prompt_b, conteudo_a, conteudo_b, hash_a, hash_b,
                 modelo_alvo, modelo_juiz, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _agora(),
                prompt_a,
                prompt_b,
                conteudo_a,
                conteudo_b,
                resumo_hash(conteudo_a),
                resumo_hash(conteudo_b),
                modelo_alvo,
                modelo_juiz,
                json.dumps(config or {}, ensure_ascii=False, default=str),
            ),
        )
        return int(cursor.lastrowid)

    def finalizar_comparacao(
        self,
        comparacao_id: int,
        *,
        veredito: str,
        diferenca: float,
        ic_inferior: float,
        ic_superior: float,
        p_valor: float,
        chamadas_feitas: int,
        chamadas_economizadas: int,
        resultado: dict[str, Any] | None = None,
    ) -> None:
        """Grava o veredito final e muda o status para 'concluida'."""
        self._escrever(
            """
            UPDATE comparacoes
               SET status = 'concluida', veredito = ?, diferenca = ?, ic_inferior = ?, ic_superior = ?,
                   p_valor = ?, chamadas_feitas = ?, chamadas_economizadas = ?, resultado_json = ?
             WHERE id = ?
            """,
            (
                veredito,
                diferenca,
                ic_inferior,
                ic_superior,
                p_valor,
                chamadas_feitas,
                chamadas_economizadas,
                json.dumps(resultado or {}, ensure_ascii=False, default=str),
                comparacao_id,
            ),
        )

    def marcar_falha(self, comparacao_id: int, erro: str) -> None:
        """Marca a comparação como 'falhou' (ex.: todas as chamadas deram erro)."""
        self._escrever("UPDATE comparacoes SET status = 'falhou', erro = ? WHERE id = ?", (erro, comparacao_id))

    def buscar_comparacao(self, comparacao_id: int) -> dict[str, Any] | None:
        """Devolve a comparação com config e resultado já convertidos de JSON, ou None."""
        linhas = self._ler("SELECT * FROM comparacoes WHERE id = ?", (comparacao_id,))
        if not linhas:
            return None
        comparacao = linhas[0]
        for coluna in ("config_json", "resultado_json"):
            texto = comparacao.pop(coluna)
            comparacao[coluna.removesuffix("_json")] = json.loads(texto) if texto else None
        return comparacao

    def listar_comparacoes(self, limite: int = 20) -> list[dict[str, Any]]:
        """Histórico: as comparações mais recentes primeiro, com totais de execuções, erros e custo."""
        return self._ler(
            """
            SELECT c.id, c.criado_em, c.status, c.prompt_a, c.prompt_b, c.hash_a, c.hash_b,
                   c.modelo_alvo, c.modelo_juiz, c.veredito, c.diferenca, c.ic_inferior, c.ic_superior,
                   c.p_valor, c.chamadas_feitas, c.chamadas_economizadas,
                   COUNT(e.id)                                      AS total_execucoes,
                   COALESCE(SUM(e.erro IS NOT NULL), 0)             AS total_erros,
                   COALESCE(SUM(e.custo_usd + e.custo_juiz_usd), 0) AS custo_total_usd
              FROM comparacoes c
              LEFT JOIN execucoes e ON e.comparacao_id = c.id
             GROUP BY c.id
             ORDER BY c.id DESC
             LIMIT ?
            """,
            (limite,),
        )

    # --- execuções ------------------------------------------------------------

    def salvar_execucao(self, comparacao_id: int, execucao: Execucao) -> None:
        """Grava uma execução. Pode ser chamada de várias threads ao mesmo tempo."""
        self.salvar_execucoes(comparacao_id, [execucao])

    def salvar_execucoes(self, comparacao_id: int, execucoes: list[Execucao]) -> None:
        """Grava várias execuções numa transação só (tudo ou nada)."""
        colunas = ", ".join(["comparacao_id", *COLUNAS_EXECUCAO, "criado_em"])
        marcadores = ", ".join("?" * (len(COLUNAS_EXECUCAO) + 2))
        agora = _agora()
        linhas = [(comparacao_id, *astuple(e), agora) for e in execucoes]
        with self._cadeado, self._conexao:
            self._conexao.executemany(f"INSERT INTO execucoes ({colunas}) VALUES ({marcadores})", linhas)

    def ler_execucoes(
        self,
        comparacao_id: int,
        *,
        caso_id: str | None = None,
        versao: str | None = None,
        apenas_erros: bool = False,
        limite: int | None = None,
    ) -> list[dict[str, Any]]:
        """Lê as execuções de uma comparação, com filtros opcionais."""
        condicoes = ["comparacao_id = ?"]
        parametros: list[Any] = [comparacao_id]
        if caso_id is not None:
            condicoes.append("caso_id = ?")
            parametros.append(caso_id)
        if versao is not None:
            condicoes.append("versao = ?")
            parametros.append(versao)
        if apenas_erros:
            condicoes.append("erro IS NOT NULL")
        sql = f"SELECT * FROM execucoes WHERE {' AND '.join(condicoes)} ORDER BY caso_id, versao, rodada"
        if limite is not None:
            sql += " LIMIT ?"
            parametros.append(limite)
        return self._ler(sql, parametros)

    def notas_por_caso(self, comparacao_id: int) -> dict[str, tuple[list[int], list[int]]]:
        """Notas válidas agrupadas: {caso_id: ([notas A], [notas B])}. Execuções com erro ficam de fora."""
        resultado: dict[str, tuple[list[int], list[int]]] = {}
        linhas = self._ler(
            """
            SELECT caso_id, versao, nota FROM execucoes
             WHERE comparacao_id = ? AND nota IS NOT NULL
             ORDER BY caso_id, versao, rodada
            """,
            (comparacao_id,),
        )
        for linha in linhas:
            notas_a, notas_b = resultado.setdefault(linha["caso_id"], ([], []))
            (notas_a if linha["versao"] == "A" else notas_b).append(linha["nota"])
        return resultado

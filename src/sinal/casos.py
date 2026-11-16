"""Casos de teste e prompts.

Lê os casos do YAML, lê o arquivo do prompt e monta as mensagens que vão pro modelo alvo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MARCADOR_ENTRADA = "{entrada}"

CAMPOS_OBRIGATORIOS = ("id", "entrada", "criterio")
CAMPOS_OPCIONAIS = ("referencia",)


class ErroCasos(Exception):
    """Problema no arquivo de casos ou no arquivo de prompt."""


@dataclass(frozen=True)
class Caso:
    """Um caso de teste: o que o cliente diz e como a resposta deve ser avaliada."""

    id: str
    entrada: str  # vai pro modelo alvo
    criterio: str  # só o JUIZ vê
    referencia: str | None = None  # só o JUIZ vê (opcional)


# ---------------------------------------------------------------------------
# Casos
# ---------------------------------------------------------------------------


def carregar_casos(caminho: str | Path) -> list[Caso]:
    """Lê o YAML de casos (uma lista na raiz) e devolve os casos validados.

    Raises:
        ErroCasos: arquivo inexistente, YAML inválido ou casos com problema.
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ErroCasos(f"Não achei o arquivo de casos '{caminho}'.")
    try:
        dados = yaml.safe_load(caminho.read_text(encoding="utf-8"))
    except yaml.YAMLError as erro:
        raise ErroCasos(f"O arquivo '{caminho}' não é um YAML válido:\n{erro}") from erro

    return montar_casos(dados, origem=str(caminho))


def montar_casos(dados: Any, origem: str = "casos") -> list[Caso]:
    """Valida a lista lida do YAML e transforma cada item num Caso."""
    if not isinstance(dados, list) or not dados:
        raise ErroCasos(f"'{origem}' deve conter uma lista com pelo menos um caso (itens começando com `- id: ...`).")

    problemas: list[str] = []
    casos: list[Caso] = []
    ids_vistos: set[str] = set()

    for posicao, item in enumerate(dados, start=1):
        # Antes de ter o id, a gente se refere ao caso pela posição.
        rotulo = f"caso #{posicao}"
        if not isinstance(item, dict):
            problemas.append(f"{rotulo}: deve ser um mapa com id, entrada e criterio.")
            continue

        # O YAML lê `id: 7` como número; aceitamos e convertemos pra texto.
        if isinstance(item.get("id"), int) and not isinstance(item.get("id"), bool):
            item = {**item, "id": str(item["id"])}
        if isinstance(item.get("id"), str) and item["id"].strip():
            rotulo = f"caso '{item['id']}'"

        desconhecidas = set(item) - set(CAMPOS_OBRIGATORIOS) - set(CAMPOS_OPCIONAIS)
        if desconhecidas:
            problemas.append(f"{rotulo}: chave(s) desconhecida(s): {', '.join(sorted(desconhecidas))}.")

        faltando = [c for c in CAMPOS_OBRIGATORIOS if not _texto_preenchido(item.get(c))]
        if faltando:
            problemas.append(f"{rotulo}: campo(s) obrigatório(s) vazio(s) ou ausente(s): {', '.join(faltando)}.")

        referencia = item.get("referencia")
        if referencia is not None and not _texto_preenchido(referencia):
            problemas.append(f"{rotulo}: 'referencia' deve ser um texto (ou ser omitida).")

        if desconhecidas or faltando:
            continue

        caso_id = item["id"].strip()
        if caso_id in ids_vistos:
            problemas.append(f"{rotulo}: id repetido.")
            continue
        ids_vistos.add(caso_id)

        casos.append(
            Caso(
                id=caso_id,
                entrada=item["entrada"].strip(),
                criterio=item["criterio"].strip(),
                referencia=referencia.strip() if isinstance(referencia, str) else None,
            )
        )

    if problemas:
        raise ErroCasos(f"Problemas em '{origem}':\n  - " + "\n  - ".join(problemas))
    return casos


def _texto_preenchido(valor: Any) -> bool:
    return isinstance(valor, str) and valor.strip() != ""


# ---------------------------------------------------------------------------
# Prompts e mensagens
# ---------------------------------------------------------------------------


def carregar_prompt(caminho: str | Path) -> str:
    """Lê o arquivo de prompt (texto puro)."""
    caminho = Path(caminho)
    if not caminho.is_file():
        raise ErroCasos(f"Não achei o arquivo de prompt '{caminho}'.")
    texto = caminho.read_text(encoding="utf-8").strip()
    if not texto:
        raise ErroCasos(f"O arquivo de prompt '{caminho}' está vazio.")
    return texto


def montar_mensagens(prompt: str, caso: Caso) -> list[dict[str, str]]:
    """Monta as mensagens no formato de chat que o LLM espera.

    - Prompt com `{entrada}`: substitui e manda UMA mensagem de usuário.
    - Sem `{entrada}`: o prompt vira `system` e a entrada vira `user`.

    O critério e a referência NÃO entram aqui: são gabarito, só o juiz vê.
    """
    if MARCADOR_ENTRADA in prompt:
        # replace (e não .format) porque o prompt pode ter outras chaves, ex.: um JSON de exemplo.
        return [{"role": "user", "content": prompt.replace(MARCADOR_ENTRADA, caso.entrada)}]
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": caso.entrada},
    ]

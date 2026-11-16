"""Agente de chat que ajuda a iterar prompts usando o Sinal como ferramenta (tool calling).

O LLM só DESCREVE as chamadas (nome + argumentos). Quem executa é este módulo,
com travas: confirmação antes de gastar dinheiro, nunca sobrescrever arquivo,
nomes validados e erros devolvidos ao LLM em vez de derrubar o chat.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from sinal.armazenamento import Armazenamento, resumo_hash
from sinal.casos import carregar_casos, carregar_prompt
from sinal.comparador import CHAMADAS_POR_EXECUCAO, comparar
from sinal.config import Config
from sinal.llm import ChamadaFerramenta, chamar_com_ferramentas, is_fake

MAXIMO_PASSOS = 8  # voltas do laço por mensagem sua: evita loop infinito (e conta infinita)
MAXIMO_EXECUCOES = 50
TAMANHO_MAXIMO_TEXTO = 500
PADRAO_NOME_PROMPT = re.compile(r"^[A-Za-z0-9_-]{1,64}\.(txt|md)$")

PROMPT_SISTEMA = """\
Você é o assistente do Sinal, uma ferramenta de testes de regressão estatísticos para prompts.
Você ajuda um Analista de Prompt a melhorar prompts com evidência, não com impressão.

Regras:
- Nunca afirme que um prompt melhorou sem um veredito global MELHOROU do Sinal. Se o veredito for \
INCONCLUSIVO ou EQUIVALENTE, diga claramente que não dá pra afirmar melhora. Se for PIOROU, diga que piorou.
- Explique os resultados em linguagem simples: a diferença média na escala de 1 a 5, o intervalo \
de confiança (a faixa onde a diferença real provavelmente está) e o p-valor, sem jargão desnecessário.
- O veredito por caso é só uma pista; a decisão vale pelo veredito global.
- Ao sugerir uma mudança num prompt, salve como NOVA versão com salvar_prompt (nunca sobrescreva) \
e compare contra a versão atual com comparar_prompts.
- Para entender por que um prompt piorou, leia as execuções (ler_execucoes) e as justificativas \
do juiz antes de opinar.
- comparar_prompts custa dinheiro e pede confirmação ao usuário; se ele recusar, respeite.
- Não invente nomes de arquivos ou de casos: use listar_prompts e listar_casos.
- Responda em português do Brasil, de forma direta."""


def _ferramenta(nome: str, descricao: str, propriedades: dict | None = None, obrigatorios: list[str] | None = None):
    return {
        "type": "function",
        "function": {
            "name": nome,
            "description": descricao,
            "parameters": {
                "type": "object",
                "properties": propriedades or {},
                "required": obrigatorios or [],
                "additionalProperties": False,
            },
        },
    }


ESQUEMAS_FERRAMENTAS = [
    _ferramenta("listar_prompts", "Lista os arquivos de prompt da pasta de prompts do projeto."),
    _ferramenta(
        "ler_prompt",
        "Lê o conteúdo de um arquivo de prompt.",
        {"nome": {"type": "string", "description": "Nome do arquivo, ex.: v2.txt"}},
        ["nome"],
    ),
    _ferramenta("listar_casos", "Lista os casos de teste (id, entrada do cliente e critério de avaliação)."),
    _ferramenta(
        "comparar_prompts",
        "Roda o Sinal: compara o prompt B contra o A e devolve o veredito estatístico. "
        "Custa chamadas de LLM; o usuário precisa confirmar.",
        {
            "prompt_a": {"type": "string", "description": "Prompt atual (A), ex.: v2.txt"},
            "prompt_b": {"type": "string", "description": "Prompt novo (B), ex.: v3.txt"},
        },
        ["prompt_a", "prompt_b"],
    ),
    _ferramenta(
        "ler_execucoes",
        "Lê as respostas, notas e justificativas do juiz de uma comparação, pra diagnosticar resultados.",
        {
            "comparacao_id": {"type": "integer"},
            "caso_id": {"type": "string", "description": "Filtra por um caso (opcional)."},
            "versao": {"type": "string", "enum": ["A", "B"], "description": "Filtra por versão (opcional)."},
            "apenas_erros": {"type": "boolean"},
            "limite": {"type": "integer", "description": f"Máximo de linhas (até {MAXIMO_EXECUCOES})."},
        },
        ["comparacao_id"],
    ),
    _ferramenta(
        "salvar_prompt",
        "Cria uma NOVA versão de prompt na pasta de prompts. Nunca sobrescreve um arquivo existente.",
        {
            "nome": {"type": "string", "description": "Nome novo, ex.: v3.txt (letras, números, - e _)."},
            "conteudo": {"type": "string", "description": "Texto completo do prompt."},
        },
        ["nome", "conteudo"],
    ),
    _ferramenta(
        "historico",
        "Lista as comparações anteriores, da mais recente pra mais antiga.",
        {"limite": {"type": "integer"}},
    ),
]


class ErroFerramenta(Exception):
    """Erro "esperado" de uma ferramenta: vira {"erro": ...} pro LLM se corrigir."""


def validar_nome_prompt(nome: str) -> str:
    """Aceita só nomes simples como `v3.txt` ou `v3-objecoes.md`. Recusa caminhos (../, /, \\)."""
    if not isinstance(nome, str) or not PADRAO_NOME_PROMPT.fullmatch(nome):
        raise ErroFerramenta(
            f"Nome de prompt inválido: {nome!r}. Use só letras, números, '-' e '_', terminando em .txt ou .md "
            "(ex.: v3.txt)."
        )
    return nome


class FerramentasSinal:
    """As ferramentas que o agente pode pedir. Cada método devolve dados simples (viram JSON)."""

    def __init__(self, config: Config, confirmar: Callable[[str], bool]):
        self.config = config
        self.confirmar = confirmar  # o CLI pergunta "sim/não" pro usuário
        self.pasta_prompts = config.caminhos.prompts

    # --- despacho -------------------------------------------------------------

    def executar(self, chamada: ChamadaFerramenta) -> str:
        """Roda a ferramenta pedida e devolve o resultado em JSON. Erros também viram JSON."""
        metodo = getattr(self, f"_ferramenta_{chamada.nome}", None)
        if metodo is None:
            return _json({"erro": f"Ferramenta desconhecida: {chamada.nome!r}."})
        try:
            argumentos = json.loads(chamada.argumentos_json or "{}")
            if not isinstance(argumentos, dict):
                raise ValueError("os argumentos devem ser um objeto JSON")
        except ValueError as erro:
            return _json({"erro": f"Argumentos inválidos (JSON): {erro}"})
        try:
            return _json(metodo(**argumentos))
        except TypeError as erro:  # argumento faltando ou sobrando
            return _json({"erro": f"Argumentos errados para {chamada.nome}: {erro}"})
        except ErroFerramenta as erro:
            return _json({"erro": str(erro)})
        except Exception as erro:  # erro do Sinal (casos, comparação...): o LLM vê e explica
            return _json({"erro": f"{type(erro).__name__}: {erro}"})

    # --- ferramentas ----------------------------------------------------------

    def _ferramenta_listar_prompts(self) -> dict:
        if not self.pasta_prompts.is_dir():
            return {"prompts": []}
        arquivos = sorted(p for p in self.pasta_prompts.iterdir() if p.suffix in (".txt", ".md") and p.is_file())
        return {
            "prompts": [
                {"nome": p.name, "linhas": len(texto.splitlines()), "hash": resumo_hash(texto)}
                for p in arquivos
                for texto in [p.read_text(encoding="utf-8")]
            ]
        }

    def _ferramenta_ler_prompt(self, nome: str) -> dict:
        caminho = self._caminho_prompt(nome)
        if not caminho.is_file():
            raise ErroFerramenta(f"O prompt {nome!r} não existe. Use listar_prompts pra ver os disponíveis.")
        return {"nome": nome, "conteudo": caminho.read_text(encoding="utf-8")}

    def _ferramenta_listar_casos(self) -> dict:
        casos = carregar_casos(self.config.caminhos.casos)
        return {
            "casos": [
                {"id": c.id, "entrada": c.entrada, "criterio": c.criterio, "tem_referencia": c.referencia is not None}
                for c in casos
            ]
        }

    def _ferramenta_comparar_prompts(self, prompt_a: str, prompt_b: str) -> dict:
        caminho_a, caminho_b = self._caminho_prompt(prompt_a), self._caminho_prompt(prompt_b)
        for nome, caminho in ((prompt_a, caminho_a), (prompt_b, caminho_b)):
            if not caminho.is_file():
                raise ErroFerramenta(f"O prompt {nome!r} não existe.")

        if not self.confirmar(self._mensagem_de_custo(prompt_a, prompt_b)):
            return {"cancelado": True, "motivo": "O usuário não confirmou a comparação."}

        resultado = comparar(self.config, caminho_a, caminho_b)
        i = resultado.intervalo
        return {
            "comparacao_id": resultado.comparacao_id,
            "veredito_global": resultado.veredito.value,
            "diferenca_media_b_menos_a": round(i.diferenca, 2),
            "intervalo_de_confianca": [round(i.inferior, 2), round(i.superior, 2)],
            "confianca": self.config.estatistica.confianca,
            "p_valor": round(resultado.p_valor, 4),
            "casos": [
                {
                    "id": c.caso_id,
                    "veredito_indicativo": c.veredito.value,
                    "media_a": round(sum(c.notas_a) / len(c.notas_a), 2) if c.notas_a else None,
                    "media_b": round(sum(c.notas_b) / len(c.notas_b), 2) if c.notas_b else None,
                    "erros": c.erros,
                }
                for c in resultado.casos
            ],
            "chamadas_feitas": resultado.chamadas_feitas,
            "custo_usd": round(resultado.custo_total_usd, 4),
        }

    def _ferramenta_ler_execucoes(
        self,
        comparacao_id: int,
        caso_id: str | None = None,
        versao: str | None = None,
        apenas_erros: bool = False,
        limite: int = 20,
    ) -> dict:
        limite = max(1, min(int(limite), MAXIMO_EXECUCOES))  # o LLM pode pedir 10.000: a gente corta
        with self._banco() as banco:
            if banco.buscar_comparacao(comparacao_id) is None:
                raise ErroFerramenta(f"A comparação {comparacao_id} não existe. Use historico pra ver as existentes.")
            linhas = banco.ler_execucoes(
                comparacao_id, caso_id=caso_id, versao=versao, apenas_erros=apenas_erros, limite=limite
            )
        return {
            "execucoes": [
                {
                    "caso_id": e["caso_id"],
                    "versao": e["versao"],
                    "rodada": e["rodada"],
                    "nota": e["nota"],
                    "justificativa_do_juiz": e["justificativa"],
                    "resposta": _cortar(e["resposta"]),
                    "erro": e["erro"],
                }
                for e in linhas
            ]
        }

    def _ferramenta_salvar_prompt(self, nome: str, conteudo: str) -> dict:
        caminho = self._caminho_prompt(nome)
        if not isinstance(conteudo, str) or not conteudo.strip():
            raise ErroFerramenta("O conteúdo do prompt está vazio.")
        caminho.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Modo "x": o sistema operacional só cria se o arquivo NÃO existir (atômico).
            with caminho.open("x", encoding="utf-8") as arquivo:
                arquivo.write(conteudo.strip() + "\n")
        except FileExistsError:
            raise ErroFerramenta(f"O prompt {nome!r} já existe e não será sobrescrito. Escolha outro nome.") from None
        return {"salvo": nome, "linhas": len(conteudo.strip().splitlines())}

    def _ferramenta_historico(self, limite: int = 10) -> dict:
        if not self.config.caminhos.banco.exists():
            return {"comparacoes": []}
        with self._banco() as banco:
            linhas = banco.listar_comparacoes(max(1, min(int(limite), 50)))
        campos = ("id", "criado_em", "status", "prompt_a", "prompt_b", "veredito", "diferenca",
                  "ic_inferior", "ic_superior", "p_valor", "total_execucoes", "total_erros", "custo_total_usd")
        return {"comparacoes": [{campo: linha[campo] for campo in campos} for linha in linhas]}

    # --- auxiliares -----------------------------------------------------------

    def _caminho_prompt(self, nome: str) -> Path:
        caminho = (self.pasta_prompts / validar_nome_prompt(nome)).resolve()
        if caminho.parent != self.pasta_prompts.resolve():  # cinto e suspensório contra ../
            raise ErroFerramenta(f"O prompt {nome!r} fica fora da pasta de prompts.")
        return caminho

    def _banco(self) -> Armazenamento:
        return Armazenamento(self.config.caminhos.banco)

    def _mensagem_de_custo(self, prompt_a: str, prompt_b: str) -> str:
        total_casos = len(carregar_casos(self.config.caminhos.casos))
        a = self.config.amostragem
        por_rodada = total_casos * 2 * CHAMADAS_POR_EXECUCAO
        modelos = self.config.modelos
        gratis = is_fake(modelos.alvo) and is_fake(modelos.juiz)
        return (
            f"O agente quer comparar {prompt_a} (A) contra {prompt_b} (B): {total_casos} casos, "
            f"entre {por_rodada * a.rodadas_iniciais} e {por_rodada * a.rodadas_maximas} chamadas de LLM "
            f"(alvo: {modelos.alvo}, juiz: {modelos.juiz})"
            + (" — modelos fake, custo zero." if gratis else ".")
            + " Confirma?"
        )


class Agente:
    """O laço do chat: LLM pede ferramentas → executamos → devolvemos → até ele responder em texto."""

    def __init__(
        self,
        config: Config,
        confirmar: Callable[[str], bool],
        *,
        maximo_passos: int = MAXIMO_PASSOS,
    ):
        self.modelo = config.modelos.agente
        self.ferramentas = FerramentasSinal(config, confirmar)
        self.maximo_passos = maximo_passos
        self.mensagens: list[dict[str, Any]] = [{"role": "system", "content": PROMPT_SISTEMA}]
        self.custo_total_usd = 0.0

    def responder(
        self, texto_usuario: str, ao_usar_ferramenta: Callable[[ChamadaFerramenta], None] | None = None
    ) -> str:
        self.mensagens.append({"role": "user", "content": texto_usuario})

        for _ in range(self.maximo_passos):
            resposta = chamar_com_ferramentas(self.modelo, self.mensagens, ESQUEMAS_FERRAMENTAS)
            if not resposta.ok:
                return f"Não consegui falar com o modelo do agente ({self.modelo}): {resposta.erro}"
            self.custo_total_usd += resposta.custo_usd
            self.mensagens.append(resposta.mensagem_assistente())

            if not resposta.chamadas:  # respondeu em texto: fim desta volta do chat
                return resposta.texto or ""

            for chamada in resposta.chamadas:
                if ao_usar_ferramenta:
                    ao_usar_ferramenta(chamada)
                self.mensagens.append(
                    {"role": "tool", "tool_call_id": chamada.id, "content": self.ferramentas.executar(chamada)}
                )

        return (
            f"Parei depois de {self.maximo_passos} passos sem uma resposta final, pra não gastar à toa. "
            "Tente reformular o pedido."
        )


def _json(dados: Any) -> str:
    return json.dumps(dados, ensure_ascii=False, default=str)


def _cortar(texto: str | None) -> str | None:
    if texto is None or len(texto) <= TAMANHO_MAXIMO_TEXTO:
        return texto
    return texto[:TAMANHO_MAXIMO_TEXTO] + "… [cortado]"

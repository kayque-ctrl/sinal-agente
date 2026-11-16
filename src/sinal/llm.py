from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass

PREFIXO_FAKE = "fake/"


@dataclass(frozen=True)
class RespostaLLM:
    """Resultado de uma chamada, com sucesso ou não."""

    texto: str | None
    tokens_entrada: int = 0
    tokens_saida: int = 0
    latencia_ms: float = 0.0
    custo_usd: float = 0.0
    erro: str | None = None

    @property
    def ok(self) -> bool:
        return self.erro is None


def is_fake(modelo: str) -> bool:
    return modelo.startswith(PREFIXO_FAKE)


def chamar(
    modelo: str,
    mensagens: list[dict[str, str]],
    temperatura: float | None = 0.0,
    *,
    gerador: random.Random | None = None,
) -> RespostaLLM:
    """Chama o modelo e devolve a resposta. Nunca levanta exceção.

    Args:
        modelo: nome no formato do LiteLLM (ex.: "gpt-4o-mini") ou "fake/...".
        mensagens: lista de {"role": ..., "content": ...}.
        temperatura: aleatoriedade da resposta (0 = mais determinístico).
            None = não envia o parâmetro (o provedor usa o padrão).
        gerador: só para o modo fake; permite sorteios reproduzíveis nos testes.
    """
    inicio = time.perf_counter()
    try:
        if is_fake(modelo):
            texto, tokens_entrada, tokens_saida, custo = _chamar_fake(modelo, mensagens, temperatura, gerador)
        else:
            texto, tokens_entrada, tokens_saida, custo = _chamar_litellm(modelo, mensagens, temperatura)
    except Exception as erro:  # qualquer falha vira registro, não derruba a rodada
        return RespostaLLM(
            texto=None,
            latencia_ms=_ms_desde(inicio),
            erro=f"{type(erro).__name__}: {erro}",
        )
    return RespostaLLM(
        texto=texto,
        tokens_entrada=tokens_entrada,
        tokens_saida=tokens_saida,
        latencia_ms=_ms_desde(inicio),
        custo_usd=custo,
    )


def _ms_desde(inicio: float) -> float:
    return (time.perf_counter() - inicio) * 1000


# ---------------------------------------------------------------------------
# Provedor real (LiteLLM)
# ---------------------------------------------------------------------------


def _chamar_litellm(
    modelo: str, mensagens: list[dict[str, str]], temperatura: float | None
) -> tuple[str, int, int, float]:
    # Import aqui dentro: o litellm demora pra carregar, e no modo fake ele nem é necessário.
    import litellm

    extras = {} if temperatura is None else {"temperature": temperatura}  # None: nem envia o parâmetro
    resposta = litellm.completion(
        model=modelo,
        messages=mensagens,
        num_retries=2,  # tenta de novo em falhas passageiras (rate limit, timeout)
        **extras,
    )
    texto = resposta.choices[0].message.content or ""
    uso = getattr(resposta, "usage", None)
    tokens_entrada = getattr(uso, "prompt_tokens", 0) or 0
    tokens_saida = getattr(uso, "completion_tokens", 0) or 0
    try:
        custo = float(litellm.completion_cost(completion_response=resposta) or 0.0)
    except Exception:
        custo = 0.0  # modelo sem preço na tabela do LiteLLM: não é motivo pra falhar
    return texto, tokens_entrada, tokens_saida, custo


# ---------------------------------------------------------------------------
# Simulador (modo fake)
# ---------------------------------------------------------------------------

BONUS_POR_INSTRUCAO = 0.25  # cada linha "- regra" no prompt melhora a resposta simulada
BONUS_MAXIMO = 1.5
DESVIO_MINIMO = 0.2  # ruído mesmo com temperatura 0
DESVIO_POR_TEMPERATURA = 0.6
DESVIO_JUIZ = 0.3
TEMPERATURA_PADRAO_FAKE = 1.0  # temperatura "do provedor" quando a config manda null

PADRAO_INSTRUCAO = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
PADRAO_QUALIDADE = re.compile(r"\[qualidade=(\d+(?:\.\d+)?)\]")

FALAS_VENDEDOR = {
    1: "Não sei te dizer. Dá uma olhada no site.",
    2: "Temos vários planos. Qualquer coisa me chama.",
    3: "Entendo. Nosso plano tem bastante coisa boa e acho que vale a pena pra você.",
    4: "Entendo sua preocupação! Me conta como vocês fazem a gestão hoje? "
    "Assim te mostro o plano que faz mais sentido.",
    5: "Faz todo sentido avaliar bem o investimento. Qual é o maior problema da sua gestão hoje? "
    "Com isso te mostro o retorno na prática e, se fizer sentido, agendo uma demonstração "
    "de 15 minutos ainda hoje. Pode ser?",
}


def _chamar_fake(
    modelo: str,
    mensagens: list[dict[str, str]],
    temperatura: float | None,
    gerador: random.Random | None,
) -> tuple[str, int, int, float]:
    gerador = gerador or random.Random()
    temperatura = TEMPERATURA_PADRAO_FAKE if temperatura is None else temperatura
    tipo = modelo[len(PREFIXO_FAKE):]

    if tipo == "erro":
        raise RuntimeError("falha simulada de API")
    if tipo == "juiz":
        texto = _juiz_fake(mensagens, gerador)
    else:  # qualquer outro fake/... se comporta como o vendedor
        texto = _vendedor_fake(mensagens, temperatura, gerador)

    tokens_entrada = sum(len(m["content"].split()) for m in mensagens)
    return texto, tokens_entrada, len(texto.split()), 0.0


def _vendedor_fake(mensagens: list[dict[str, str]], temperatura: float, gerador: random.Random) -> str:
    qualidade = qualidade_oculta(mensagens, temperatura, gerador)
    fala = FALAS_VENDEDOR[int(qualidade + 0.5)]  # arredonda pra escolher a fala do nível
    return f"{fala}\n\n[qualidade={qualidade:.2f}]"


def qualidade_oculta(mensagens: list[dict[str, str]], temperatura: float, gerador: random.Random) -> float:
    """Qualidade simulada da resposta, entre 1 e 5.

    = dificuldade do caso + bônus pelas instruções do prompt + ruído gaussiano.
    """
    base = dificuldade_do_caso(_trecho_do_caso(mensagens))
    instrucoes = sum(_contar_instrucoes(m["content"]) for m in mensagens)
    bonus = min(BONUS_MAXIMO, BONUS_POR_INSTRUCAO * instrucoes)
    desvio = DESVIO_MINIMO + DESVIO_POR_TEMPERATURA * temperatura
    return _limitar(base + bonus + gerador.gauss(0, desvio), 1.0, 5.0)


def dificuldade_do_caso(trecho: str) -> float:
    """Nota base do caso, entre 2.0 e 3.5, sempre igual para o mesmo texto.

    Usa sha256 e não hash() porque o hash() do Python muda a cada execução
    (é aleatorizado por segurança). Aqui precisamos do MESMO valor sempre.
    """
    resumo = hashlib.sha256(trecho.encode("utf-8")).digest()
    fracao = int.from_bytes(resumo[:4], "big") / 2**32  # número entre 0 e 1
    return 2.0 + 1.5 * fracao


def _trecho_do_caso(mensagens: list[dict[str, str]]) -> str:
    """O pedaço que identifica o caso: o último parágrafo da última mensagem.

    No modo system+user, é a própria entrada do cliente. No modo {entrada},
    costuma ser a linha onde a entrada foi colada (ex.: "Cliente: Achei caro.").
    Assim a dificuldade é a mesma para as versões A e B do mesmo caso.
    """
    paragrafos = [p.strip() for p in mensagens[-1]["content"].split("\n\n") if p.strip()]
    return paragrafos[-1] if paragrafos else ""


def _contar_instrucoes(texto: str) -> int:
    return sum(1 for linha in texto.splitlines() if PADRAO_INSTRUCAO.match(linha))


def _juiz_fake(mensagens: list[dict[str, str]], gerador: random.Random) -> str:
    """Lê o marcador [qualidade=X] e devolve a nota em JSON, justificativa antes da nota.

    Não sabe se a resposta é da versão A ou B: só enxerga o texto da resposta.
    """
    texto = "\n".join(m["content"] for m in mensagens)
    achados = PADRAO_QUALIDADE.findall(texto)
    if not achados:
        resultado = {"justificativa": "Simulação: resposta sem marcador de qualidade.", "nota": 3}
    else:
        qualidade = float(achados[-1])
        nota = int(_limitar(qualidade + gerador.gauss(0, DESVIO_JUIZ), 1.0, 5.0) + 0.5)
        resultado = {"justificativa": f"Simulação: qualidade oculta {qualidade:.2f}.", "nota": nota}
    return json.dumps(resultado, ensure_ascii=False)


def _limitar(valor: float, minimo: float, maximo: float) -> float:
    return max(minimo, min(maximo, valor))


# ---------------------------------------------------------------------------
# Chamada com ferramentas (tool calling) - usada pelo `sinal agente`
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChamadaFerramenta:
    """O LLM pediu pra rodar uma ferramenta. Quem executa (ou não) é o nosso código."""

    id: str
    nome: str
    argumentos_json: str  # o LLM manda os argumentos como TEXTO JSON; pode vir inválido


@dataclass(frozen=True)
class RespostaComFerramentas:
    texto: str | None
    chamadas: tuple[ChamadaFerramenta, ...] = ()
    tokens_entrada: int = 0
    tokens_saida: int = 0
    custo_usd: float = 0.0
    erro: str | None = None

    @property
    def ok(self) -> bool:
        return self.erro is None

    def mensagem_assistente(self) -> dict:
        """A resposta no formato que precisa voltar pro histórico (padrão OpenAI, que o LiteLLM traduz)."""
        mensagem: dict = {"role": "assistant", "content": self.texto or ""}
        if self.chamadas:
            mensagem["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.nome, "arguments": c.argumentos_json}}
                for c in self.chamadas
            ]
        return mensagem


def chamar_com_ferramentas(modelo: str, mensagens: list[dict], ferramentas: list[dict]) -> RespostaComFerramentas:
    """Chama o LLM oferecendo ferramentas. Nunca levanta exceção.

    Não envia temperatura: o agente usa o padrão do provedor.
    """
    try:
        if is_fake(modelo):
            texto, chamadas = _agente_fake(mensagens)
            return RespostaComFerramentas(texto=texto, chamadas=tuple(chamadas))

        import litellm

        resposta = litellm.completion(
            model=modelo, messages=mensagens, tools=ferramentas, tool_choice="auto", num_retries=2
        )
        mensagem = resposta.choices[0].message
        chamadas = tuple(
            ChamadaFerramenta(id=c.id, nome=c.function.name, argumentos_json=c.function.arguments or "{}")
            for c in (getattr(mensagem, "tool_calls", None) or [])
        )
        uso = getattr(resposta, "usage", None)
        try:
            custo = float(litellm.completion_cost(completion_response=resposta) or 0.0)
        except Exception:
            custo = 0.0
        return RespostaComFerramentas(
            texto=mensagem.content,
            chamadas=chamadas,
            tokens_entrada=getattr(uso, "prompt_tokens", 0) or 0,
            tokens_saida=getattr(uso, "completion_tokens", 0) or 0,
            custo_usd=custo,
        )
    except Exception as erro:
        return RespostaComFerramentas(texto=None, erro=f"{type(erro).__name__}: {erro}")


PADRAO_ARQUIVO_PROMPT = re.compile(r"[\w-]+\.(?:txt|md)", re.ASCII)


def _agente_fake(mensagens: list[dict]) -> tuple[str | None, list[ChamadaFerramenta]]:
    """Agente simulado: um roteador de palavras-chave. Não é inteligente, mas exercita o laço todo.

    - Se a última mensagem é resultado de ferramenta: responde com um resumo do resultado.
    - Senão: escolhe UMA ferramenta pelas palavras do pedido.
    """
    ultima = mensagens[-1]
    if ultima["role"] == "tool":
        resultados = []
        for mensagem in reversed(mensagens):
            if mensagem["role"] != "tool":
                break
            resultados.insert(0, mensagem["content"][:1500])
        texto = "(agente simulado) Resultado da ferramenta:\n" + "\n".join(resultados)
        if '"veredito_global"' in texto:
            texto += "\n\nLembrete: só dá pra afirmar melhora quando o veredito global é MELHOROU."
        return texto, []

    pedido = ultima["content"]
    minusculo = pedido.lower()
    arquivos = PADRAO_ARQUIVO_PROMPT.findall(pedido)

    # A "/" torna `ferramenta` só-posicional: assim um argumento chamado `nome` (ex.: ler_prompt(nome=...))
    # vai pros **argumentos em vez de colidir com o primeiro parâmetro.
    def pedir(ferramenta: str, /, **argumentos) -> tuple[None, list[ChamadaFerramenta]]:
        return None, [ChamadaFerramenta(id=f"chamada_{len(mensagens)}", nome=ferramenta,
                                        argumentos_json=json.dumps(argumentos, ensure_ascii=False))]

    if "compar" in minusculo and len(arquivos) >= 2:
        return pedir("comparar_prompts", prompt_a=arquivos[0], prompt_b=arquivos[1])
    if "salv" in minusculo and arquivos and ":" in pedido:
        conteudo = pedido.split(":", 1)[1].strip()
        return pedir("salvar_prompt", nome=arquivos[0], conteudo=conteudo)
    if ("ler" in minusculo or "mostr" in minusculo) and arquivos:
        return pedir("ler_prompt", nome=arquivos[0])
    if "execu" in minusculo:
        numero = re.search(r"\d+", pedido)
        return pedir("ler_execucoes", comparacao_id=int(numero.group()) if numero else 1)
    if "caso" in minusculo:
        return pedir("listar_casos")
    if "hist" in minusculo:
        return pedir("historico")
    if "prompt" in minusculo:
        return pedir("listar_prompts")
    return (
        "(agente simulado) Entendo pedidos como: 'liste os prompts', 'mostre v1.txt', 'liste os casos', "
        "'compare v1.txt com v2.txt', 'leia as execuções da comparação 1', 'histórico' e "
        "'salve v3.txt: <conteúdo do prompt>'. Com um modelo real no sinal.yaml, converso de verdade.",
        [],
    )

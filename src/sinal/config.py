from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

NOME_ARQUIVO_PADRAO = "sinal.yaml"


class ErroConfig(Exception):
    """Problema na configuração, com mensagem pensada pra quem vai ler no terminal."""


# ---------------------------------------------------------------------------
# Seções da configuração (cada uma vira um bloco no sinal.yaml)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Modelos:
    """Quais LLMs usar. Nomes no formato do LiteLLM; `fake/...` não chama API."""

    alvo: str = "fake/vendedor"  # o modelo que executa o prompt testado
    juiz: str = "fake/juiz"  # o modelo que dá a nota de 1 a 5
    agente: str = "fake/agente"  # o modelo do chat `sinal agente`


@dataclass(frozen=True)
class Amostragem:
    """Quantas vezes cada prompt roda em cada caso (amostragem adaptativa)."""

    rodadas_iniciais: int = 3  # todo caso começa com essas rodadas
    tamanho_lote: int = 2  # quantas rodadas a mais por lote nos casos inconclusivos
    rodadas_maximas: int = 12  # teto por caso e por versão
    minimo_para_equivalencia: int = 6  # n mínimo pra aceitar EQUIVALENTE


@dataclass(frozen=True)
class Estatistica:
    """Parâmetros dos testes estatísticos."""

    confianca: float = 0.95  # nível do IC do veredito GLOBAL
    # IC por caso, que decide se o caso para ou ganha mais rodadas. É mais alto porque a
    # amostragem adaptativa "espia" o resultado várias vezes, e cada espiada é uma chance de alarme falso.
    confianca_parada: float = 0.99
    reamostragens: int = 10_000  # nº de reamostragens do bootstrap/permutação
    limiar_irrelevante: float = 0.25  # diferença (em pontos de nota) que não importa
    semente: int | None = 42  # fixa o sorteio pra resultados reproduzíveis


@dataclass(frozen=True)
class Caminhos:
    """Onde ficam os arquivos. Caminhos relativos são relativos ao sinal.yaml."""

    casos: Path = Path("casos/casos.yaml")
    prompts: Path = Path("prompts")
    rubrica: Path | None = None  # None = usa a rubrica embutida no juiz
    banco: Path = Path(".sinal/sinal.db")
    resultados: Path = Path(".sinal/resultados")


@dataclass(frozen=True)
class Config:
    """Configuração completa do Sinal."""

    modelos: Modelos = field(default_factory=Modelos)
    # Temperaturas. None (null no YAML) = não envia o parâmetro e o provedor usa o padrão dele.
    # Necessário em modelos que recusam `temperature` (ex.: Claude Opus 5.5 e Sonnet 5.5).
    temperatura: float | None = 0.7  # modelo ALVO
    temperatura_juiz: float | None = 0.0  # juiz: 0 = o mais determinístico possível
    concorrencia: int = 8  # nº de chamadas simultâneas
    amostragem: Amostragem = field(default_factory=Amostragem)
    estatistica: Estatistica = field(default_factory=Estatistica)
    caminhos: Caminhos = field(default_factory=Caminhos)


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------


def carregar_config(caminho: str | Path | None = None) -> Config:
    """Lê o sinal.yaml e devolve uma Config validada.

    Args:
        caminho: caminho do arquivo. Se None, procura `sinal.yaml` na pasta atual.

    Raises:
        ErroConfig: arquivo inexistente, YAML inválido, chave desconhecida ou valor fora da regra.
    """
    caminho = Path(caminho) if caminho else Path.cwd() / NOME_ARQUIVO_PADRAO
    if not caminho.is_file():
        raise ErroConfig(
            f"Não achei o arquivo de configuração '{caminho}'. "
            "Rode `sinal init` pra criar um projeto de exemplo."
        )

    try:
        dados = yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as erro:
        raise ErroConfig(f"O arquivo '{caminho}' não é um YAML válido:\n{erro}") from erro

    return montar_config(dados, pasta_base=caminho.parent.resolve())


def montar_config(dados: dict[str, Any], pasta_base: Path) -> Config:
    """Transforma o dicionário lido do YAML numa Config validada.

    Separada de `carregar_config` pra dar pra testar sem criar arquivo.
    """
    if not isinstance(dados, dict):
        raise ErroConfig("O sinal.yaml deve ser um mapa de chaves (ex.: `modelos:`, `temperatura:`).")

    _recusar_chaves_desconhecidas(Config, dados, "raiz do sinal.yaml")

    config = Config(
        modelos=_montar_secao(Modelos, dados.get("modelos"), "modelos"),
        temperatura=dados.get("temperatura", Config.temperatura),
        temperatura_juiz=dados.get("temperatura_juiz", Config.temperatura_juiz),
        concorrencia=dados.get("concorrencia", Config.concorrencia),
        amostragem=_montar_secao(Amostragem, dados.get("amostragem"), "amostragem"),
        estatistica=_montar_secao(Estatistica, dados.get("estatistica"), "estatistica"),
        caminhos=_resolver_caminhos(
            _montar_secao(Caminhos, dados.get("caminhos"), "caminhos"), pasta_base
        ),
    )
    _validar(config)
    return config


def _recusar_chaves_desconhecidas(classe: type, dados: dict[str, Any], onde: str) -> None:
    """Dá erro se o YAML tiver chave que a classe não conhece (pega erro de digitação)."""
    validas = {f.name for f in fields(classe)}
    desconhecidas = set(dados) - validas
    if desconhecidas:
        raise ErroConfig(
            f"Chave(s) desconhecida(s) em '{onde}': {', '.join(sorted(desconhecidas))}. "
            f"As válidas são: {', '.join(sorted(validas))}."
        )


def _montar_secao(classe: type, dados: Any, nome: str) -> Any:
    """Cria uma seção (ex.: Amostragem) a partir do bloco do YAML; o que faltar fica no padrão."""
    if dados is None:
        return classe()
    if not isinstance(dados, dict):
        raise ErroConfig(f"A seção '{nome}' deve ser um mapa (chave: valor), mas veio: {dados!r}")
    _recusar_chaves_desconhecidas(classe, dados, nome)
    return classe(**dados)


def _resolver_caminhos(caminhos: Caminhos, pasta_base: Path) -> Caminhos:
    """Transforma texto em Path e torna relativo à pasta do sinal.yaml (não à pasta atual)."""

    def resolver(valor: Any) -> Path | None:
        if valor is None:
            return None
        caminho = Path(valor)
        return caminho if caminho.is_absolute() else pasta_base / caminho

    return replace(
        caminhos,
        casos=resolver(caminhos.casos),
        prompts=resolver(caminhos.prompts),
        rubrica=resolver(caminhos.rubrica),
        banco=resolver(caminhos.banco),
        resultados=resolver(caminhos.resultados),
    )


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------


def _eh_inteiro(valor: Any) -> bool:
    # bool é subclasse de int em Python (True == 1), então precisa excluir na mão.
    return isinstance(valor, int) and not isinstance(valor, bool)


def _eh_numero(valor: Any) -> bool:
    return _eh_inteiro(valor) or isinstance(valor, float)


def _validar(config: Config) -> None:
    """Confere tipos e regras de negócio. Junta TODOS os problemas e mostra de uma vez."""
    problemas: list[str] = []

    def exigir(condicao: bool, mensagem: str) -> None:
        if not condicao:
            problemas.append(mensagem)

    # --- modelos
    for nome in ("alvo", "juiz", "agente"):
        valor = getattr(config.modelos, nome)
        exigir(isinstance(valor, str) and valor.strip() != "", f"modelos.{nome} deve ser um texto não vazio.")

    # --- execução
    for nome in ("temperatura", "temperatura_juiz"):
        t = getattr(config, nome)
        exigir(
            t is None or (_eh_numero(t) and 0 <= t <= 2),
            f"{nome} deve ser um número entre 0 e 2, ou null pra não enviar (veio {t!r}).",
        )
    c = config.concorrencia
    exigir(_eh_inteiro(c) and c >= 1, f"concorrencia deve ser um inteiro >= 1 (veio {c!r}).")

    # --- amostragem: primeiro os tipos; as comparações entre campos só fazem sentido se forem inteiros
    a = config.amostragem
    inteiros_ok = True
    for nome in ("rodadas_iniciais", "tamanho_lote", "rodadas_maximas", "minimo_para_equivalencia"):
        valor = getattr(a, nome)
        if not _eh_inteiro(valor):
            problemas.append(f"amostragem.{nome} deve ser inteiro (veio {valor!r}).")
            inteiros_ok = False
    if inteiros_ok:
        exigir(a.rodadas_iniciais >= 2, "amostragem.rodadas_iniciais deve ser >= 2 (com 1 nota não há variação).")
        exigir(a.tamanho_lote >= 1, "amostragem.tamanho_lote deve ser >= 1.")
        exigir(
            a.rodadas_maximas >= a.rodadas_iniciais,
            "amostragem.rodadas_maximas deve ser >= rodadas_iniciais.",
        )
        exigir(
            a.minimo_para_equivalencia <= a.rodadas_maximas,
            "amostragem.minimo_para_equivalencia deve ser <= rodadas_maximas "
            "(senão nenhum caso nunca poderia ser EQUIVALENTE).",
        )

    # --- estatística
    e = config.estatistica
    exigir(
        _eh_numero(e.confianca) and 0 < e.confianca < 1,
        f"estatistica.confianca deve estar entre 0 e 1, ex.: 0.95 (veio {e.confianca!r}).",
    )
    cp = e.confianca_parada
    if not (_eh_numero(cp) and 0 < cp < 1):
        problemas.append(f"estatistica.confianca_parada deve estar entre 0 e 1, ex.: 0.99 (veio {cp!r}).")
    elif _eh_numero(e.confianca) and cp < e.confianca:
        problemas.append(
            "estatistica.confianca_parada deve ser >= confianca: a decisão de parar espia o resultado "
            "várias vezes, então precisa ser MAIS exigente, não menos."
        )
    exigir(
        _eh_inteiro(e.reamostragens) and e.reamostragens >= 100,
        f"estatistica.reamostragens deve ser um inteiro >= 100 (veio {e.reamostragens!r}).",
    )
    exigir(
        _eh_numero(e.limiar_irrelevante) and e.limiar_irrelevante >= 0,
        f"estatistica.limiar_irrelevante deve ser um número >= 0 (veio {e.limiar_irrelevante!r}).",
    )
    exigir(
        e.semente is None or _eh_inteiro(e.semente),
        f"estatistica.semente deve ser inteiro ou vazio (veio {e.semente!r}).",
    )

    if problemas:
        raise ErroConfig("Problemas no sinal.yaml:\n  - " + "\n  - ".join(problemas))

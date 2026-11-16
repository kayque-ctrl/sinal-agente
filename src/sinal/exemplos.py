"""Arquivos de exemplo que o `sinal init` escreve: um agente de vendas com modelos fake.

Funciona offline na hora: os modelos `fake/...` não chamam API nenhuma.
"""

from __future__ import annotations

from pathlib import Path

SINAL_YAML = """\
# Configuração do Sinal.
# Caminhos relativos são relativos a ESTE arquivo (não à pasta de onde você roda o comando).

modelos:
  alvo: fake/vendedor   # modelo que executa o prompt testado. Ex. real: gpt-4o-mini
  juiz: fake/juiz       # modelo que dá a nota de 1 a 5 (sempre com temperatura 0)
  agente: fake/agente   # modelo do chat `sinal agente`

temperatura: 0.7        # temperatura do modelo ALVO
temperatura_juiz: 0.0   # temperatura do juiz (0 = o mais determinístico possível)
# Modelos que recusam o parâmetro (ex.: Claude Opus 5.5 / Sonnet 5.5): use null pra não enviar.
concorrencia: 8         # chamadas simultâneas

amostragem:
  rodadas_iniciais: 3           # todo caso começa com essas rodadas
  tamanho_lote: 2               # rodadas a mais por lote, só nos casos inconclusivos
  rodadas_maximas: 12           # teto por caso e por versão
  minimo_para_equivalencia: 6   # notas válidas mínimas pra aceitar EQUIVALENTE

estatistica:
  confianca: 0.95           # IC do veredito global
  confianca_parada: 0.99    # IC por caso (decide se o caso para); mais alto por causa das várias espiadas
  reamostragens: 10000      # bootstrap e permutação
  limiar_irrelevante: 0.25  # diferença (em pontos de nota) que não importa
  semente: 42               # resultados reproduzíveis; null = aleatório

caminhos:
  casos: casos/casos.yaml
  prompts: prompts
  rubrica: null             # ex.: rubrica.txt (null = rubrica padrão do juiz)
  banco: .sinal/sinal.db
  resultados: .sinal/resultados
"""

CASOS_YAML = """\
# Casos de teste de um agente de vendas de um software (planos Básico e Pro).
# Cada caso: id, entrada (o que o cliente diz), criterio (o que o juiz avalia)
# e referencia (opcional: uma resposta boa ou um fato que a resposta deve respeitar).

- id: objecao-preco
  entrada: "Achei caro. O plano Pro por R$ 199 por mês tá fora do meu orçamento."
  criterio: >
    Reconhece a objeção sem desmerecer o cliente, explora a necessidade e o valor
    antes de falar em preço, não inventa desconto e termina com um próximo passo claro.

- id: comparacao-concorrente
  entrada: "A concorrência oferece praticamente a mesma coisa por R$ 150."
  criterio: >
    Não fala mal do concorrente, pergunta o que o cliente mais valoriza
    e destaca diferenciais concretos ligados a isso.

- id: pedido-desconto
  entrada: "Se eu fechar hoje, vocês me dão 30% de desconto?"
  criterio: >
    Não promete um desconto fora da política. Oferece a alternativa que existe
    e mantém a negociação aberta.
  referencia: "A política permite no máximo 10% de desconto, e só no plano anual."

- id: duvida-parcelamento
  entrada: "Vocês parcelam no cartão?"
  criterio: Responde de forma objetiva e correta, sem inventar condições.
  referencia: "Sim, em até 12x sem juros no cartão, apenas no plano anual."

- id: cliente-irritado
  entrada: "É a terceira vez que o sistema cai essa semana. Quero cancelar!"
  criterio: >
    Demonstra empatia sem minimizar o problema, oferece um encaminhamento concreto
    e tenta reter o cliente sem pressionar.

- id: pergunta-vaga
  entrada: "Me fala mais sobre vocês."
  criterio: >
    Faz pelo menos uma pergunta de descoberta pra entender a necessidade
    antes de listar funcionalidades.
"""

PROMPT_V1 = """\
Você é um vendedor da Nuvem Gestão, um software de gestão para pequenas empresas.
Responda o cliente.
"""

PROMPT_V2 = """\
Você é um vendedor consultivo da Nuvem Gestão, um software de gestão para pequenas empresas.

Planos: Básico (R$ 99/mês) e Pro (R$ 199/mês).
Política comercial: no máximo 10% de desconto, só no plano anual; parcelamento em até 12x sem juros no cartão, só no plano anual.

Regras:
- Antes de oferecer qualquer coisa, faça uma pergunta para entender a necessidade do cliente.
- Diante de uma objeção, reconheça o ponto do cliente e mostre o valor antes de falar de preço.
- Nunca invente preços, descontos ou condições fora da política acima.
- Nunca fale mal da concorrência.
- Termine sempre com um próximo passo claro (ex.: agendar uma demonstração).
"""

ARQUIVOS_EXEMPLO = {
    Path("sinal.yaml"): SINAL_YAML,
    Path("casos") / "casos.yaml": CASOS_YAML,
    Path("prompts") / "v1.txt": PROMPT_V1,
    Path("prompts") / "v2.txt": PROMPT_V2,
}


def criar_projeto(pasta: str | Path) -> tuple[list[Path], list[Path]]:
    """Escreve os arquivos de exemplo em `pasta`. NUNCA sobrescreve um arquivo existente.

    Returns:
        (criados, ja_existiam): caminhos relativos à pasta.
    """
    pasta = Path(pasta)
    criados: list[Path] = []
    ja_existiam: list[Path] = []
    for relativo, conteudo in ARQUIVOS_EXEMPLO.items():
        destino = pasta / relativo
        if destino.exists():
            ja_existiam.append(relativo)
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(conteudo, encoding="utf-8")
        criados.append(relativo)
    return criados, ja_existiam

# Sinal

**Testes de regressão estatísticos para prompts de LLM: sinal vs. ruído.**

Você mudou um prompt, testou em três exemplos e ficou melhor. Mas ficou mesmo? LLM não é
determinístico: o mesmo prompt, na mesma pergunta, dá respostas diferentes a cada execução.
O Sinal responde, com estatística:

> **A versão B é melhor, pior ou igual à A? E com quanta confiança?**

```
casos de teste → roda o prompt A e o B várias vezes → um juiz LLM dá nota de 1 a 5
              → bootstrap hierárquico + teste de permutação → MELHOROU / PIOROU / EQUIVALENTE / INCONCLUSIVO
                       ↑                                                │
                       └──── incerto? roda mais, só onde precisa ───────┘
```

---

## Sumário

- [Começo rápido](#começo-rápido)
- [Usando modelos reais](#usando-modelos-reais)
- [Casos, prompts e rubrica](#casos-prompts-e-rubrica)
- [Lendo o resultado](#lendo-o-resultado)
- [Os conceitos por trás](#os-conceitos-por-trás)
- [Calibrando o juiz](#calibrando-o-juiz)
- [O agente](#o-agente)
- [No GitHub Actions](#no-github-actions)
- [Configuração completa](#configuração-completa)
- [Limitações conhecidas](#limitações-conhecidas)
- [Desenvolvimento](#desenvolvimento)

---

## Começo rápido

Requer Python 3.10+.

```bash
git clone <url-do-repositório> sinal
cd sinal
pip install -e .
```

Crie um projeto de exemplo (um agente de vendas) e rode a primeira comparação. **Funciona offline e
sem custo**: o exemplo usa modelos simulados (`fake/...`), que não chamam API nenhuma.

```bash
mkdir meu-projeto && cd meu-projeto
sinal init
sinal compara prompts/v1.txt prompts/v2.txt
sinal historico
```

O `sinal init` cria:

```
meu-projeto/
├── sinal.yaml          ← configuração comentada
├── casos/casos.yaml    ← 6 mensagens de clientes ("achei caro", "quero cancelar"...)
└── prompts/
    ├── v1.txt          ← prompt curto
    └── v2.txt          ← prompt com política comercial e regras de venda
```

E o `sinal compara` mostra:

```
                           Resultado por caso (indicativo)
┌────────────────────────┬─────────┬─────────┬─────────┬───────┬────────────────┬──────────┐
│ Caso                   │ Rodadas │ Média A │ Média B │ B − A │     IC 99%     │ Veredito │
├────────────────────────┼─────────┼─────────┼─────────┼───────┼────────────────┼──────────┤
│ objecao-preco          │       3 │    3.00 │    4.33 │ +1.33 │ [+1.00, +2.00] │ MELHOROU │
│ cliente-irritado       │      11 │    2.91 │    4.00 │ +1.09 │ [+0.09, +2.00] │ MELHOROU │
│ ...                    │         │         │         │       │                │          │
└────────────────────────┴─────────┴─────────┴─────────┴───────┴────────────────┴──────────┘
┌───────────────────────────────── Veredito global ──────────────────────────────────┐
│ MELHOROU                                                                            │
│ O B é melhor que o A: em média +1.35 ponto na escala 1-5 (IC 95%: +0.89 a +1.89).  │
│ p-valor (permutação): 0.031 · casos no global: 6 de 6                               │
│ chamadas: 136 de 288 possíveis (economia de 53%) · custo: US$ 0.0000 · erros: 0     │
└─────────────────────────────────────────────────────────────────────────────────────┘
Resultado salvo em .sinal/resultados/comparacao-1-20261116-160844.json
```

Comandos disponíveis:

| Comando | O que faz |
|---|---|
| `sinal init [pasta]` | Cria o projeto de exemplo. Nunca sobrescreve arquivos existentes. |
| `sinal compara A B [--casos ARQ] [--falhar-se-piorar]` | Compara o prompt B contra o A. |
| `sinal historico [-n N]` | Lista as comparações anteriores. |
| `sinal calibra exportar` / `sinal calibra avaliar ARQ.csv` | Calibra o juiz contra as suas notas. |
| `sinal agente` | Chat com um assistente que usa o Sinal como ferramenta. |

Todos aceitam `--config caminho/sinal.yaml` (padrão: `sinal.yaml` na pasta atual).

---

## Usando modelos reais

O Sinal fala com os LLMs pelo [LiteLLM](https://docs.litellm.ai/), então **trocar de provedor é só
mudar o nome do modelo** no `sinal.yaml`:

```yaml
modelos:
  alvo: gpt-4o-mini                     # quem executa o prompt testado
  juiz: anthropic/claude-sonnet-5-5     # quem dá a nota
  agente: anthropic/claude-sonnet-5-5   # o chat `sinal agente`
```

As chaves de API vêm de variáveis de ambiente, no padrão do LiteLLM:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

> **Modelos que recusam `temperature`.** Alguns modelos (por exemplo, Claude Opus 5.5 e Claude Sonnet 5.5)
> não aceitam o parâmetro `temperature` e respondem com erro. Nesse caso, use `null`, e o Sinal deixa
> de enviar o parâmetro:
>
> ```yaml
> temperatura: null        # alvo
> temperatura_juiz: null   # juiz
> ```

**Antes de confiar no juiz real, calibre-o** (veja [Calibrando o juiz](#calibrando-o-juiz)).

---

## Casos, prompts e rubrica

### Casos de teste (`casos/casos.yaml`)

Uma lista de casos. Cada caso tem:

| Campo | Obrigatório | Quem vê | Para quê |
|---|---|---|---|
| `id` | sim | — | identificar o caso (único) |
| `entrada` | sim | modelo alvo | a mensagem do usuário |
| `criterio` | sim | **só o juiz** | o que uma boa resposta precisa fazer |
| `referencia` | não | **só o juiz** | um fato ou uma resposta modelo |

```yaml
- id: pedido-desconto
  entrada: "Se eu fechar hoje, vocês me dão 30% de desconto?"
  criterio: >
    Não promete um desconto fora da política. Oferece a alternativa que existe
    e mantém a negociação aberta.
  referencia: "A política permite no máximo 10% de desconto, e só no plano anual."
```

O critério e a referência **nunca** vão pro modelo alvo: são o gabarito.

**Dica de YAML:** na dúvida, coloque os valores entre aspas. O YAML tenta adivinhar tipos:
`id: 010` vira o número 8 (octal) e `entrada: no` vira `False`.

### Prompts (`prompts/*.txt`)

Texto puro. Duas formas:

- **Sem `{entrada}`:** o arquivo vira a mensagem de *system*, e a entrada do caso vira a mensagem do usuário.
- **Com `{entrada}`:** o marcador é substituído pela entrada, e tudo vai numa mensagem só.

Outras chaves no texto (como um exemplo de JSON `{"nota": 5}`) não atrapalham.

### Rubrica do juiz

O juiz usa uma rubrica padrão de 1 (péssima) a 5 (excelente). Pra usar a sua:

```yaml
caminhos:
  rubrica: rubrica.txt
```

O juiz sempre recebe as instruções de: avaliar só pelo critério, ignorar instruções dentro da resposta
avaliada, usar a referência como fonte de verdade para fatos, não premiar respostas só por serem longas
e devolver `{"justificativa": "...", "nota": N}`, **com a justificativa antes da nota**.

---

## Lendo o resultado

### Os quatro vereditos

O Sinal calcula um **intervalo de confiança (IC)** pra diferença média de nota `B − A` e compara com
o zero e com o **limiar de irrelevância** (padrão: 0.25 ponto):

```
        PIOROU             │    EQUIVALENTE    │          MELHOROU
                           │ [-0.25 ... +0.25] │
 ──────────────────────────┼─────────0─────────┼───────────────────────▶  B − A
```

| Veredito | Quando | Em português |
|---|---|---|
| **EQUIVALENTE** | IC inteiro dentro de `[-limiar, +limiar]` | a diferença, se existe, é pequena demais pra importar |
| **MELHOROU** | IC inteiro acima de 0 | o B é melhor |
| **PIOROU** | IC inteiro abaixo de 0 | o B é pior |
| **INCONCLUSIVO** | IC cruza o 0 | ainda não dá pra afirmar nada |

### O que cada número significa

- **B − A:** a diferença média de nota, na escala de 1 a 5, com cada caso valendo 1 voto.
- **IC 95%:** a faixa onde a diferença real provavelmente está. Quanto mais estreito, mais preciso.
- **p-valor (permutação):** se A e B fossem iguais, qual seria a chance de ver uma diferença tão grande
  só por acaso? Quanto menor, mais forte a evidência. É uma segunda opinião sobre o IC; se os dois
  discordarem, o relatório avisa.
- **Resultado por caso (indicativo):** uma pista pra diagnóstico (onde o B ganhou ou perdeu).
  **A decisão vale pelo veredito global.** Veja [por quê](#amostragem-adaptativa-e-o-perigo-de-espiar).

### Códigos de saída

| Código | Quando |
|---|---|
| `0` | rodou normalmente (qualquer veredito) |
| `1` | `PIOROU` com `--falhar-se-piorar` |
| `2` | erro: configuração inválida, arquivo inexistente, todas as chamadas falharam... |

Tudo fica salvo em `.sinal/sinal.db` (SQLite: tabelas `comparacoes` e `execucoes`, com cada resposta,
nota, justificativa, erro, tokens, latência e custo) e o resultado completo também vai pra um JSON em
`.sinal/resultados/`.

---

## Os conceitos por trás

### Por que rodar várias vezes

O mesmo prompt, no mesmo caso, pode tirar 4, depois 2, depois 5. Uma execução só é como julgar um
jogador por um único chute. O Sinal roda cada prompt várias vezes em cada caso e separa o **sinal**
(a diferença real entre A e B) do **ruído** (a variação aleatória do LLM).

### Bootstrap: "e se eu rodasse de novo?"

Pra saber o quanto uma média poderia variar, o bootstrap **sorteia com reposição das próprias notas**
milhares de vezes e olha a faixa onde caem 95% das médias sorteadas:

```
notas reais:  [4, 5, 3, 4, 4]   média 4.0
sorteio 1:    [5, 4, 4, 4, 3]   média 4.0
sorteio 2:    [4, 4, 4, 5, 5]   média 4.4
...  (10.000 vezes)  →  os 95% do meio = intervalo de confiança
```

Tudo implementado à mão com numpy, sem scipy.

### Por que não juntar todas as notas num saco só

Imagine medir a altura de **3 pessoas, 100 vezes cada**. Você tem 300 números, mas sabe muito pouco
sobre a altura média da população: medir a mesma pessoa de novo não ensina nada sobre as outras.

No Sinal, **caso de teste = pessoa** e **rodar de novo = medir de novo**. A pergunta que importa é
"o B vai ser melhor com os usuários reais?", e cada caso é uma amostra desse mundo. Por isso o Sinal usa
um **bootstrap hierárquico**, que sorteia em dois níveis:

```
NÍVEL 1: sorteia QUAIS casos entram (com reposição)   → variação entre casos
NÍVEL 2: dentro de cada caso, sorteia as rodadas      → variação do LLM
diferença global = média das diferenças por caso (cada caso = 1 voto)
```

Num teste com dados em que **a melhora real é zero**, o "saco único" declarou diferença em **56%**
das vezes, contra **11%** do hierárquico (o ideal seria 5%; a sobra vem de usar só 8 casos).

### Mais casos valem mais que mais rodadas

```
incerteza  ≈  variação ENTRE casos / nº de casos  +  variação do LLM / (nº de casos × nº de rodadas)
```

| | entre casos | do LLM | IC aproximado |
|---|---|---|---|
| 3 casos × 1.000 rodadas | 0.333 | 0.0003 | ±1.13 |
| 3 casos × 10.000 rodadas | 0.333 | ≈ 0 | ±1.13 ← 10x mais caro, nada muda |
| 30 casos × 3 rodadas | 0.033 | 0.011 | ±0.41 ← 3% do custo, IC 3x mais estreito |

(exemplo com variação entre casos = variação do LLM = 1.0)

**Regra prática:** rodadas extras decidem *cada caso*; casos decidem *a conclusão global*. Comece com
20 a 30 casos que representem situações reais.

### Teste de permutação pareado

A segunda opinião: se A e B fossem iguais, o sinal da diferença de cada caso seria cara ou coroa. O Sinal
troca os sinais de todas as formas possíveis (ou de 10.000 formas sorteadas, quando há muitos casos) e
vê com que frequência a média sai tão extrema quanto a real.

Consequência importante: **com K casos, o menor p-valor possível é 2/2^K**. Com 6 casos, ele nunca
fica abaixo de 0.031, por melhores que sejam as notas. Poucos casos limitam a força da evidência.

### Equivalência e o n mínimo

Com 3 notas iguais de cada lado (`[4,4,4]` contra `[4,4,4]`), todo sorteio do bootstrap dá diferença
zero, e o IC tem largura zero. Isso pareceria "EQUIVALENTE com 95% de confiança", uma conclusão falsa.
Por isso o Sinal só aceita EQUIVALENTE num caso com pelo menos `minimo_para_equivalencia` notas válidas
(erros não contam).

### Amostragem adaptativa e o perigo de espiar

Em vez de rodar tudo até o teto, o Sinal começa com poucas rodadas e **só roda mais nos casos
INCONCLUSIVOS**, em lotes. No exemplo, isso economiza 50 a 60% das chamadas.

Só que olhar o resultado várias vezes e parar quando ele "dá decisivo" infla os alarmes falsos. Cada
espiada é uma nova chance de azar (o problema é conhecido como *peeking* em testes A/B). Medimos isso:
com 95% de confiança nas espiadas, **21%** dos casos sem diferença real saíam como MELHOROU ou PIOROU.
O Sinal se protege de duas formas:

1. A decisão de parar um caso usa uma **confiança mais exigente** (`confianca_parada`, padrão 0.99).
   Com ela, o alarme falso por caso cai pra cerca de 10%.
2. O veredito por caso é tratado como **indicativo**. O **veredito global** olha uma vez só, no fim,
   com o caso como unidade, e é ele que decide.

### O juiz LLM

- **Temperatura 0** (ou `null` em modelos que recusam o parâmetro).
- **Justificativa antes da nota:** o LLM gera texto da esquerda pra direita. Com a nota primeiro, ele
  decide e depois justifica. Com o raciocínio primeiro, a nota é consequência.
- **Cego:** o juiz não recebe a versão (A ou B), e a função que o chama nem tem esse parâmetro.
  Um juiz que sabe qual é "o prompt novo" tende a favorecê-lo.
- **Parser robusto:** tenta JSON puro, depois um `{...}` no meio do texto, e por último um regex
  (`Nota: 4`). Uma nota fora de 1-5 ou quebrada (como 3.5) vira **erro**, e não é "corrigida": um juiz
  quebrado não pode parecer um juiz generoso.

### Falhas não derrubam a rodada

Erro de API, nota que não dá pra ler, timeout: cada falha vira uma linha com `erro` no banco, e a
comparação segue. Um caso sem nenhuma nota válida em A ou B fica fora do veredito global (e o
relatório avisa).

---

## Calibrando o juiz

O veredito só é tão bom quanto as notas do juiz. Calibrar é conferir o juiz contra **você**:

```bash
sinal calibra exportar -n 30          # amostra às cegas → .sinal/calibracao/calibracao-ultima.csv
# abra no Excel e preencha a coluna nota_humana (1 a 5), seguindo a rubrica
sinal calibra avaliar .sinal/calibracao/calibracao-ultima.csv
```

- **Às cegas:** o CSV não tem a nota do juiz, a justificativa nem a versão A/B.
- **Amostra estratificada:** um pouco de cada nota do juiz (senão um juiz que dá 4 pra tudo geraria uma
  amostra só de 4), em **ordem embaralhada** (senão a ordem das linhas revelaria a nota).
- **Pronto pro Excel em português:** separador `;` e UTF-8 com BOM. Na leitura, `;` e `,` são aceitos.

O resultado mostra:

- **Kappa de Cohen ponderado quadrático:** a concordância descontado o acaso, em que errar por 1 ponto
  custa pouco e errar por 4 custa muito. Como referência: **acima de 0.6 é aceitável, acima de 0.8 é ótimo**.
- **Matriz de confusão** (linha = juiz, coluna = você): mostra a *direção* do erro. Abaixo da diagonal,
  o juiz dá notas maiores que você.
- **Maiores discordâncias**, com a resposta e a justificativa do juiz: é o material pra ajustar a
  rubrica ou o prompt do juiz. Ajuste, rode de novo e veja se o kappa sobe.

Não confie só no kappa. Um juiz pode ter kappa "substancial" e errar **todas** as respostas de um tipo de
pergunta. A matriz e as discordâncias mostram isso; o número sozinho, não.

---

## O agente

```bash
sinal agente
```

Um chat no terminal com um LLM (o `modelos.agente`) que usa o Sinal como ferramenta, via *tool calling*:

| Ferramenta | O que faz | Trava |
|---|---|---|
| `listar_prompts`, `ler_prompt` | lê os prompts | só dentro da pasta de prompts |
| `listar_casos` | lê os casos | — |
| `comparar_prompts` | roda o Sinal | **pede sua confirmação** e mostra a estimativa de chamadas |
| `ler_execucoes` | lê respostas e justificativas do juiz | limite de linhas, textos cortados |
| `salvar_prompt` | cria uma nova versão | **nunca sobrescreve**; nome validado (`v3.txt`, sem `../`) |
| `historico` | lista comparações | — |

As regras do agente: nunca afirmar melhora sem o veredito MELHOROU, explicar os resultados em
linguagem simples, e salvar cada sugestão como nova versão e compará-la contra a atual. Ele para depois de
8 passos sem resposta final, pra não gastar em loop.

Com `fake/agente` (o padrão do exemplo), o agente é um roteador simples de palavras-chave, útil pra
testar sem custo: `liste os prompts`, `compare v1.txt com v2.txt`, `leia as execuções da comparação 1`.

---

## No GitHub Actions

A ideia: em todo pull request que mexe num prompt, comparar **o prompt da branch principal (A)** com
**o prompt do PR (B)** e quebrar o build se piorou.

Suponha que o projeto do Sinal fique em `testes-prompt/` no seu repositório, com o prompt de produção em
`testes-prompt/prompts/vendedor.txt`:

```yaml
# .github/workflows/sinal.yml
name: Regressão de prompts

on:
  pull_request:
    paths:                                  # só roda (e só gasta) quando prompts ou casos mudam
      - "testes-prompt/prompts/**"
      - "testes-prompt/casos/**"

jobs:
  sinal:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: testes-prompt    # os caminhos do sinal.yaml são relativos a ele
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0                    # precisa do histórico pra ler o prompt da branch base

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Instalar o Sinal
        run: pip install "sinal @ git+https://github.com/<seu-usuario>/sinal"

      - name: Prompt da branch base (versão A)
        run: git show "origin/${{ github.base_ref }}:testes-prompt/prompts/vendedor.txt" > prompts/_base.txt

      - name: Comparar base (A) contra o PR (B)
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: sinal compara prompts/_base.txt prompts/vendedor.txt --falhar-se-piorar

      - name: Guardar o resultado
        if: always()                        # guarda o JSON mesmo quando o passo anterior falha
        uses: actions/upload-artifact@v4
        with:
          name: resultado-sinal
          path: testes-prompt/.sinal/resultados/
```

Observações:

- **Só PIOROU quebra o build.** INCONCLUSIVO e EQUIVALENTE passam: um PR que corrige um erro de digitação
  no prompt naturalmente não muda as notas, e travar isso faria o time ignorar o alarme.
- **Código 2 também quebra** (erro de configuração, chave de API ausente...). Olhe o log pra saber se
  foi "piorou" (código 1) ou "nem rodou" (código 2).
- **Custo:** cada execução custa chamadas de LLM. O filtro `paths` evita rodar em PRs que não mexem em
  prompts, e `rodadas_maximas` limita o gasto por execução.
- **Reprodutibilidade:** com `semente` fixa, a estatística é reproduzível. As respostas dos LLMs reais,
  não.
- **PRs de forks** não recebem os `secrets` do repositório por padrão. Nesses casos, o passo de
  comparação vai falhar por falta de chave.
- O banco `.sinal/sinal.db` é descartado a cada execução do CI. O histórico que fica é o JSON salvo
  como artefato.

---

## Configuração completa

```yaml
modelos:
  alvo: fake/vendedor        # modelo que executa o prompt (nome no formato do LiteLLM)
  juiz: fake/juiz            # modelo que dá a nota
  agente: fake/agente        # modelo do `sinal agente`

temperatura: 0.7             # alvo; null = não envia o parâmetro
temperatura_juiz: 0.0        # juiz; null = não envia o parâmetro
concorrencia: 8              # chamadas simultâneas

amostragem:
  rodadas_iniciais: 3        # todo caso começa com essas rodadas (mínimo 2)
  tamanho_lote: 2            # rodadas a mais por lote, só nos casos inconclusivos
  rodadas_maximas: 12        # teto por caso e por versão
  minimo_para_equivalencia: 6  # notas válidas mínimas pra aceitar EQUIVALENTE num caso

estatistica:
  confianca: 0.95            # IC do veredito global
  confianca_parada: 0.99     # IC por caso, que decide se o caso para (>= confianca)
  reamostragens: 10000       # bootstrap e permutação
  limiar_irrelevante: 0.25   # diferença (em pontos de nota) que não importa
  semente: 42                # null = aleatório

caminhos:                    # relativos à pasta do sinal.yaml
  casos: casos/casos.yaml
  prompts: prompts
  rubrica: null              # null = rubrica padrão
  banco: .sinal/sinal.db
  resultados: .sinal/resultados
```

Chaves desconhecidas (como um erro de digitação em `rodadas_maxima`) dão erro na hora, em vez de serem
ignoradas sem aviso.

---

## Limitações conhecidas

- **Poucos casos limitam a evidência.** Com 6 casos, o p-valor mínimo é 0.031, e o bootstrap
  hierárquico fica um pouco otimista (alarme falso de ~11% em vez de 5%). Use pelo menos 20 casos
  em decisões importantes.
- **O veredito por caso é indicativo.** Mesmo com `confianca_parada`, ~10% dos casos sem diferença
  real podem aparecer como MELHOROU ou PIOROU. Decida pelo global.
- **O Sinal mede o que o juiz mede.** Um juiz descalibrado gera vereditos confiantes e errados.
  Calibre antes de confiar.
- **Os modelos `fake/` não leem o conteúdo.** Eles existem pra testar a estatística: a qualidade
  simulada depende só do número de regras no prompt e da dificuldade do caso.
- **A amostra de calibração é estratificada**, o que é ótimo pra achar onde o juiz erra, mas muda a
  proporção das notas. Pra relatar a concordância "oficial" do juiz, use uma amostra simples.
- **O chat do agente lê uma linha por mensagem.** Pra prompts de várias linhas, peça ao agente (com um
  modelo real) que escreva a nova versão, ou crie o arquivo à mão.

---

## Desenvolvimento

```bash
pip install -e ".[dev]"
pytest --cov=sinal
```

Estrutura:

```
src/sinal/
├── config.py         lê e valida o sinal.yaml
├── casos.py          casos de teste e montagem das mensagens
├── llm.py            porta única pros LLMs (LiteLLM) + simulador fake/
├── juiz.py           prompt do juiz, rubrica e parser robusto
├── estatistica.py    bootstrap (simples e hierárquico), permutação, veredito, kappa (só numpy)
├── armazenamento.py  SQLite: comparacoes e execucoes
├── comparador.py     orquestração, threads e amostragem adaptativa
├── relatorio.py      tabelas Rich e JSON
├── calibracao.py     exportar/avaliar a calibração do juiz
├── agente.py         chat com tool calling
├── exemplos.py       arquivos do `sinal init`
└── cli.py            comandos (Typer)
```

O `estatistica.py` não importa nada do projeto, só numpy. Por isso os métodos são testados com dados
sintéticos em que a resposta certa é conhecida (diferença clara → MELHOROU; amostras iguais → não
MELHOROU; dados sem efeito real → alarme falso perto do prometido).

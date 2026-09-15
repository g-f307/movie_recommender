# Atualização incremental do perfil

## Objetivo

O módulo `cinebot_ml.personalization.update` implementa uma política única,
versionada e reproduzível para transformar feedback explícito em preferências
aprendidas. A atualização é independente do recomendador: ela recebe um
`StateSnapshot`, um `FeedbackEvent`, o item avaliado e um timestamp lógico de
corte e devolve outro snapshot.

B5 poderá consumir o estado produzido, mas não fará atualizações durante o
cálculo do ranking.

## Configuração congelada

A política oficial está em `configs/profile_update_v1.yaml`, validada por JSON
Schema e protegida por lock SHA-256. A versão 1 declara:

- política `bounded_additive_decay`;
- taxa de aprendizado `0,25`;
- decaimento `0,95` por interação;
- afinidades limitadas a `[-1, 1]`;
- direção `+1` para like e `-1` para dislike;
- pesos de gênero, diretor, palavras-chave e termos da sinopse;
- até 20 termos distintos da sinopse por item;
- marcação do item avaliado como apresentado e consumido.

Os parâmetros são escolhas *a priori* do protocolo v1. Eles não foram ajustados
com validação ou holdout e não devem ser descritos como ótimos. Qualquer seleção
empírica posterior deverá usar somente treino e validação e gerar uma nova
versão da política.

## Equação

Antes de incorporar o evento `t`, toda afinidade existente sofre decaimento:

```text
base_t(f, v) = decay × affinity_(t-1)(f, v)
```

Para cada valor `v` da característica habilitada `f` presente no item:

```text
delta_t(f, v) = direction(feedback_t) × learning_rate × weight(f)

affinity_t(f, v) = clip(
    base_t(f, v) + delta_t(f, v),
    minimum_affinity,
    maximum_affinity
)
```

`direction` é positiva para like e negativa para dislike. Valores ausentes no
item não recebem contribuição. O resultado de cada característica e evento é
armazenado em `learned_preferences.event_contributions`.

## Características

| Bloco | Representação | Peso v1 |
| --- | --- | ---: |
| Gêneros | categorias canônicas do catálogo | 1,00 |
| Diretores | valores textuais normalizados | 0,75 |
| Palavras-chave | valores textuais normalizados | 0,50 |
| Sinopse | tokens normalizados distintos | 0,25 |

Termos da sinopse são normalizados, filtrados pelo tamanho mínimo, ordenados e
limitados antes da atualização. A seleção não depende de frequência global nem
consulta dados futuros.

## Exemplo

```python
from cinebot_ml.personalization import FeedbackEvent, ProfileUpdater

event = FeedbackEvent(
    event_id="event-001",
    movie_id=101,
    feedback="like",
    timestamp="2026-09-15T10:00:00Z",
    sequence=1,
    source="synthetic_user",
)

after = ProfileUpdater().update(
    before,
    event,
    movie,
    logical_timestamp="2026-09-15T10:00:00Z",
)

assert after.previous_state_id == before.state.state_id
assert after.state.version == before.state.version + 1
assert before.state.version == 0
```

Quando não existe feedback, `event=None` e `movie=None` devolvem o mesmo
snapshot sem avançar versão ou timestamp. Ausência de interação não é convertida
em dislike.

## Controles temporais e estruturais

- o evento não pode ser posterior ao timestamp lógico de corte;
- o evento não pode ser anterior ao estado recebido;
- a sequência deve continuar exatamente a versão atual;
- `event_id` não pode se repetir;
- o item fornecido deve possuir o mesmo `movie_id` do evento;
- estado e perfil inicial anteriores permanecem imutáveis;
- feedback contraditório é mantido no histórico e nas contribuições;
- afinidades preexistentes precisam ser numéricas, finitas e estar nos limites;
- o mesmo estado, evento, item e configuração produzem o mesmo snapshot.

## Metadados ausentes

Diretor, palavras-chave ou sinopse ausentes não geram erro nem valores
artificiais: o bloco correspondente simplesmente não recebe contribuição. Um
item sem esses campos ainda pode atualizar seus gêneros quando disponíveis.

## Manifesto

`ProfileUpdater.method_manifest()` registra política, versão, caminho relativo,
hash e snapshot integral da configuração, ausência de fit e proibição de uso do
holdout. Os snapshots registram separadamente a identidade do estado anterior e
o evento responsável pela transição.

## Limitações

A política é linear, não modela incerteza e trata cada evento com a mesma taxa
base. O decaimento ocorre por interação, não por duração cronológica. Termos da
sinopse não utilizam relevância global e podem conter palavras pouco
informativas. Essas decisões tornam a versão 1 auditável, mas deverão ser
consideradas na análise de robustez e nas ablações.

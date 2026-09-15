# Contrato de estado e histórico do usuário

## Objetivo

Os experimentos incrementais precisam reconstruir exatamente quais informações
estavam disponíveis antes de cada recomendação. O módulo
`cinebot_ml.personalization.contracts` define eventos, estados e snapshots
imutáveis para B4, B5 e para o futuro simulador de usuários sintéticos.

Este contrato representa evidência observada. A política que transforma um
evento em novas afinidades pertence ao módulo de atualização de perfil e não faz
parte desta etapa.

## Tipos

- `FeedbackEvent`: like ou dislike explícito, item, origem, instante e posição na
  sequência;
- `UserState`: perfil inicial, preferências aprendidas, itens apresentados,
  consumidos e avaliados, histórico e timestamp lógico;
- `StateSnapshot`: envelope que liga um estado ao estado anterior e ao evento
  responsável pela transição.

Cada estrutura declara `schema_version="1.0"`. `state_id` e `snapshot_id` são
derivados de uma representação JSON canônica e não dependem de memória, ordem de
chaves ou caminho local.

## Ciclo de vida

```text
snapshot v0
    │
    ├── recomendação e apresentação do filme 101
    │
    ├── FeedbackEvent(event-001, movie=101, like)
    │
    ▼
snapshot v1 ── previous_state_id = state_id de v0
```

O snapshot anterior nunca é alterado. Em uma etapa posterior, o atualizador de
perfil receberá o estado anterior e o evento e produzirá um novo `UserState`.

## Exemplo completo

```python
from cinebot_ml.personalization import FeedbackEvent, StateSnapshot, UserState

initial_state = UserState(
    subject_id="synthetic-agent-001",
    identity_kind="synthetic",
    logical_timestamp="2026-09-15T09:00:00Z",
    initial_profile={"ranked_genres": ["drama", "comedia"]},
)
before = StateSnapshot(state=initial_state)

feedback = FeedbackEvent(
    event_id="event-001",
    movie_id=101,
    feedback="like",
    timestamp="2026-09-15T10:00:00Z",
    sequence=1,
    source="synthetic_user",
)

updated_state = UserState(
    subject_id=initial_state.subject_id,
    identity_kind=initial_state.identity_kind,
    logical_timestamp=feedback.timestamp,
    version=1,
    initial_profile=initial_state.initial_profile,
    learned_preferences={"genre_affinity": {"drama": 0.75}},
    presented_movie_ids=(101,),
    consumed_movie_ids=(101,),
    history=(feedback,),
)
after = StateSnapshot(
    state=updated_state,
    previous_state_id=initial_state.state_id,
    transition_event_id=feedback.event_id,
)

payload = after.to_json()
reconstructed = StateSnapshot.from_json(payload)
assert reconstructed.snapshot_id == after.snapshot_id
assert before.state.version == 0
```

## Invariantes

- estado, eventos, snapshots e seus objetos internos são imutáveis;
- a versão inicial é zero e corresponde à quantidade de eventos observados;
- sequências começam em 1, são contínuas e não se repetem;
- `event_id` não se repete no histórico;
- eventos são ordenados por timestamp UTC, sequência e identificador;
- eventos posteriores ao timestamp lógico do estado são rejeitados;
- eventos no próprio timestamp do estado são permitidos, pois representam a
  transição que produziu aquele snapshot;
- todo item avaliado ou consumido precisa ter sido apresentado;
- ausência de evento não é convertida em dislike;
- origem do feedback é `real_feedback`, `synthetic_user` ou
  `human_judgment`;
- a identidade é explicitamente `synthetic` ou `pseudonymous`;
- mapas e listas internas são congelados recursivamente;
- valores não finitos, segredos e chaves indicativas de dados pessoais são
  rejeitados.

## Ordenação e timestamps

Todos os timestamps usam ISO 8601 com fuso horário e são comparados em UTC. A
ordem fornecida pelo chamador deve ser a ordem canônica; históricos fora de
ordem são rejeitados em vez de corrigidos silenciosamente. A sequência elimina
ambiguidade quando dois eventos possuem o mesmo timestamp.

## Identidade e privacidade

`subject_id` identifica a unidade experimental, não uma pessoa diretamente. O
campo `identity_kind` torna explícito se o identificador pertence a um agente
sintético ou se foi pseudonimizado antes de entrar no experimento.

Perfis, preferências e metadados não aceitam chaves que indiquem nomes, e-mails,
telefones, senhas, tokens, credenciais ou segredos. O contrato reduz o risco de
exportação acidental, mas não substitui auditoria e governança da fonte.

## Limites desta etapa

O contrato não define pesos, decaimento, reforço positivo ou penalidade por
dislike. Também não cria agentes sintéticos e não executa recomendações. Essas
responsabilidades serão implementadas nas issues posteriores da Etapa 3.

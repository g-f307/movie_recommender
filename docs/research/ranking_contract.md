# Contrato comum dos recomendadores

## Objetivo

Os métodos B0--B6 devem receber o mesmo tipo de requisição e devolver o mesmo
tipo de resultado. O contrato está implementado em
`cinebot_ml/ranking/contracts.py` e transforma as regras da seção 6 do protocolo
experimental em validações executáveis.

## Tipos

- `RecommendationRequest`: identidade do experimento, método, condição, perfil,
  seed, instante lógico, candidatos, K, perfil inicial e histórico disponível;
- `RecommendationItem`: `movie_id`, posição, score bruto e metadados permitidos;
- `RankingResult`: identidade, estado, latência, ranking e erro explícito;
- `Recommender`: protocolo estrutural implementado pelos baselines B0--B6.

Os objetos são dataclasses congeladas. Listas estruturais são representadas por
tuplas para evitar alteração acidental depois da validação.

## Exemplo completo

```python
from cinebot_ml.ranking import (
    RankingResult,
    RecommendationRequest,
    rank_scored_candidates,
)

request = RecommendationRequest(
    experiment_id="experiment-v1.0__b0__c0__p0__s42",
    protocol_version="protocol-v1.0",
    method="B0",
    method_version="1.0",
    condition="C0",
    profile="P0",
    seed=42,
    logical_timestamp="2026-09-09T04:00:00Z",
    candidate_movie_ids=(101, 102, 103),
    k=2,
    unit_id="synthetic-agent-001",
)

items = rank_scored_candidates(
    [(102, 0.75, {"title": "Filme B"}), (101, 0.75, {"title": "Filme A"})],
    request.k,
)

result = RankingResult(
    experiment_id=request.experiment_id,
    method=request.method,
    method_version=request.method_version,
    condition=request.condition,
    profile=request.profile,
    seed=request.seed,
    logical_timestamp=request.logical_timestamp,
    status="completed",
    latency_ms=1.2,
    ranked_items=items,
    metadata={"candidate_set_id": "candidate-set-001"},
)
result.validate_against(request)
payload = result.to_json()
```

No empate, o exemplo posiciona `movie_id=101` antes de `movie_id=102`.

## Invariantes

- `movie_id` é inteiro não negativo ou texto não vazio e não se repete;
- posições começam em 1, são contínuas e independem do valor bruto do score;
- scores são numéricos e finitos;
- a ordem é score descendente e, no empate, `movie_id` ascendente;
- todos os itens pertencem aos candidatos e a lista contém no máximo K itens;
- ranking concluído só pode ser vazio quando não existem candidatos;
- falhas têm `status=failed`, `error_code` e nenhum item;
- a identidade do resultado deve ser idêntica à da requisição;
- todo evento histórico possui timestamp com fuso estritamente anterior ao
  instante lógico da recomendação;
- metadados usam lista permitida e não aceitam campos pessoais ou secretos;
- requisições e resultados possuem ida e volta JSON validada.

Quando IDs inteiros e textuais coexistirem, inteiros são ordenados antes dos
textuais; dentro de cada tipo vale a ordem natural. Os módulos de candidatos
devem preservar um único tipo sempre que possível.

## Metadados permitidos

Itens podem registrar `title`, `reason`, `score_components` e
`explanation_tags`. Resultados podem registrar `candidate_set_id`, `model_id` e
`warnings`. Novos campos exigem revisão do contrato, testes e justificativa para
evitar divergência entre baselines ou exposição acidental de dados.

`unit_id` e `session_id` devem ser identificadores sintéticos ou
pseudonimizados. Nome, e-mail, telefone, credenciais e conteúdo da `.env` não
podem ser inseridos no perfil, histórico ou metadados.

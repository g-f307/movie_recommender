# Avaliação unificada de ranking

## Objetivo

O módulo `cinebot_ml.ranking.evaluation` executa B0–B3 sob o mesmo contrato,
preserva resultados por unidade experimental e só depois produz agregações. As
métricas de classificação do B3 não participam deste pipeline.

NDCG@5 é a métrica primária. Os valores oficiais de K são 5 e 10 e vêm de
`configs/experiment_v1.yaml`; o executor rejeita outros valores.

## Métricas

O módulo `cinebot_ml.ranking.metrics` implementa:

- Precision@K, Recall@K e F1@K;
- NDCG@K com ganho `2^relevância - 1`;
- MAP@K;
- MRR@K;
- Hit Rate@K;
- cobertura do catálogo pela união dos rankings das unidades.

Relevância binária usa valores zero ou positivos. NDCG também aceita graus não
negativos. Rankings duplicados, julgamentos negativos ou não finitos e itens
fora do catálogo são rejeitados.

Diversidade e novidade já aparecem no contrato com valor nulo. Elas exigem,
respectivamente, representação de distância entre itens e probabilidade de
exposição/popularidade versionada; não são inferidas silenciosamente nesta
etapa.

## Casos sem evidência suficiente

Ausência não vira relevância zero. Cada unidade recebe um estado:

- `evaluated`: todos os candidatos foram julgados e existe item relevante;
- `no_relevant_items`: julgamento completo, mas nenhum item é relevante;
- `no_judgments`: nenhum candidato foi julgado;
- `incomplete_judgments`: apenas parte do conjunto foi julgada;
- `ranking_failed`: o método falhou antes da avaliação.

Nos dois casos sem julgamentos completos, as métricas ficam ausentes e a
unidade continua no arquivo bruto. Quando não há item relevante, Precision, MRR
e Hit Rate são zero, enquanto Recall, F1, NDCG e MAP ficam nulos por serem
indefinidos. As contagens desses estados aparecem na agregação.

## Unidade e comparação justa

Uma `BenchmarkUnit` contém:

- `RecommendationRequest` com `unit_id` pseudonimizado ou sintético;
- um `CandidateSet` comum;
- julgamentos com fonte, versão e completude;
- exatamente os recomendadores B0, B1, B2 e B3.

Antes de executar, o pipeline compara IDs e identidade do conjunto candidato de
todos os métodos. Divergências bloqueiam a comparação. Para cada método e K é
preservado ranking, scores, condição, perfil, seed, fonte de relevância e estado.
Uma falha parcial não encerra os outros métodos ou unidades.

## Fontes de relevância

As fontes aceitas e mantidas separadamente são:

```text
real_feedback
human_judgment
public_dataset
synthetic_user
heuristic_proxy
```

Fonte e versão compõem `relevance_id` e a identidade determinística do
benchmark. Agregações não misturam ou renomeiam a natureza da evidência nos
dados individuais.

## Arquivo de unidades

O comando recebe JSON declarativo. Julgamentos são lista, para preservar o tipo
de `movie_id`:

```json
{
  "relevance_source": "synthetic_user",
  "relevance_version": "agents-v1",
  "units": [
    {
      "complete": true,
      "request": {
        "experiment_id": "benchmark-v1-unit-001",
        "protocol_version": "protocol-v1.0",
        "method": "B0",
        "method_version": "1.0",
        "condition": "C0",
        "profile": "P4",
        "seed": 42,
        "logical_timestamp": "2026-09-15T07:00:00Z",
        "candidate_movie_ids": [],
        "k": 5,
        "unit_id": "synthetic-unit-001",
        "profile_data": {
          "ranked_genres": ["acao", "drama", "comedia"],
          "decade_preference": "moderno",
          "popularity_preference": "popular"
        },
        "history": []
      },
      "relevance": [
        {"movie_id": 101, "value": 1},
        {"movie_id": 102, "value": 0}
      ]
    }
  ]
}
```

Com `complete=true`, a lista precisa conter exatamente todos os candidatos
calculados. Uma lista parcial deve declarar `complete=false` e ficará registrada
sem métricas confirmatórias.

## Comando

```bash
python -m cinebot_ml.benchmark benchmark \
  --units results/inputs/benchmark_units.json \
  --b2-artifact results/artifacts/b2_tfidf.json \
  --b3-model results/artifacts/b3_model.joblib \
  --b3-metadata results/artifacts/b3_metadata.json \
  --k 5 --k 10
```

O B2 deve ter sido ajustado somente em treino. O B3 deve ter sido retreinado
pelo fluxo sem leakage e congelado antes do holdout. O comando apenas carrega os
artefatos: ele não ajusta modelos, vocabulário, threshold ou hiperparâmetros.

## Saídas

O `benchmark_id` é derivado da configuração, unidades, candidatos, relevância e
manifests dos métodos. Execuções equivalentes produzem o mesmo ID.

As saídas regeneráveis seguem a configuração:

- manifesto em `results/manifests/executions/`;
- registros individuais JSONL em `results/raw/`;
- tabela individual e agregada em `results/tables/`.

Arquivos existentes não são sobrescritos. Esses diretórios permanecem fora do
versionamento; a evidência deve ser reconstruída a partir da configuração,
artefatos congelados e arquivo de unidades.

# Conjunto padronizado de candidatos

## Objetivo

O módulo `cinebot_ml.ranking.candidates` define uma única etapa de elegibilidade
antes da execução dos recomendadores. B0--B6 devem receber o mesmo
`CandidateSet`; filtros próprios dentro de um baseline são proibidos porque
tornariam a comparação experimental injusta.

## Ordem de processamento

O conjunto é construído sempre nesta ordem:

1. carregar o catálogo pelo caminho relativo da configuração experimental;
2. rejeitar IDs duplicados;
3. remover itens inválidos pela política versionada;
4. validar os filmes referenciados no histórico;
5. remover itens apresentados ou avaliados anteriormente;
6. aplicar somente filtros declarados em `profile_data.candidate_filters`;
7. ordenar os IDs canonicamente;
8. calcular a identidade e produzir o manifesto resumido.

A política `1.0` exige ID válido, título, ano entre 1888 e 2200 e ao menos um
gênero reconhecido. Sinopse, streaming, poster e palavras-chave não são critérios
mínimos porque sua ausência já foi medida pela auditoria e alguns métodos
possuem fallbacks próprios. Qualquer mudança nessa política deve criar uma nova
versão.

## Filtros explícitos

Os únicos filtros aceitos nesta versão são:

```json
{
  "candidate_filters": {
    "genres": ["drama", "acao"],
    "decade": "moderno",
    "popularity": "joia_escondida"
  }
}
```

Gêneros usam correspondência por interseção. Década e popularidade são filtros
exatos calculados pelas funções canônicas do dataset. Preferências declaradas
fora de `candidate_filters` não são inferidas como filtros: isso preserva a
separação entre elegibilidade comum e score específico de cada método.

Um filtro válido sem resultados produz um conjunto vazio e auditável. O pipeline
de ranking deverá registrar `insufficient_candidates` ou a falha aplicável,
conforme o protocolo.

## Uso

```python
from cinebot_ml.ranking import RecommendationRequest
from cinebot_ml.ranking.candidates import build_candidate_set_from_config

initial_request = RecommendationRequest(
    experiment_id="experiment-v1.0__b0__c0__p0__s42",
    protocol_version="protocol-v1.0",
    method="B0",
    method_version="1.0",
    condition="C0",
    profile="P0",
    seed=42,
    logical_timestamp="2026-09-12T20:00:00Z",
    candidate_movie_ids=(),
    k=5,
    profile_data={"candidate_filters": {"decade": "moderno"}},
)

candidate_set = build_candidate_set_from_config(initial_request)
request = candidate_set.attach_to_request(initial_request)
manifest = candidate_set.to_manifest()
```

Se uma requisição já declarar candidatos, eles deverão ser idênticos e estar na
mesma ordem do conjunto calculado. Divergências são bloqueadas, não corrigidas
silenciosamente.

## Identidade e manifesto

O `candidate_set_id` é derivado de:

- versão do contrato;
- hash do catálogo;
- política de elegibilidade;
- filtros normalizados;
- itens consumidos;
- IDs elegíveis em ordem canônica.

O timestamp não participa da identidade. Contextos equivalentes geram o mesmo
ID; alteração de catálogo, política, filtro, histórico ou candidatos gera outro.

O manifesto contém caminho relativo e hash do catálogo, política, filtros,
contagem original, IDs candidatos e contagens por motivo de exclusão. Não inclui
identificador de usuário, conteúdo do histórico ou credenciais.

## Motivos de exclusão

| Código | Significado |
|---|---|
| `invalid_id` | ID ausente ou incompatível |
| `missing_title` | título obrigatório ausente |
| `invalid_year` | ano ausente ou fora do intervalo permitido |
| `missing_genre` | nenhum gênero reconhecido |
| `already_presented_or_rated` | item presente no histórico anterior |
| `genre_filter` | item incompatível com gêneros explícitos |
| `decade_filter` | item incompatível com década explícita |
| `popularity_filter` | item incompatível com popularidade explícita |
| `invalid_popularity_metadata` | nota ou votos inválidos para o filtro solicitado |

Eventos futuros são rejeitados pelo `RecommendationRequest` antes da construção
do conjunto. Eventos passados que referenciem filmes inexistentes também são
bloqueados para impedir que inconsistências de dados sejam ocultadas.

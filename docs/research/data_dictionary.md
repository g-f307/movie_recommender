# Dicionário de dados

## Dataset supervisionado

| Campo | Tipo lógico | Origem | Finalidade |
|---|---|---|---|
| `movie_id` | identificador | catálogo TMDB | Identificar o filme de forma estável. |
| `titulo` | texto | catálogo | Título de exibição e auditoria do item. |
| `sinopse` | texto | catálogo enriquecido | Representação textual do conteúdo. |
| `genero` | categoria | catálogo normalizado | Gênero principal do filme. |
| `generos_texto` | texto | catálogo derivado | Gêneros concatenados para vetorização. |
| `diretor` | categoria/texto | catálogo | Representar afinidade com direção. |
| `ano` | inteiro | catálogo | Ano de lançamento. |
| `nota` | decimal | TMDB | Nota agregada do filme. |
| `votos` | inteiro | TMDB | Quantidade de votos usada em popularidade. |
| `duracao` | inteiro | catálogo | Duração em minutos. |
| `streaming_count` | inteiro | catálogo derivado | Quantidade de provedores de streaming. |
| `release_period` | categoria | ano derivado | Período usado no filtro de década. |
| `poster` | URL/texto | catálogo | Imagem de apresentação do filme. |
| `url` | URL/texto | catálogo | Referência pública do item. |
| `sinopse_word_count` | inteiro | sinopse derivada | Medir disponibilidade textual. |
| `keyword_count` | inteiro | palavras-chave derivadas | Medir densidade de metadados. |
| `secondary_genre_count` | inteiro | gêneros derivados | Quantidade de gêneros secundários. |
| `has_streaming` | binário | streaming derivado | Indicar disponibilidade de streaming. |
| `has_poster` | binário | poster derivado | Indicar disponibilidade de poster. |
| `pref_1` | categoria | perfil gerado | Primeiro gênero declarado. |
| `pref_2` | categoria | perfil gerado | Segundo gênero declarado. |
| `pref_3` | categoria | perfil gerado | Terceiro gênero declarado. |
| `decade_pref` | categoria | perfil gerado | Preferência de período de lançamento. |
| `popularity_pref` | categoria | perfil gerado | Preferência por popularidade. |
| `ranked_profile_text` | texto | perfil derivado | Representação textual do perfil. |
| `primary_match_pref_1` | binário | item e perfil | Aderência ao primeiro gênero. |
| `secondary_match_pref_2` | binário | item e perfil | Aderência ao segundo gênero. |
| `secondary_match_pref_3` | binário | item e perfil | Aderência ao terceiro gênero. |
| `ranked_match_count` | inteiro | item e perfil | Quantidade de correspondências ranqueadas. |
| `genre_overlap_count` | inteiro | item e perfil | Interseção entre gêneros do item e perfil. |
| `matches_decade_pref` | binário | item e perfil | Aderência ao filtro de década. |
| `popular_score` | decimal | metadados derivados | Score heurístico de popularidade. |
| `hidden_gem_score` | decimal | metadados derivados | Score heurístico de joia escondida. |
| `matches_popularity_pref` | binário | item e perfil | Aderência ao perfil de popularidade. |
| `popularity_match_score` | decimal | item e perfil | Intensidade da aderência de popularidade. |
| `score_heuristico` | decimal | regra de supervisão | Proxy que origina o rótulo inicial. |
| `relevante` | binário | proxy ou feedback | Alvo supervisionado; origem em label_source. |
| `label_source` | categoria | pipeline de supervisão | Distinguir proxy heurístico e feedback. |
| `feedback_applied` | binário | feedback derivado | Indicar sobrescrita por feedback. |
| `feedback_scope` | categoria | feedback derivado | Escopo contextual da sobrescrita. |
| `feedback_votes` | inteiro | feedback derivado | Eventos agregados no rótulo. |

## Catálogo

| Campo | Tipo lógico | Origem | Finalidade |
|---|---|---|---|
| `id` | identificador | TMDB | Identificar o filme único entre perfis. |
| `titulo` | texto | TMDB/BotCity | Apresentar e auditar o item. |
| `sinopse` | texto | TMDB/BotCity | Representar o conteúdo textual. |
| `diretor` | texto | créditos do TMDB | Representar direção e personalização. |
| `ano` | inteiro | data de lançamento | Derivar a década do filme. |
| `nota` | decimal | TMDB | Apoiar qualidade e popularidade. |
| `votos` | inteiro | TMDB | Calcular popularidade relativa. |
| `duracao` | inteiro | TMDB | Registrar duração em minutos. |
| `streaming` | lista de textos | provedores TMDB/BR | Informar disponibilidade. |
| `poster` | URL/texto | TMDB | Apresentar imagem do item. |
| `palavras_chave` | lista de textos | TMDB | Enriquecer conteúdo e personalização. |
| `perfil` | categoria | coleta | Registrar o perfil de origem da ocorrência. |
| `generos_secundarios` | lista de categorias | TMDB normalizado | Representar gêneros adicionais. |

## Feedback

| Campo | Tipo lógico | Origem | Finalidade |
|---|---|---|---|
| `user_id` | identificador pseudonimizável | Telegram | Relacionar eventos do mesmo usuário; nunca exportado na auditoria. |
| `movie_id` | identificador | recomendação | Identificar o filme avaliado. |
| `pref_1`–`pref_3` | categorias | usuário | Registrar os gêneros do contexto. |
| `decade_pref` | categoria | usuário | Registrar o filtro de década. |
| `popularity_pref` | categoria | usuário | Registrar o filtro de popularidade. |
| `feedback` | categoria | usuário | `like` ou `dislike`. |
| `label` | binário | feedback derivado | Representar feedback positivo ou negativo. |
| `timestamp` | data/hora UTC | sistema | Ordenar eventos e prevenir uso de informação futura. |

O catálogo preserva os campos originais por filme dentro de `perfis`. O `id` do TMDB define o item único; aparições em diferentes perfis são ocorrências do mesmo filme.

## Regras de interpretação

- `label_source=heuristic_proxy` não é julgamento humano.
- `feedback_exact_context` e `feedback_profile` indicam sobrescrita por feedback agregado.
- Campos derivados devem ser recalculados somente pelo pipeline versionado.
- Identificadores de usuários não podem aparecer em relatórios compartilháveis.

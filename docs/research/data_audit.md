# Auditoria dos dados

**Versão da auditoria:** 1.0
**Modo:** full

Este relatório é gerado automaticamente por `python -m cinebot_ml.data_audit`.
Não contém identificadores de usuários nem valores da `.env`.

## Como reproduzir

Auditoria completa:

```bash
python -m cinebot_ml.data_audit
```

Auditoria rápida para CI:

```bash
python -m cinebot_ml.data_audit --sample-rows 1000 \
  --json-output /tmp/data_audit.json \
  --markdown-output /tmp/data_audit.md \
  --dictionary-output /tmp/data_dictionary.md
```

## Resumo

| Indicador | Valor |
|---|---:|
| Ocorrências no catálogo | 1476 |
| Filmes únicos | 1476 |
| Filmes em múltiplos perfis | 0 |
| Linhas auditadas do dataset | 1859760 |
| Eventos de feedback | 14 |
| Usuários distintos, somente contagem | 1 |

## Distribuições do catálogo

| Gênero/perfil | Filmes |
|---|---:|
| acao | 283 |
| comedia | 283 |
| drama | 289 |
| romance | 116 |
| scifi | 133 |
| suspense | 100 |
| terror | 272 |

| Década | Filmes |
|---|---:|
| anos_2000 | 260 |
| antes_2000 | 469 |
| moderno | 747 |

| Popularidade | Filmes |
|---|---:|
| joia_escondida | 471 |
| popular | 1005 |

## Cobertura do catálogo

| Campo | Presentes | Total | Cobertura |
|---|---:|---:|---:|
| sinopse | 1476 | 1476 | 100.00% |
| diretor | 1476 | 1476 | 100.00% |
| streaming | 1072 | 1476 | 72.63% |
| poster | 1476 | 1476 | 100.00% |
| palavras_chave | 1459 | 1476 | 98.85% |

## Valores ausentes no catálogo

| Campo | Ausentes | Taxa |
|---|---:|---:|
| id | 0 | 0.00% |
| titulo | 0 | 0.00% |
| sinopse | 0 | 0.00% |
| diretor | 0 | 0.00% |
| ano | 0 | 0.00% |
| nota | 0 | 0.00% |
| votos | 0 | 0.00% |
| duracao | 0 | 0.00% |
| streaming | 404 | 27.37% |
| poster | 0 | 0.00% |
| palavras_chave | 17 | 1.15% |

## Cobertura do dataset

| Campo | Presentes | Total | Cobertura |
|---|---:|---:|---:|
| sinopse | 1859760 | 1859760 | 100.00% |
| diretor | 1859760 | 1859760 | 100.00% |
| streaming_count | 1350720 | 1859760 | 72.63% |
| poster | 1859760 | 1859760 | 100.00% |
| keyword_count | 1838340 | 1859760 | 98.85% |

## Supervisão e classes

| Origem do rótulo | Linhas |
|---|---:|
| feedback_exact_context | 11 |
| heuristic_proxy | 1859749 |

| Classe | Linhas |
|---|---:|
| 0 | 1638096 |
| 1 | 221664 |

Taxa positiva: **11.92%**.
Razão maioria/minoria: **7.389996**.

> `heuristic_proxy` é supervisão calculada por regra e não representa julgamento de usuário.

## Feedback anonimizado

| Feedback | Eventos |
|---|---:|
| dislike | 5 |
| like | 9 |

O relatório exporta somente contagens agregadas; nenhum `user_id` é incluído.

## Duplicidade e viés

- Ocorrências duplicadas entre perfis: 0.
- Registros conflitantes do mesmo filme: 0.
- Participação do decil superior nos votos: 36.57%.

## Limitações

- O catálogo é filtrado na coleta e não representa todo o universo de filmes.
- Distribuições em modo `sample` não substituem a auditoria completa.
- Métricas sobre rótulos heurísticos avaliam reprodução do proxy.
- A auditoria descreve os dados; ela não corrige valores automaticamente.

# Descoberta, não inferioridade e eficiência

**Issue:** #72 — `feat(discovery-and-cost): completar métricas de descoberta e eficiência`
**Matriz oficial:** `2942e51456add29e4c999307`
**Configuração congelada:** `configs/experiments/discovery_cost_v1.json`
**Evidência:** agentes sintéticos; 15.120 rankings oficiais

## Definições e protocolo

As métricas são calculadas individualmente e agregadas primeiro por agente. Diversidade intra-lista, novidade e cobertura têm direção preferida maior; repetição tem direção menor; exposição é descritiva e viés de popularidade prefere proximidade de zero. NDCG e custos permanecem separados, sem score composto.

Novidade, exposição e viés usam `votos` do mesmo catálogo congelado dos rankings. Essa referência é metadado externo estático, não feedback do holdout. Cobertura é a união dos itens recomendados dividida pelos 295 itens únicos do catálogo elegível congelado.

Antes da execução, a margem absoluta de não inferioridade de diversidade foi congelada em `−0,05` na escala `[0,1]`. H5 requer simultaneamente ganho de relevância B5×B4 e limite inferior do IC95% de diversidade acima da margem.

## Descoberta em K=5

| Método | NDCG | Diversidade | Novidade | Cobertura | Viés de popularidade |
|---|---:|---:|---:|---:|---:|
| B0 | 0,34708 | 0,94567 | 0,41594 | 0,02373 | +0,48863 |
| B1 | 0,55494 | 0,88005 | 0,57674 | 0,45763 | +0,18695 |
| B2 | 0,45977 | 0,87905 | 0,68530 | 0,29492 | −0,04989 |
| B3 | 0,52297 | 0,89408 | 0,61078 | 0,45763 | +0,11324 |
| B4 | 0,55494 | 0,88005 | 0,57674 | 0,45763 | +0,18695 |
| B5 | 0,53987 | 0,87545 | 0,61091 | 0,97627 | +0,10606 |

B5 amplia fortemente a cobertura e melhora novidade e viés em relação a B4, com diversidade praticamente igual, mas apresenta NDCG inferior. Entre transições C0–C5, B5 tem sobreposição média `0,4724` e repetição posicional `0,2737`, enquanto B4 tem `0,9789` e `0,9684`: a atualização incremental altera substancialmente a lista, sem converter essa mudança em ganho de relevância.

## Não inferioridade e H5

Em 35 agentes, B5−B4 em diversidade@5 foi `−0,00553`, IC95% `[−0,00937; −0,00172]`; o limite inferior supera a margem `−0,05` e o teste unilateral de não inferioridade produz `p=2,91×10⁻¹¹`. A pequena perda é estatisticamente detectável, mas permanece dentro da margem de não inferioridade.

H5, contudo, permanece **não sustentada**, pois B5−B4 em NDCG@5 é `−0,01809` e o protocolo exige ganho de relevância simultâneo. Descoberta favorável não compensa essa perda por score arbitrário.

## Eficiência controlada

O benchmark usa uma unidade C5/P5/K=10 por persona, seed 42, os mesmos 295 candidatos, 3 warm-ups e 20 repetições por unidade. Foram medidos 140 rankings por método no mesmo processo, com Python 3.14.7 em Linux. Latência usa `perf_counter_ns`; memória é o pico incremental de `tracemalloc`.

| Método | Latência mediana (ms) | P95 (ms) | Memória média (KiB) | Artefato (bytes) |
|---|---:|---:|---:|---:|
| B0 | 18,6981 | 19,5600 | 147,83 | 0 |
| B1 | 35,5572 | 36,8992 | 250,25 | 0 |
| B2 | 15,7600 | 16,4382 | 386,15 | 224.246 |
| B3 | 204,6234 | 211,7763 | 1.449,60 | 231.988 |
| B4 | 35,5345 | 36,3328 | 250,94 | 0 |
| B5 | 80,1858 | 82,1334 | 347,70 | 0 |

Tempo de treinamento não foi reestimado: B2/B3 usam artefatos oficiais congelados e os demais métodos não possuem etapa de treino separada. O resultado é registrado como `not_applicable_frozen_artifact`, sem inventar custo ausente. Tamanho considera somente artefatos serializados de inferência.

## RQ3 e RQ4

RQ3 é respondida negativamente no simulador: B5 preserva diversidade e melhora cobertura/novidade, mas perde relevância contra B4. RQ4 passa a ter evidência de qualidade e custo, porém não existe vencedor único. Todos os métodos permaneceram na fronteira multidimensional de Pareto porque cada um preserva ao menos um compromisso entre qualidade, descoberta, latência e memória. A conclusão correta é apresentar os trade-offs, não escolher um campeão por ponderação retrospectiva.

## Reprodução

```bash
make discovery-cost PYTHON=.venv/bin/python
dvc pull results/derived/specialized_v1/discovery_cost/2942e51456add29e4c999307.dvc
```

O pacote contém tabelas de descoberta, estabilidade, eficiência, não inferioridade e Pareto; duas figuras; relatório JSON; e manifesto com hashes.

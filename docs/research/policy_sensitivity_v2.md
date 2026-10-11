# Sensibilidade da política incremental v2

## Pergunta e isolamento

A execução v1.1 usa uma única escolha *a priori* para taxa de aprendizado,
decaimento e peso do ajuste por feedback. A execução complementar v2 avalia se
o comportamento de B5 depende de um desses valores, sem reabrir nem modificar
o holdout oficial v1.1.

Toda escolha de cenário ocorre em `validation`. O holdout v2 só poderá ser
consultado por uma etapa posterior depois que a decisão de validação estiver
congelada. A seleção não torna o cenário escolhido ótimo fora do simulador nem
altera a conclusão da execução v1.1.

## Cenários congelados

`configs/experiments/policy_sensitivity_v2.yaml` contém sete cenários e é
protegido por schema e lock SHA-256. S0 preserva os parâmetros v1; os demais
alteram uma variável por vez.

| Cenário | Taxa de aprendizado | Decaimento | Peso de feedback |
| --- | ---: | ---: | ---: |
| S0 | 0,25 | 0,95 | 0,50 |
| S1 | 0,10 | 0,95 | 0,50 |
| S2 | 0,40 | 0,95 | 0,50 |
| S3 | 0,25 | 0,80 | 0,50 |
| S4 | 0,25 | 1,00 | 0,50 |
| S5 | 0,25 | 0,95 | 0,25 |
| S6 | 0,25 | 0,95 | 0,75 |

Para cada cenário, o módulo materializa configurações em memória de
`ProfileUpdater` e B5. As configurações v1 permanecem imutáveis; os manifestos
do cenário carregam o hash da matriz e seus três parâmetros.

## Regra de seleção

O pipeline aceita somente observações com `partition=validation`. Cada cenário
precisa apresentar o mesmo conjunto de `comparison_id`, e cada par deve manter
o mesmo agente, seed e `candidate_set_id`. Falhas, exclusões e métricas ausentes
são preservadas; uma seleção só é produzida quando todos os cenários possuem
pares completos e avaliados.

`write_validation_observations` grava as linhas de validação sem sobrescrevê-las,
incluindo cenários avaliados, falhos e excluídos. Isso mantém os resultados que
não foram selecionados disponíveis para a análise posterior.

A regra pré-especificada é a maior média de NDCG@5 nos pares completos. Empates
são resolvidos pelo identificador canônico do cenário. O relatório registra a
média de cada cenário, contagem de pares, observações inválidas, hash da matriz
e `selection_id`.

`freeze_validation_selection` grava essa decisão uma única vez em JSON e cria
um lock SHA-256. `load_frozen_validation_selection` verifica hash, execução,
matriz, cenário, cobertura de todos os cenários e a ausência de uso de holdout.
Uma etapa de avaliação v2 deve usar esse carregador antes de abrir o holdout.

## Reprodução

```bash
.venv/bin/python -m unittest \
  tests.test_policy_sensitivity \
  tests.test_profile_update \
  tests.test_baseline_incremental \
  tests.test_temporal_simulation
```

Os testes verificam schema/lock, execução determinística dos cenários,
rastreabilidade de parâmetros, isolamento da validação, consistência do
pareamento e congelamento verificável da seleção.

## Limites

Esta matriz não gera por si só resultados científicos: ela define e protege a
fase de seleção. A futura execução v2 deverá registrar todas as células,
inclusive cenários negativos, falhos ou não selecionados, e manter os resultados
de validação separados da avaliação final.

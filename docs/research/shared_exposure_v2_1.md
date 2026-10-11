# Exposição compartilhada v2.1

## Decisão metodológica

O diagnóstico da execução `protocol-v2.0` mostrou que, a partir de C2, B4 e B5
passavam a receber conjuntos de candidatos e relevâncias diferentes. Cada
método definia o item exposto em sua própria trajetória; o feedback e as
remoções posteriores, portanto, deixavam de representar o mesmo problema de
ranking. A auditoria bloqueou corretamente essa matriz. Esses resultados não
podem ser congelados, usados no holdout ou citados como evidência confirmatória.

O `protocol-v2.1` corrige o desenho, sem alterar a execução v1.1 nem apagar os
artefatos diagnósticos do v2.0. Para cada `comparison_id`, a exposição será uma
permutação canônica determinística dos itens elegíveis, parametrizada pela
identidade que já contém a seed. A ordem é criada antes da avaliação dos
métodos e não pode depender de rankings, scores ou feedback produzidos por B4
ou B5.

## Contratos congelados

| Artefato | Função |
| --- | --- |
| `closed_loop_v2_1.yaml` | Declara C0–C5, tempo lógico, exaustão e a política `shared-seeded-v2.1`. |
| `execution_v2_1.yaml` | Cria a execução `execution-v2.1` e isola toda saída em `results/v2_1/`. |
| `matrix_v2_1.yaml` | Mantém B4/B5, personas, perfis, condições, seeds, K e cenários S0–S6. |
| `experiment_v2_1_{validation,holdout}.yaml` | Separa seleção de cenário na validação e avaliação única no holdout. |
| `policy_sensitivity_v2_1.yaml` | Mantém a seleção pré-holdout sob a nova identidade de execução. |

Todos os YAMLs possuem schema estrito e lock SHA-256. Os schemas recusam
protocolo, política de exposição, caminho de saída ou identidade de execução de
outra versão. Em particular, `protocol-v2.1` só pode escrever em
`results/v2_1/`; assim, uma reexecução não sobrescreve nem aparenta validar os
artefatos v2.0.

## Invariantes que a implementação deve preservar

- B4 e B5 recebem a mesma sequência de itens elegíveis para cada par;
- a sequência varia com a identidade do par e sua seed, não com o método;
- C0–C5, tempo lógico, remoção de itens consumidos e exaustão continuam
  registrados; e
- histórico, candidatos, relevâncias e hash da trajetória precisam coincidir
  antes de uma diferença B5 − B4 ser analisada.

Esta mudança versiona o desenho e seus limites. A materialização da trajetória
compartilhada, o hash por comparação e a adaptação do runner pertencem à issue
#103. Até ela ser concluída, nenhuma matriz v2.1 deve ser executada.

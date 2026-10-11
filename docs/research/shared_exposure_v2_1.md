# Exposição compartilhada v2.1

## Decisão metodológica

O diagnóstico da execução `protocol-v2.0` mostrou que, a partir de C2, B4 e B5
passavam a receber conjuntos de candidatos e relevâncias diferentes. Cada
método definia o item exposto em sua própria trajetória; o feedback e as
remoções posteriores, portanto, deixavam de representar o mesmo problema de
ranking. A auditoria bloqueou corretamente essa matriz. Esses resultados não
podem ser congelados, usados no holdout ou citados como evidência confirmatória.

O `protocol-v2.1` corrige o desenho, sem alterar a execução v1.1 nem apagar os
artefatos diagnósticos do v2.0. A revisão operacional `2.1.1` também isola uma
tentativa v2.1 que não alcançou o congelamento da seleção. Ela escreve somente
em `results/v2_1/revision_2/`; esses artefatos anteriores continuam diagnósticos
e não participam da evidência confirmatória.

Para cada `exposure_id`, a exposição é uma permutação canônica determinística
dos itens elegíveis. Essa identidade inclui partição, condição, perfil, persona,
seed e K, mas exclui método, cenário de política e seleção. A ordem é criada
antes da avaliação dos métodos e não pode depender de rankings, scores ou
feedback produzidos por B4 ou B5. Assim, os cenários S0–S6 comparam políticas
sobre a mesma entrada observável, e não sobre trajetórias distintas.

## Contratos congelados

| Artefato | Função |
| --- | --- |
| `closed_loop_v2_1.yaml` | Declara C0–C5, tempo lógico, exaustão e a política `shared-seeded-v2.1`. |
| `execution_v2_1.yaml` | Cria a revisão `2.1.1` de `execution-v2.1` e isola toda saída em `results/v2_1/revision_2/`. |
| `matrix_v2_1.yaml` | Mantém B4/B5, personas, perfis, condições, seeds, K e cenários S0–S6. |
| `experiment_v2_1_{validation,holdout}.yaml` | Separa seleção de cenário na validação e avaliação única no holdout. |
| `policy_sensitivity_v2_1.yaml` | Mantém a seleção pré-holdout sob a nova identidade de execução. |

Todos os YAMLs possuem schema estrito e lock SHA-256. Os schemas recusam
protocolo, política de exposição, caminho de saída ou identidade de execução de
outra versão. Em particular, `protocol-v2.1` só pode escrever em
`results/v2_1/revision_2/`; assim, uma reexecução não sobrescreve nem aparenta
validar os artefatos v2.0 ou a tentativa v2.1 anterior.

## Materialização e auditoria

- B4 e B5 recebem a mesma sequência de itens elegíveis em todos os cenários
  S0–S6 de uma mesma exposição;
- a sequência é uma permutação por hash de `exposure_id`, que já incorpora a
  seed, e não consulta método, cenário, score, ranking ou feedback de B4/B5;
- C0–C5, tempo lógico, remoção de itens consumidos e exaustão continuam
  registrados; e
- histórico, candidatos, relevâncias e hash da trajetória precisam coincidir
  antes de uma diferença B5 − B4 ser analisada.

`write_units` grava um único arquivo em
`results/v2_1/revision_2/units/<exposure_id>.json`. Esse arquivo contém a
trajetória, seu SHA-256, sua auditoria, os `comparison_id` de S0–S6 e as
unidades derivadas por método e cenário. O runner recusa o arquivo se a célula
solicitada não pertencer a essa exposição ou se candidatos, relevâncias e hash
não coincidirem. Antes do congelamento, a seleção compara entre cenários
`agent_id`, seed, candidato, relevância, versão de estado e hash da trajetória;
a auditoria dos checkpoints e a análise v2.1 repetem as verificações.

## Execução

Depois de uma recuperação limpa dos dados e artefatos, use a mesma sequência
da orquestração com o protocolo explícito:

```bash
.venv/bin/python -m cinebot_ml.experiments.v2_orchestration validation \
  --protocol protocol-v2.1
```

As etapas seguintes devem usar o mesmo `--protocol protocol-v2.1`. A execução
completa pode demandar tempo considerável; o resultado v2.0 continua somente
um diagnóstico bloqueado e não deve ser combinado a esta nova família.

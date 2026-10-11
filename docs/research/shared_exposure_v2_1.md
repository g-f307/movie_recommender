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

## Materialização e auditoria

- B4 e B5 recebem a mesma sequência de itens elegíveis para cada par;
- a sequência é uma permutação por hash de `comparison_id`, que já incorpora a
  seed, e não consulta método, score, ranking ou feedback de B4/B5;
- C0–C5, tempo lógico, remoção de itens consumidos e exaustão continuam
  registrados; e
- histórico, candidatos, relevâncias e hash da trajetória precisam coincidir
  antes de uma diferença B5 − B4 ser analisada.

`write_units` grava um único arquivo em
`results/v2_1/units/<comparison_id>.json`. Esse arquivo contém a trajetória,
seu SHA-256, sua auditoria e duas unidades derivadas (`B4` e `B5`). O runner
recusa o arquivo se candidatos, relevâncias ou hash não coincidirem entre os
métodos. A auditoria dos checkpoints repete essa verificação antes de autorizar
o congelamento; a análise v2.1 também a exige ao ler um manifesto selado.

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

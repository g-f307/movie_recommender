# Orquestração da execução v2

## Objetivo

`cinebot_ml.experiments.v2_orchestration` impõe a sequência da execução
complementar: `validation` completa, seleção congelada, `holdout`, auditoria e
manifesto selado. A execução oficial v1.1 não é lida nem modificada.

As saídas das partições permanecem separadas em
`results/v2/raw/validation/` e `results/v2/raw/holdout/`. A finalização cria
uma cópia indexada em `results/v2/manifests/executions/<id>/` apenas para o
consumo da análise v2; ela preserva os hashes e a origem de cada célula.

## Ordem obrigatória

```bash
# Estas duas execuções podem demorar; rode-as manualmente.
.venv/bin/python -m cinebot_ml.experiments.v2_orchestration validation
.venv/bin/python -m cinebot_ml.experiments.v2_orchestration freeze-selection

# Copie o caminho JSON impresso pelo comando anterior.
.venv/bin/python -m cinebot_ml.experiments.v2_orchestration holdout \
  --selection results/v2/reports/validation/<matrix_id>/selection.json
.venv/bin/python -m cinebot_ml.experiments.v2_orchestration audit \
  --selection results/v2/reports/validation/<matrix_id>/selection.json
.venv/bin/python -m cinebot_ml.experiments.v2_orchestration finalize \
  --selection results/v2/reports/validation/<matrix_id>/selection.json
```

Os comandos `validation` e `holdout` materializam unidades e executam a matriz
inteira por padrão. Para um teste controlado, informe o mesmo recorte em todas
as etapas, por exemplo `--condition C0 --profile P0 --persona consistent
--seed 42 --k 5`. Não misture recortes entre validation, congelamento e
holdout: isso altera a identidade da matriz e a finalização recusará o fluxo.

No início de `validation`, o projeto gera
`results/v2/manifests/splits/movie_id_split.json` a partir de
`datasets/movie_preferences.csv`, usando seed 42 e o protocolo v2. O manifesto
inclui a identidade e o hash da execução e não reutiliza o split legado. Essa
etapa pode demorar com o dataset completo.

## Proteções

- `holdout` recusa qualquer chamada sem `selection.json` e lock compatíveis;
- falhas, exclusões e métricas ausentes da validation são gravadas em
  `validation_observations.jsonl` e bloqueiam o congelamento;
- a seleção verifica os sete cenários S0–S6 e exige pares compatíveis;
- auditoria confirma completude, hashes, pareamento, método, partição e
  proveniência da execução antes da finalização; e
- `execution.manifest.json` não pode ser sobrescrito e é aceito por
  `cinebot_ml.analysis.v2_analysis`.

O dataset de origem precisa estar disponível localmente antes dos comandos de
execução. A recuperação de dados privados usa o remoto DVC documentado em
`artifact_reproduction.md`; credenciais nunca são passadas na linha de comando
nem registradas nos manifestos.

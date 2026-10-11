# Configuração operacional da execução v2

## Finalidade

As matrizes da execução complementar v2 são separadas em duas partições:
`validation` e `holdout`. Ambas usam somente B4 e B5, o perfil observado v2,
exposição em ciclo fechado e os mesmos candidatos, personas, seeds, condições e
níveis de K declarados em `configs/experiments/matrix_v2.yaml`.

Essa separação não reabre nem modifica o holdout oficial v1.1. A validação
avalia os sete cenários S0–S6; o holdout só pode enumerar o cenário escolhido
e congelado a partir da validação.

## Contratos congelados

- `configs/experiment_v2_validation.yaml` e seu lock descrevem a partição de
  validação, cuja saída pertence a `results/v2/raw/validation`.
- `configs/experiment_v2_holdout.yaml` e seu lock descrevem a partição final,
  cuja saída pertence a `results/v2/raw/holdout`.
- `configs/experiments/matrix_v2.yaml` e seu lock declaram B4/B5, C0–C5,
  P0–P5, as sete personas sintéticas, cinco seeds, K=5 e S0–S6.
- Os schemas rejeitam métodos da v1.1, exposição neutra, caminhos fora da
  árvore v2 e configurações não congeladas.

Cada célula v2 registra `execution_id`, hash da configuração de execução,
`partition`, identificador e hash do cenário de política. No holdout também
registra a identidade e o hash do arquivo de seleção. Esses campos entram nos
IDs de célula, pareamento e matriz; células v1.1 não mudam de identidade porque
os campos ausentes não são serializados.

## Proteção do holdout

`build_v2_experiment_matrix` exige uma seleção carregada por
`load_frozen_validation_selection`. O arquivo precisa ter lock SHA-256,
pertencer a `execution-v2`, cobrir todos os cenários da matriz de sensibilidade
e comprovar que o holdout não foi usado na seleção. Sem essa referência, a
enumeração falha antes de qualquer unidade, ranking ou escrita.

## Inspeção sem escrita

Liste uma amostra da matriz de validação:

```bash
.venv/bin/python -m cinebot_ml.experiments.v2_config list \
  --partition validation --condition C0 --profile P0 \
  --persona consistent --seed 42 --k 5
```

Após a futura orquestração congelar a seleção, liste o holdout usando o arquivo
e lock gerados por ela:

```bash
.venv/bin/python -m cinebot_ml.experiments.v2_config list \
  --partition holdout \
  --selection results/v2/reports/<selection_id>/selection.json \
  --condition C0 --profile P0 --persona consistent --seed 42 --k 5
```

Esses comandos apenas produzem o manifesto da matriz em stdout. A geração das
unidades, o runner B4/B5 e a sequência validation → seleção → holdout pertencem
às issues #96 e #97.

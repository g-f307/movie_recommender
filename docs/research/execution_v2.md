# Execução experimental v2 isolada

## Finalidade e estado atual

A execução v2 é uma área separada para o estudo complementar da Etapa 8. Ela
não altera, reutiliza nem reavalia o holdout oficial v1.1. A infraestrutura
introduz uma identidade própria, uma configuração congelada e diretórios
exclusivos; ainda não produz evidência científica v2. A geração em ciclo
fechado será implementada na issue #87.

## Contrato

`configs/experiments/execution_v2.yaml` descreve a execução `execution-v2` e
seu lock registra o SHA-256 exato do arquivo. A carga falha antes de criar
artefatos se o schema, o lock ou os caminhos forem inválidos. A especificação
também referencia a configuração congelada de perfil declarado v2; ela é um
insumo auditável, não uma saída e não permite que B4/B5 acessem a preferência
latente.

As saídas ficam abaixo de `results/v2/`:

```text
results/v2/
├── derived/test_catalog.json
├── manifests/
│   ├── executions/
│   └── splits/movie_id_split.json
├── raw/
├── reports/
└── units/
```

Os diretórios e arquivos da v1.1, incluindo `results/units_v1_1/`,
`results/raw/official_v1_1/` e o catálogo oficial, são proibidos para a v2.
Cada célula e o manifesto da matriz v2 carregam `execution_id` e
`execution_config_sha256`; por isso, uma célula v1.1 não possui a mesma
identidade de uma célula v2.

## Comandos

Validar schema e lock, sem escrita:

```bash
python -m cinebot_ml.experiments.execution validate \
  --spec configs/experiments/execution_v2.yaml
```

Preparar apenas os diretórios da v2:

```bash
python -m cinebot_ml.experiments.execution prepare \
  --spec configs/experiments/execution_v2.yaml
```

Quando os contratos v2 de configuração, matriz e unidades estiverem
implementados, auditar os checkpoints e finalizar sem sobrescrever o manifesto:

```bash
python -m cinebot_ml.experiments.execution audit \
  --spec configs/experiments/execution_v2.yaml \
  --experiment-config configs/experiment_v2.yaml \
  --matrix-config configs/experiments/matrix_v2.yaml

python -m cinebot_ml.experiments.finalize \
  --execution-spec configs/experiments/execution_v2.yaml \
  --config configs/experiment_v2.yaml \
  --matrix-config configs/experiments/matrix_v2.yaml
```

O comando de auditoria recusa manifestos, células e diretórios que não
coincidam com a execução declarada. A finalização grava
`execution.manifest.json` uma única vez, com hashes da especificação, matriz,
insumos e checkpoints.

## Compatibilidade

Sem `--execution-spec`, os comandos existentes conservam o comportamento,
identidades e artefatos v1.1. A nova especificação não transforma a política
`neutral-exposure-v1` em ciclo fechado: uma tentativa de gerar unidades v2
nessa política é recusada até a implementação específica da #87.

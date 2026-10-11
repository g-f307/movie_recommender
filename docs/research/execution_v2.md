# Execução experimental v2 isolada

## Finalidade e estado atual

A execução v2 é uma área separada para o estudo complementar da Etapa 8. Ela
não altera, reutiliza nem reavalia o holdout oficial v1.1. A infraestrutura
introduz uma identidade própria, uma configuração congelada, diretórios
exclusivos e unidades B4/B5 com exposição em ciclo fechado. Ainda não produz
evidência científica v2: a matriz e a execução complementar permanecem
artefatos a versionar antes de qualquer análise.

## Contrato

`configs/experiments/execution_v2.yaml` descreve a execução `execution-v2` e
seu lock registra o SHA-256 exato do arquivo. A carga falha antes de criar
artefatos se o schema, o lock ou os caminhos forem inválidos. A especificação
também referencia as configurações congeladas de perfil declarado e exposição
em ciclo fechado v2. Ambas são insumos auditáveis, não saídas. O perfil
declarado não permite que B4/B5 acessem a preferência latente e a política de
exposição registra o ranking que originou cada evento de feedback.

A matriz de sensibilidade também é um insumo separado. Ela varia parâmetros de
B5 exclusivamente na validação v2 e exige uma seleção congelada antes de abrir
qualquer holdout v2. Os arquivos v1 continuam imutáveis durante esse processo.
As configurações operacionais da validação e do holdout estão descritas em
[`v2_experiment_config.md`](v2_experiment_config.md).

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

As matrizes v2 podem ser inspecionadas sem escrita. A validação enumera S0–S6;
o holdout exige uma seleção congelada e compatível:

```bash
python -m cinebot_ml.experiments.v2_config list --partition validation
python -m cinebot_ml.experiments.v2_config list --partition holdout \
  --selection results/v2/reports/<selection_id>/selection.json
```

Depois que as unidades e o runner forem implementados, será possível auditar os
checkpoints e finalizar sem sobrescrever o manifesto:

O runner de células e seus requisitos de proveniência são documentados em
[`v2_cell_runner.md`](v2_cell_runner.md). A orquestração completa continua
separada: a validação deve terminar e congelar a seleção antes da abertura do
holdout.

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
identidades e artefatos v1.1. Com a especificação v2, a geração de unidades
usa exclusivamente `closed-loop-v2`, preserva uma unidade por método e associa
B4 e B5 por uma chave de pareamento compartilhada. A política
`neutral-exposure-v1` continua restrita ao fluxo legado.

Após o congelamento da seleção paramétrica, a análise v2 é executada por
`cinebot_ml.analysis.v2_analysis`. Ela exige o manifesto selado, mantém
`validation` e `holdout` separados e pode reportar v1.1 somente como uma
comparação de desenho, sem combinar estatisticamente as duas execuções. Veja
[`v2_analysis.md`](v2_analysis.md) para o contrato de inferência e de artefatos.

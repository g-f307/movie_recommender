# Runner de células v2

## Escopo

`cinebot_ml.experiments.v2_cell_runner` executa B4 ou B5 a partir de uma
unidade v2 previamente materializada. Ele não consulta `results/units_v1_1`,
`experiment_v1_holdout.yaml`, variáveis de ambiente nem artefatos de B2/B3.

Antes do ranking, o runner verifica a identidade da execução, a partição, o
cenário de política, os snapshots, o histórico causal, a auditoria da
trajetória, o pareamento e a reconstrução do conjunto candidato. Falhas são
lançadas de modo estruturado e preservadas por `execute_matrix` como checkpoints
com status `failed`; não são convertidas em resultados avaliados.

## Materialização

`prepare_v2_units(matrix, execution_spec)` reconstrói o catálogo v2 a partir
dos caminhos declarados na especificação e grava uma unidade por célula sob
`paths.units`. Cada unidade contém perfil declarado, proveniência do ruído,
trajetória de ciclo fechado, relevância sintética, snapshots e chave B4/B5.

As unidades normalizam `latency_ms` para `0.0` dentro das trajetórias
serializadas. Esse valor é telemetria dependente da máquina e não participa da
relevância, da política de feedback ou do pareamento; mantê-lo impediria a
reconstrução determinística da mesma unidade. Avaliações de custo permanecem
em artefatos próprios de desempenho.

## Avaliação

`run_v2_cell` retorna o registro individual do benchmark e preserva no mesmo
objeto a partição, cenário e hash, perfil observado, política de exposição,
auditoria causal, pareamento, candidato e relevância. `build_v2_runner` fornece
o adaptador sem estado global que será usado pela orquestração da #97.

O runner não abre o holdout por conta própria. A autorização para enumerar essa
partição continua pertencendo ao contrato de seleção congelada da #95.

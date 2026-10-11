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
dos caminhos declarados na especificação. No `protocol-v2.0`, grava uma unidade
por célula sob `paths.units`. Na revisão 2.1.1 do `protocol-v2.1`, grava uma
única unidade por `exposure_id`, compartilhada por B4/B5 e pelos cenários S0–S6
da mesma entrada experimental.

Para v2.1, a exposição é ordenada apenas por `exposure_id`; ela não consulta
cenário de política nem ranking de nenhum método. O arquivo registra
`trajectory_sha256`, os `comparison_id` atendidos e a célula B4/B5 de cada um.
O runner rejeita uma unidade que não contenha a célula solicitada ou que tenha
candidatos, relevâncias e hash de trajetória divergentes. O mesmo hash entra na
auditoria de checkpoints e na análise selada, impedindo que uma divergência
silenciosa seja tratada como par válido.

O nome do arquivo usa `materialization_id`: em validation ele é o próprio
`exposure_id`; no holdout, também inclui a seleção congelada. A trajetória
causal permanece definida por `exposure_id`, enquanto o arquivo não colide com
uma execução anterior que tenha escolhido outro cenário.

As unidades normalizam `latency_ms` para `0.0` dentro das trajetórias
serializadas. Esse valor é telemetria dependente da máquina e não participa da
relevância, da política de feedback ou do pareamento; mantê-lo impediria a
reconstrução determinística da mesma unidade. Avaliações de custo permanecem
em artefatos próprios de desempenho.

## Avaliação

`run_v2_cell` retorna o registro individual do benchmark e preserva no mesmo
objeto a partição, cenário e hash, perfil observado, política de exposição,
auditoria causal, pareamento, hash da trajetória, candidato e relevância. `build_v2_runner` fornece
o adaptador sem estado global que será usado pela orquestração da #97.

O runner não abre o holdout por conta própria. A autorização para enumerar essa
partição continua pertencendo ao contrato de seleção congelada da #95.

# Pré-voo do congelamento experimental

A issue #46 exige execução piloto e oficial. Este pré-voo **não** é o piloto
nem o congelamento: ele enumera uma matriz reduzida verificável e registra
ausências antes de executar qualquer célula. A matriz seleciona B0–B5,
C0/C2/C4, P0/P4/P5, personas `consistent` e `noisy`, seeds 42 e 137,
e os dois valores oficiais de K: 432 células.

```bash
python -m cinebot_ml.experiments.freeze --output /tmp/movie-freeze-preflight.json
```

O comando sai com código 2 quando existem bloqueios. O relatório contém
commit, dimensões e IDs de cada célula, presença, tamanho e SHA-256 dos dados
e artefatos locais, além da validação de contrato do split e dos artefatos.
Dados e resultados brutos permanecem fora do Git. Mesmo metadados B3 válidos
**não** comprovam que o modelo foi treinado no split atual; a proveniência de
treino ainda precisa ser verificada separadamente.

O escopo v1 segue o roadmap com B0–B5. B6 fica adiado porque ainda não há
hipótese nem componente técnico distinto de B5 definidos antes do holdout;
resultados do teste final não poderão ser usados para inventá-lo. A revisão do
orientador está indisponível e **não** é tratada como aprovação. Isso deve ser
relatado como limitação metodológica na pesquisa.

O split e o artefato B2 foram gerados localmente a partir dos dados atuais,
mas são ignorados pelo Git e devem ser regenerados e verificados em outro clone.
Antes de executar o piloto faltam as unidades pareadas e o artefato B3 treinado
no split experimental. Antes da execução oficial ainda são necessárias
a verificação de completude, retomada e isolamento entre saídas piloto e
oficiais. Nenhuma tag experimental ou resultado confirmatório deve ser criado
a partir deste pré-voo.

O adaptador `cinebot_ml.experiments.cell_runner:run_cell` já chama o benchmark
B0–B5 para uma unidade declarativa por `comparison_id` em `results/units/`.
Ele não gera julgamentos, feedback nem unidades sintéticas: se a unidade falta,
ou suas dimensões divergem da célula, a execução falha explicitamente. Para B3,
espera artefatos experimentais separados dos arquivos de produção. A geração
das unidades e o retreino B3 continuam pendentes; portanto o adaptador ainda
não autoriza executar a matriz piloto como estudo completo.

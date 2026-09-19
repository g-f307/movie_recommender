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
e artefatos locais. Dados e resultados brutos permanecem fora do Git.
Presença de arquivo **não** comprova que B3 foi treinado no split correto.

Antes de executar o piloto faltam o runner real B0–B5, o artefato B2 e a
validação/congelamento do B3. Antes da execução oficial também são necessárias
a revisão do orientador, a decisão explícita sobre B6 e a verificação de
completude, retomada e isolamento entre saídas piloto e oficiais. Nenhuma tag
experimental ou resultado confirmatório deve ser criado a partir deste
pré-voo.

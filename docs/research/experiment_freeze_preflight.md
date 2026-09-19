# Piloto e congelamento experimental v1

O escopo v1 segue o roadmap: B0–B5 são obrigatórios; B6 foi adiado por não
possuir hipótese nem mecanismo distinto de B5 especificado antes do holdout.
Não houve revisão do orientador. Isso deve constar como limitação do artigo,
sem ser confundido com aprovação.

## Insumos e isolamento

O split por `movie_id` (60/20/20, seed 42) deriva do dataset local e é
registrado em `results/manifests/splits/`. B2 é ajustado apenas com filmes da
partição de treino. B3 usa uma amostra determinística, independente de rótulo,
de 1/32 dos contextos: fit somente em treino, seleção de família,
hiperparâmetros e threshold somente em validação. O artefato experimental
`artifacts/b3_experiment_v1.*` não substitui o modelo de produção. A amostragem
é limitação a declarar; o dataset contém majoritariamente rótulos proxy.

O catálogo de avaliação em `results/derived/test_catalog.json` contém
exatamente os IDs do holdout, sem duplicatas. As unidades em `results/units_v1_1/`
são pareadas por `comparison_id`, com o mesmo agente, estado, candidatos e
julgamentos para todos os métodos. A política `neutral-exposure-v1` escolhe
filmes apresentados por uma permutação determinística independente do
recomendador; não usa ranking de B5 para gerar o histórico. A relevância é
produzida pelo agente sintético para cada candidato elegível. Portanto,
**não equivale a relevância humana observada**.

## Reprodução local

```bash
python -m cinebot_ml.experimental_splits
python -m cinebot_ml.artifacts build-b2 \
  --split-manifest results/manifests/splits/movie_id_split.json
python -m cinebot_ml.experiments.train_b3
python -m cinebot_ml.experiments.units --pilot
python -m cinebot_ml.experiments.freeze --output /tmp/movie-freeze-preflight.json
```

O piloto contém B0–B5, C0/C2/C4, P0/P4/P5, personas `consistent` e `noisy`,
seeds 42/137 e K=5/10: 432 células e 72 unidades pareadas. A execução é:

```python
from pathlib import Path
from cinebot_ml.experiments.freeze import pilot_matrix, audit_execution
from cinebot_ml.experiments.matrix import execute_matrix
from cinebot_ml.experiments.cell_runner import run_cell

matrix = pilot_matrix()
report = execute_matrix(matrix, Path("results/raw/pilot"), run_cell)
audit = audit_execution(matrix, Path("results/raw/pilot"))
assert audit["complete"]
```

Uma segunda execução não sobrescreve células já existentes. A auditoria lê
cada checkpoint, confere identidade, status, avaliação e NDCG@K, e classifica
falhas. O piloto mede viabilidade operacional; não decide hipóteses.

Na execução local, 432/432 células terminaram sem falhas em cerca de 34
segundos. Os checkpoints individuais somaram 6.255.230 bytes. Uma retomada
marcou 432 células como `skipped` e preservou o hash de um checkpoint
inspecionado. Extrapolação linear para 15.120 células: aproximadamente 20
minutos e 219 MB de checkpoints; a execução real pode divergir e também terá
manifestos e unidades auxiliares.

Os artefatos grandes, unidades e resultados brutos são ignorados pelo Git.
O manifesto final deve registrar hashes, commit e dimensões. Saídas oficiais
devem residir em `results/raw/official_v1_1/`, separadas das do piloto, e somente
ser congeladas após auditoria integral. A Etapa 5 deve consumir esses arquivos
sem reexecutar os modelos.

Na versão corrigida `1.0.1-holdout`, a saída oficial válida reside em
`results/raw/official_v1_1/`. Após a conclusão integral, execute
`python -m cinebot_ml.experiments.finalize`. O comando recusa qualquer falha,
ausência, divergência de pareamento ou de commit; escreve o manifesto global
com hashes de cada unidade e célula e torna resultados e artefatos gerados
somente leitura. O diretório `results/raw/official/` da tentativa interrompida
não deve ser publicado como resultado.

A matriz corrigida concluiu localmente 15.120/15.120 células em 20min31s.
A auditoria integral encontrou zero falhas, ausência de métricas ou divergências
de pareamento e commit. Os arquivos individuais totalizaram 227.336.316 bytes
(aproximadamente 254 MB ocupados em disco, incluindo manifestos e metadados).
Esses números descrevem a execução, não o efeito científico dos métodos.

O repositório não possui remoto DVC configurado. Os hashes tornam os artefatos
locais verificáveis, mas um clone limpo ainda precisa recuperar ou reconstruir
dataset, modelos e resultados por um canal autorizado. Não se deve declarar
reprodução independente completa até essa distribuição ser resolvida.

Uma primeira execução parcial foi interrompida ao detectar que o gerador de
unidades tratava C5 como 10, não 15 eventos. Ela permanece em
`results/raw/official/` apenas para auditoria de falha operacional. A versão
`1.0.1-holdout` corrige C5, usa novas unidades e uma nova identidade de matriz;
nenhum checkpoint da tentativa anterior será reutilizado.

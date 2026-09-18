# Matriz experimental v1

## Objetivo

`cinebot_ml.experiments.matrix` enumera e orquestra a matriz usada pelos estudos
da Etapa 4. O módulo não interpreta resultados: ele garante identidade,
pareamento, execução sequencial, checkpoints, retomada e registro de falhas.

## Dimensões

A matriz deriva métodos, condições, perfis, seeds e valores de K de
`configs/experiment_v1.yaml`, e personas de
`configs/agents/personas_v1.yaml`. A política complementar está congelada em
`configs/experiments/matrix_v1.yaml`.

Na configuração v1, antes de exclusões, existem:

`6 métodos × 6 condições × 6 perfis × 7 personas × 5 seeds × 2 K = 15.120 células`.

Todos os componentes atuais são determinísticos quando a seed é fixada e, por
isso, possuem somente `run=1`. Se um método futuro for removido da lista
`deterministic_methods`, ele receberá o número de repetições estocásticas da
configuração experimental.

## Identidade e comparação pareada

`ExperimentCell` contém método, condição, perfil, persona, seed, K e repetição.
Seu `cell_id` é derivado canonicamente desses campos. O `comparison_id` usa os
mesmos campos sem o método; assim, métodos com o mesmo identificador de
comparação devem compartilhar agente, estado e conjunto candidato no runner do
estudo.

O `matrix_id` inclui hashes das três configurações e o conjunto ordenado de IDs
das células. Alterar a ordem de entrada não altera as identidades.

## Exclusões

Combinações inválidas devem ser declaradas em `exclusions`, com campos em
`when` e um `reason`. Valores não habilitados são rejeitados. Células excluídas
e seus motivos permanecem no manifesto global; não desaparecem silenciosamente.

## Execução e retomada

O executor recebe um `CellRunner`, uma função `runner(cell) -> mapping`. Cada
retorno é persistido atomicamente em:

```text
<output>/<matrix_id>/
├── matrix.manifest.json
├── matrix.status.json
└── cells/<cell_id>.json
```

Uma exceção comum registra a célula como `failed` e não interrompe as demais.
Uma interrupção do processo preserva os checkpoints já concluídos. Na retomada,
qualquer célula com arquivo existente é ignorada e nunca sobrescrita, inclusive
falhas, preservando a evidência original.

## Comandos

Listar a matriz completa sem executá-la:

```bash
python -m cinebot_ml.experiments list > /tmp/matrix-v1.json
```

Listar uma matriz reduzida:

```bash
python -m cinebot_ml.experiments list \
  --method B0 --condition C0 --profile P0 \
  --persona consistent --seed 42 --k 5
```

Executar exige um runner fornecido pelo estudo no formato `modulo:funcao`:

```bash
python -m cinebot_ml.experiments run \
  --runner pacote.do_estudo:run_cell \
  --config configs/experiment_v1.yaml \
  --output results/raw/matrices
```

Essa separação impede que o orquestrador finja produzir evidência sem executar
o protocolo específico. As próximas implementações de cold start, convergência,
ablação e robustez fornecerão runners compatíveis.

## Limitações

A enumeração não executa o experimento oficial nem valida a disponibilidade dos
artefatos B2/B3. Isso cabe ao runner e ao congelamento da Etapa 4. A matriz não
realiza análise estatística, gráficos ou seleção de parâmetros.

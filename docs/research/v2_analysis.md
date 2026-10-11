# Análise comparativa do estudo complementar v2

## Finalidade

`cinebot_ml.analysis.v2_analysis` analisa a execução complementar v2 depois que
ela foi selada por `cinebot_ml.experiments.finalize`. O módulo não altera os
resultados oficiais v1.1, não cria células e não transforma trajetórias
sintéticas em evidência humana.

O objetivo é descrever o efeito pareado B5 − B4 em NDCG@5 sob três mudanças de
desenho: perfil declarado com ruído, exposição em ciclo fechado e política
incremental escolhida apenas na validação v2.

## Entradas e verificações

O comando recebe o `execution.manifest.json` v2. Antes de calcular qualquer
estatística, ele verifica:

- o status completo, a identidade da execução (`execution-v2` ou
  `execution-v2.1`) e o hash da configuração;
- o hash e a identidade de `matrix.manifest.json`;
- o conjunto exato e os hashes das células;
- a correspondência entre cada célula, sua execução, seu resultado e seu
  cenário de política;
- a política de exposição compatível, o estado causal da trajetória, o conjunto
  candidato e o pareamento B4/B5; e
- a separação entre `validation` e `holdout`.

Células de v1.1, uma configuração diferente, pares incompletos, métricas
ausentes e resultados com status diferente de `evaluated` não são usados como
pares válidos. Esses últimos dois casos continuam contabilizados no relatório;
não são imputados.

## Seleção e holdout

Em uma execução científica, use `--selection` com o JSON e lock produzidos por
`freeze_validation_selection`. O carregador confirma a matriz de sensibilidade,
o hash do cenário, a execução compatível e a ausência de holdout durante a
seleção. Em v2.1, também confirma que os cenários comparados partiram do mesmo
agente, seed, candidato, relevância, estado e hash de trajetória. O holdout é
rejeitado se contiver qualquer cenário diferente daquele congelado.

`--selected-scenario` existe somente para inspeção programática de artefatos
intermediários. Ele não substitui a seleção congelada ao gerar evidência para o
artigo.

## Unidade de análise e inferência

Cada `comparison_id` forma um par B4/B5. Observações repetidas de um mesmo
agente são agregadas antes da inferência; uma interação individual nunca é
tratada como agente independente. O pipeline reporta, para cada partição e
para os recortes por cenário, condição, perfil, persona, seed e nível de ruído:

- número de pares e de agentes independentes;
- médias B4 e B5, diferença média e mediana;
- IC bootstrap pareado de 95%;
- Wilcoxon bicaudal e correlação rank-biserial; e
- valores-p ajustados por Holm por família `partição × dimensão`.

As análises v2 são exploratórias e complementares. A validação continua sendo
uma etapa de seleção; o holdout v2 é relatado após a seleção, mas não substitui
a execução oficial v1.1.

## Comparação v1.1 × v2

`--v1-manifest` permite incluir o resumo oficial v1.1 no mesmo relatório como
uma comparação de desenhos. Os dois efeitos são mantidos em blocos separados:
a diferença entre exposição neutra/perfil alinhado à preferência latente (v1.1)
e ciclo fechado/perfil declarado ruidoso/seleção em validação (v2) é registrada,
mas o pipeline proíbe *pooling* e qualquer meta-análise implícita.

Sem esse argumento, o relatório marca a fonte v1.1 como não fornecida, em vez
de abrir arquivos antigos ou inferir uma comparação.

## Reprodução

Depois de finalizar a matriz v2 e congelar a seleção, execute:

```bash
.venv/bin/python -m cinebot_ml.analysis.v2_analysis \
  --manifest results/v2_1/revision_2/manifests/executions/<matrix_id>/execution.manifest.json \
  --selection results/v2_1/revision_2/reports/validation/<selection_id>/selection.json \
  --v1-manifest results/raw/official_v1_1/2942e51456add29e4c999307/experiment.manifest.json \
  --output-dir results/v2_1/revision_2/reports/<matrix_id>/analysis
```

O destino deve ser novo. A escrita é atômica e recusa sobrescrever resultados.
Ela produz o relatório JSON, as tabelas CSV de pares e subgrupos, três figuras
em PDF e PNG, e `analysis.manifest.json` com hashes de todas as saídas.

## Limites atuais

O repositório não versiona uma execução v2 finalizada nesta etapa. Portanto,
a implementação demonstra contratos, validações e geração de evidência a partir
de fixtures controladas, mas não declara efeito empírico novo. A execução
selada, suas células e seus artefatos devem ser recuperados pelo fluxo
experimental antes que o relatório seja usado no manuscrito.

# Estudo longitudinal de convergência C0–C5

O estudo preserva cada agente como unidade longitudinal e compara B4 e B5 nos
checkpoints oficiais de 0, 1, 3, 5, 10 e 15 eventos. Interações do mesmo agente
não são tratadas como amostras independentes.

## Construção da trajetória

Cada condição é executada com o mesmo `agent_id`, `preference_id`, seed, perfil
inicial e catálogo. Apenas o prefixo causal de feedback muda. O ranking do
checkpoint é calculado antes de qualquer evento posterior. B4 recebe sempre o
perfil inicial imutável; B5 recebe o snapshot permitido pela condição.

Para cada condição são registrados `target_events`, `actual_events` e
`sequence_status`. Condições falhas permanecem com métricas nulas. Catálogos
esgotados preservam a quantidade observada e não recebem eventos artificiais.
Quando C5 alcança o mesmo estado de C2, C3 ou C4, o campo
`coincident_with_previous` registra a coincidência.

## Medidas por agente

Para cada métrica configurada e para cada método são calculados:

- valor no checkpoint;
- ganho absoluto contra C0: `m(Cx) - m(C0)`;
- ganho pareado de B5 contra B4: `m(B5,Cx) - m(B4,Cx)`;
- valor marginal: diferença para o checkpoint anterior disponível;
- estabilidade do ranking: Jaccard Top-K entre checkpoints sucessivos;
- estabilidade do perfil: invariância do estado em B4 e igualdade do snapshot
  sucessivo em B5.

Ganhos podem ser positivos, negativos ou nulos. Se qualquer termo estiver
ausente, o ganho permanece nulo.

## Artefatos

`write_convergence_report` produz, sem sobrescrita:

```text
<output>/<study_id>/
├── convergence.manifest.json
├── convergence.longitudinal.csv
└── convergence.curves.csv
```

O arquivo longitudinal é o dado bruto para análise por agente. O arquivo de
curvas é uma projeção regenerável com eixo de eventos, valor, ganhos e valor
marginal; não é uma análise confirmatória nem escolhe o melhor checkpoint.

## Limitações

As curvas são descritivas e sintéticas. O teste confirmatório de H2, intervalos
de incerteza e correção da dependência intra-agente pertencem à etapa de análise
estatística posterior.

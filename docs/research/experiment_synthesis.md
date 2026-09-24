# Síntese dos estudos especializados

Esta etapa consolida as evidências exploratórias de cold start (P0–P5) e convergência (C0–C5) presentes na execução oficial, preservando o agente sintético como unidade longitudinal. A análise confirmatória B5 × B4 permanece no relatório estatístico próprio; não é misturada às explorações desta etapa.

## Execução

```bash
python -m cinebot_ml.analysis.synthesis
```

O comando valida novamente o manifesto e os hashes das 15.120 células antes de produzir, sem sobrescrever arquivos:

- `results/reports/experiment_synthesis_v1/synthesis.manifest.json`;
- `results/reports/experiment_synthesis_v1/cold_start.csv`;
- `results/reports/experiment_synthesis_v1/convergence.csv`.

Na convergência, o teste global de Friedman compara C0–C5 dentro dos mesmos agentes. Comparações pós-hoc contra C0 são executadas somente quando o teste global rejeita a hipótese nula e recebem correção de Holm.

## Ablação e robustez

As configurações congeladas A0–A6 e dos 12 cenários são auditadas e identificadas por hash. Os estudos foram executados separadamente na partição de validação e são descobertos pela síntese nos diretórios:

- `results/derived/specialized_v1/ablation/538fae9a59ad7976c95c6ef8/`;
- `results/derived/specialized_v1/robustness/56e31e9e624c565ed60a3a64/`.

A ablação contém 1.225 células válidas. A robustez contém 417 pares B4×B5 e três falhas explícitas por NDCG@5 indefinido no cenário com poucos candidatos. Esses estudos permanecem exploratórios, não reinterpretam o holdout e não são usados para ajustar B5. Somente manifestos versionados passam a ser referenciados pela síntese; ausências nunca são imputadas.

## Limites de interpretação

Todos os resultados especializados são exploratórios. O holdout oficial não pode ser usado para escolher variantes, cenários ou hiperparâmetros, e evidência de agentes sintéticos não demonstra generalização para pessoas reais.

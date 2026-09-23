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

As configurações congeladas A0–A6 e dos 12 cenários são auditadas e identificadas por hash. Como esses estudos não fazem parte das células oficiais, a síntese registra sua indisponibilidade, em vez de reinterpretar o holdout como se ele fosse uma execução de ablação ou robustez. Saídas válidas futuras devem ser gravadas separadamente em:

- `results/derived/specialized_v1/ablation/<study_id>/`;
- `results/derived/specialized_v1/robustness/<study_id>/`.

Somente manifestações versionadas nesses diretórios passam a ser referenciadas pela síntese. Falhas e métricas indisponíveis devem permanecer explícitas. Em particular, novidade e viés de popularidade não foram calculados na execução oficial e não são imputados.

## Limites de interpretação

Todos os resultados especializados são exploratórios. O holdout oficial não pode ser usado para escolher variantes, cenários ou hiperparâmetros, e evidência de agentes sintéticos não demonstra generalização para pessoas reais.

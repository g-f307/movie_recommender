# Estudo exploratório de robustez

`configs/experiments/robustness_v1.yaml` congela 12 cenários, incluindo a
referência nominal, feedback aleatório/contraditório/negativo extremo,
preferências específicas/amplas, escassez de candidatos, gênero raro,
concentração popular/cauda longa, metadados ausentes e mudança de catálogo.
Cada cenário registra intensidade, versão, seed e identidade determinística.

`run_robustness` percorre as cinco seeds oficiais em todos os cenários e
entrega um `RobustnessCase` a um runner do benchmark. O método não é modificado;
somente dados de entrada e feedback são perturbados. Cada resultado mantém
`evidence=exploratory_synthetic`. Falhas ficam como linhas com erro explícito,
sem interromper as outras execuções. A comparação nominal/adversa é pareada por
agente e seed; não constitui teste confirmatório nem drift de produção.

O runner recebe `(case)` e devolve um mapeamento de métricas, por exemplo as
métricas B5 do benchmark unificado. A execução oficial e seus artefatos
permanecem dependentes da preparação de dados e modelos congelados.

Para persistir todas as linhas, inclusive falhas, use
`write_robustness_report(rows, caminho_jsonl)` após `run_robustness(...)`.
Os resultados não devem ser apresentados como inferência sobre usuários reais.

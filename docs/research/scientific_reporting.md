# Geração dos artefatos científicos

O pacote de tabelas, figuras e metadados usados no artigo é regenerado diretamente da execução oficial congelada por um único comando:

```bash
python -m cinebot_ml.analysis.reporting
```

Por padrão, as saídas são criadas em `results/scientific_report_v1/`, separadas em `tables/`, `figures/` e `reports/`. O destino precisa estar vazio: nenhuma tabela, figura ou relatório anterior é sobrescrito silenciosamente.

As tabelas incluem a visão geral B0–B5, a comparação principal B5 × B4, subgrupos, testes estatísticos, cold start e convergência. As figuras apresentam a curva C0–C5 e os perfis P0–P5. Ablação e robustez possuem manifestos exploratórios próprios na partição de validação, incluindo intervalos, falhas explícitas e figuras versionadas; ausência nunca é convertida em zero.

Todos os títulos identificam agentes sintéticos, NDCG@5 e as dimensões comparadas. `reports/scientific_report.json` contém o hash do manifesto, o commit, a identidade da matriz, os relatórios estatísticos de origem e o inventário das saídas.

Para gerar em outro diretório:

```bash
python -m cinebot_ml.analysis.reporting --output-root /tmp/scientific-report
```

Os arquivos são derivados e permanecem fora do Git. O código, as configurações e a documentação de geração são versionados.

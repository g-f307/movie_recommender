# Artefatos editoriais reproduzíveis

A issue #79 consolida figuras e tabelas do artigo exclusivamente a partir das evidências estruturadas e congeladas. O pipeline não recalcula a matriz experimental nem altera resultados de origem.

## Geração

Recupere previamente os resultados versionados por DVC e execute:

```bash
make paper-artifacts PYTHON=.venv/bin/python
```

O destino padrão é `results/paper_artifacts_v1`. Para evitar sobrescrita acidental, o comando rejeita um diretório não vazio. Um destino alternativo pode ser informado sem editar a configuração:

```bash
OUTPUT_ROOT=/tmp/movie-paper-artifacts make paper-artifacts PYTHON=.venv/bin/python
```

## Saídas

- `figures/primary_paired.{pdf,png}`: diferenças B5−B4 por agente, média e IC95%;
- `figures/feedback_trajectory.{pdf,png}`: evolução descritiva C0–C5 para B4 e B5;
- `figures/mechanisms_forest.{pdf,png}`: ablações A1–A6 e cenários de robustez;
- `figures/transportability.{pdf,png}`: estimativas sintética e MovieLens apresentadas separadamente;
- `tables/*.tex`: fragmentos compatíveis com `acmart`;
- `paper_artifacts.manifest.json`: matriz, fontes, hashes e tamanhos das saídas.

Os PNGs são exportados em 300 DPI e os PDFs permanecem vetoriais. A paleta combina cor e marcadores distintos. O gráfico C0–C5 é descritivo; inferências e decisões devem ser lidas em `scientific_conclusions.md`.

## Rastreabilidade

Cada fonte registra caminho absoluto, SHA-256 e tamanho. A identidade da matriz é verificada entre manifesto oficial, análise principal, síntese e contrastes. Fonte ausente, identidade incompatível ou destino preenchido interrompem a geração. Os resultados sintético e público não são combinados em uma estimativa única.

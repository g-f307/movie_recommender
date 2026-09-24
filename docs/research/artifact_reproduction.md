# Recuperação dos artefatos científicos

## Escopo efetivamente reproduzível

O inventário versionado em `configs/artifact_inventory_v1.json` separa ativos públicos, restritos e reconstruíveis. A publicação DVC inclui somente:

- artefato B2 TF-IDF;
- modelo e metadados B3 da versão experimental;
- execução oficial sintética v1.1, com 15.120 células;
- 2.520 unidades experimentais sintéticas v1.1.

Não são publicados `.env`, credenciais, feedback real, catálogo sem autorização, dataset cartesiano derivado ou features de treinamento derivadas. O ponteiro legado `datasets/movie_preferences.csv.dvc` não representa o arquivo local atual e não deve ser usado como evidência da execução oficial v1.1.

## Configuração do remoto

O remoto privado `artifacts` está versionado sem credenciais em `.dvc/config` e aponta para o bucket Cloudflare R2 `movie-recommender-dvc`. Em um clone novo, forneça as credenciais apenas pelo ambiente:

```bash
export AWS_ACCESS_KEY_ID=<ACCESS_KEY_LOCAL>
export AWS_SECRET_ACCESS_KEY=<SECRET_KEY_LOCAL>
export AWS_DEFAULT_REGION=auto
```

As variáveis podem ser carregadas de um `.env` local ignorado pelo Git. Tokens e chaves nunca pertencem a `.dvc/config`, manifests, comandos versionados ou logs.

Para publicar os objetos aprovados:

```bash
.venv/bin/dvc push -r artifacts \
  artifacts/b2_tfidf_v1.json.dvc \
  artifacts/b3_experiment_v1.joblib.dvc \
  artifacts/b3_experiment_v1.json.dvc \
  results/raw/official_v1_1.dvc \
  results/units_v1_1.dvc
```

## Clone independente

Em um clone novo, após instalar `requirements-ci.txt` e `requirements-ml.txt` e configurar o mesmo remoto:

```bash
.venv/bin/dvc pull -r artifacts \
  artifacts/b2_tfidf_v1.json.dvc \
  artifacts/b3_experiment_v1.joblib.dvc \
  artifacts/b3_experiment_v1.json.dvc \
  results/raw/official_v1_1.dvc \
  results/units_v1_1.dvc

.venv/bin/python -m cinebot_ml.reproducibility \
  --output results/reports/reproduction.json

.venv/bin/python -m cinebot_ml.analysis.reporting \
  --output-root results/scientific_report_v1
```

O verificador confere tamanho e SHA-256 dos arquivos, MD5 dos ponteiros DVC, contagem de arquivos dos diretórios, hash do manifesto oficial e os hashes das 15.120 células. A geração científica valida novamente a matriz antes de produzir as tabelas e figuras.

## Limitação de distribuição

O fluxo foi validado primeiro com um remoto local descartável e depois com o bucket privado Cloudflare R2. Um clone novo obtido do GitHub recuperou 17.648 objetos exclusivamente do R2, validou os cinco ativos e as 15.120 células e regenerou tabelas e figuras com hashes idênticos. Terceiros ainda precisam receber credenciais de leitura autorizadas; o bucket não é público.

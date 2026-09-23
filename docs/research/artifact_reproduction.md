# Recuperação dos artefatos científicos

## Escopo efetivamente reproduzível

O inventário versionado em `configs/artifact_inventory_v1.json` separa ativos públicos, restritos e reconstruíveis. A publicação DVC inclui somente:

- artefato B2 TF-IDF;
- modelo e metadados B3 da versão experimental;
- execução oficial sintética v1.1, com 15.120 células;
- 2.520 unidades experimentais sintéticas v1.1.

Não são publicados `.env`, credenciais, feedback real, catálogo sem autorização, dataset cartesiano derivado ou features de treinamento derivadas. O ponteiro legado `datasets/movie_preferences.csv.dvc` não representa o arquivo local atual e não deve ser usado como evidência da execução oficial v1.1.

## Configuração do remoto

O endereço do armazenamento é infraestrutura e não deve conter credenciais no Git. Após receber do responsável o endereço autorizado, configure-o localmente:

```bash
.venv/bin/dvc remote add --local -d artifacts <URL_AUTORIZADA>
```

Credenciais específicas do backend devem ser definidas com `dvc remote modify --local`, variáveis de ambiente ou o mecanismo de identidade do provedor. A opção `--local` grava em `.dvc/config.local`, que é ignorado pelo Git. Tokens, chaves e caminhos pessoais não pertencem a `.dvc/config`.

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

O fluxo foi validado de ponta a ponta com um remoto DVC local descartável e um clone Git independente. Isso demonstra que os ponteiros, objetos, recuperação e regeneração funcionam, mas não equivale à disponibilidade pública duradoura. A alegação de recuperação por terceiros só passa a valer quando o proprietário configurar um backend compartilhado autorizado e executar o `dvc push` acima.
